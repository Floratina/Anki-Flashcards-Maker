use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::domain::{ProviderCallError, ProviderRuntimeConfig};

pub const DEFAULT_BASE_URL: &str = "https://aiplatform.googleapis.com";
pub const DEFAULT_LOCATION: &str = "global";

const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const JWT_AUDIENCE: &str = "https://oauth2.googleapis.com/token";
const JWT_LIFETIME_SECONDS: u64 = 3600;
const TOKEN_EXPIRY_SKEW_SECONDS: u64 = 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceAccountImport {
    pub project_id: String,
    pub client_email: String,
    pub private_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexRuntime {
    pub project_id: String,
    pub location: String,
    pub client_email: String,
    pub private_key: String,
}

#[derive(Debug, Clone)]
struct CachedToken {
    access_token: String,
    expires_at: u64,
}

#[derive(Debug, Serialize)]
struct JwtClaims<'a> {
    iss: &'a str,
    scope: &'a str,
    aud: &'a str,
    iat: u64,
    exp: u64,
}

#[derive(Debug, Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    expires_in: Option<u64>,
}

static TOKEN_CACHE: OnceLock<Mutex<HashMap<String, CachedToken>>> = OnceLock::new();

fn token_cache() -> &'static Mutex<HashMap<String, CachedToken>> {
    TOKEN_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
pub fn default_config() -> crate::domain::VertexConfig {
    crate::domain::VertexConfig {
        project_id: String::new(),
        location: DEFAULT_LOCATION.to_string(),
        client_email: String::new(),
    }
}

pub fn parse_service_account_json(value: &str) -> Result<ServiceAccountImport, String> {
    let trimmed = value.trim().trim_start_matches('\u{feff}');
    if trimmed.is_empty() {
        return Err("Service Account JSON is required".into());
    }
    let value: Value = serde_json::from_str(trimmed)
        .map_err(|error| format!("Service Account JSON is invalid: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Service Account JSON must be an object".to_string())?;
    let private_key = object
        .get("private_key")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Service Account JSON is missing private_key".to_string())?;
    let client_email = object
        .get("client_email")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Service Account JSON is missing client_email".to_string())?;
    let project_id = object
        .get("project_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_default();
    Ok(ServiceAccountImport {
        project_id: project_id.to_string(),
        client_email: client_email.to_string(),
        private_key: private_key.to_string(),
    })
}

pub fn format_private_key(private_key: &str) -> Result<String, String> {
    let mut key = private_key.trim().replace("\\n", "\n");
    if key.is_empty() {
        return Err("Private key is required".into());
    }
    if key.contains("-----BEGIN PRIVATE KEY-----") && key.contains("-----END PRIVATE KEY-----") {
        return Ok(key);
    }
    key = key
        .replace("-----BEGIN PRIVATE KEY-----", "")
        .replace("-----END PRIVATE KEY-----", "")
        .split_whitespace()
        .collect::<String>();
    if key.is_empty() {
        return Err("Private key is empty after formatting".into());
    }
    let mut lines = Vec::new();
    let mut index = 0;
    while index < key.len() {
        let next = (index + 64).min(key.len());
        lines.push(&key[index..next]);
        index = next;
    }
    Ok(format!(
        "-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----",
        lines.join("\n")
    ))
}

pub fn runtime_config(config: &ProviderRuntimeConfig) -> Result<VertexRuntime, String> {
    let project_id = required(&config.vertex.project_id, "Project ID")?;
    let client_email = required(&config.vertex.client_email, "Client email")?;
    let location = config
        .vertex
        .location
        .trim()
        .to_string()
        .if_empty(DEFAULT_LOCATION);
    let private_key = config
        .credential
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Private key is required".to_string())?
        .to_string();
    Ok(VertexRuntime {
        project_id,
        location,
        client_email,
        private_key,
    })
}

fn required(value: &str, label: &str) -> Result<String, String> {
    value
        .trim()
        .is_empty()
        .then(|| format!("{label} is required"))
        .map_or_else(|| Ok(value.trim().to_string()), Err)
}

trait EmptyDefault {
    fn if_empty(self, default: &str) -> String;
}

impl EmptyDefault for String {
    fn if_empty(self, default: &str) -> String {
        if self.is_empty() {
            default.to_string()
        } else {
            self
        }
    }
}

pub fn service_base_url(base_url: &str, project_id: &str, location: &str) -> String {
    let base = endpoint_base_url(base_url).trim_end_matches('/');
    let location = location.trim();
    if has_vertex_resource_path(base) {
        return base
            .trim_end_matches("/publishers/google")
            .trim_end_matches("/publishers/google/models")
            .to_string();
    }
    let host = if base.is_empty() || is_default_aiplatform_base(base) {
        if location == DEFAULT_LOCATION {
            DEFAULT_BASE_URL.to_string()
        } else {
            format!("https://{location}-aiplatform.googleapis.com")
        }
    } else {
        base.to_string()
    };
    let host = host.trim_end_matches('/');
    let host = host.trim_end_matches("/v1").trim_end_matches("/v1beta1");
    format!("{host}/v1/projects/{project_id}/locations/{location}")
}

pub fn service_endpoint(base_url: &str, location: &str) -> String {
    let base = endpoint_base_url(base_url).trim_end_matches('/');
    let location = location.trim();
    if base.is_empty() || is_default_aiplatform_base(base) {
        return if location == DEFAULT_LOCATION {
            DEFAULT_BASE_URL.to_string()
        } else {
            format!("https://{location}-aiplatform.googleapis.com")
        };
    }
    strip_vertex_resource_path(base)
        .trim_end_matches("/v1")
        .trim_end_matches("/v1beta1")
        .to_string()
}

pub fn model_id(model: &str) -> String {
    let trimmed = model.trim();
    if let Some(index) = trimmed.rfind("/models/") {
        return trimmed[index + "/models/".len()..].to_string();
    }
    trimmed.trim_start_matches("models/").to_string()
}

pub fn generate_content_url(
    base_url: &str,
    project_id: &str,
    location: &str,
    model: &str,
) -> String {
    format!(
        "{}/publishers/google/models/{}:generateContent",
        service_base_url(base_url, project_id, location).trim_end_matches('/'),
        model_id(model),
    )
}

pub fn publisher_models_url(base_url: &str, location: &str, publisher: &str) -> String {
    format!(
        "{}/v1beta1/publishers/{}/models",
        service_endpoint(base_url, location).trim_end_matches('/'),
        publisher.trim_matches('/')
    )
}

fn endpoint_base_url(base_url: &str) -> &str {
    base_url.split('#').next().unwrap_or(base_url).trim()
}

fn is_default_aiplatform_base(base_url: &str) -> bool {
    url::Url::parse(base_url).ok().is_some_and(|url| {
        let host = url.host_str().unwrap_or_default();
        let path = url.path().trim_end_matches('/');
        host == "aiplatform.googleapis.com"
            && (path.is_empty() || path == "/v1" || path == "/v1beta1")
    })
}

fn has_vertex_resource_path(base_url: &str) -> bool {
    let path = url::Url::parse(base_url)
        .ok()
        .map(|url| url.path().trim_end_matches('/').to_string())
        .unwrap_or_else(|| base_url.trim_end_matches('/').to_string());
    path.contains("/projects/") && path.contains("/locations/")
}

fn strip_vertex_resource_path(base_url: &str) -> &str {
    for marker in ["/v1/projects/", "/v1beta1/projects/"] {
        if let Some(index) = base_url.find(marker) {
            return &base_url[..index];
        }
    }
    base_url
}

fn cache_key(config: &VertexRuntime) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    config.project_id.hash(&mut hasher);
    config.location.hash(&mut hasher);
    config.client_email.hash(&mut hasher);
    config.private_key.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

pub async fn access_token(
    client: &Client,
    config: &ProviderRuntimeConfig,
) -> Result<String, ProviderCallError> {
    let vertex = runtime_config(config).map_err(ProviderCallError::fatal)?;
    access_token_for_service_account(client, &vertex).await
}

async fn access_token_for_service_account(
    client: &Client,
    config: &VertexRuntime,
) -> Result<String, ProviderCallError> {
    let now = now_seconds();
    let key = cache_key(config);
    {
        let cache = token_cache().lock().await;
        if let Some(cached) = cache.get(&key) {
            if cached.expires_at > now + TOKEN_EXPIRY_SKEW_SECONDS {
                return Ok(cached.access_token.clone());
            }
        }
    }

    let formatted_key =
        format_private_key(&config.private_key).map_err(ProviderCallError::fatal)?;
    let claims = JwtClaims {
        iss: &config.client_email,
        scope: CLOUD_PLATFORM_SCOPE,
        aud: JWT_AUDIENCE,
        iat: now,
        exp: now + JWT_LIFETIME_SECONDS,
    };
    let jwt = encode(
        &Header::new(Algorithm::RS256),
        &claims,
        &EncodingKey::from_rsa_pem(formatted_key.as_bytes()).map_err(|error| {
            ProviderCallError::fatal(format!("Invalid service account private key: {error}"))
        })?,
    )
    .map_err(|error| {
        ProviderCallError::fatal(format!("Unable to sign service account JWT: {error}"))
    })?;

    let form_body = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer")
        .append_pair("assertion", &jwt)
        .finish();
    let response = client
        .post(TOKEN_URL)
        .header(
            reqwest::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        )
        .body(form_body)
        .send()
        .await
        .map_err(|error| {
            retryable_reqwest_error(error, "Unable to request Agent Platform access token")
        })?;
    let status = response.status();
    let text = response.text().await.map_err(|error| {
        retryable_reqwest_error(error, "Unable to read Agent Platform token response")
    })?;
    if !status.is_success() {
        return Err(ProviderCallError::http(
            status.as_u16(),
            format!(
                "Agent Platform token request failed: HTTP {}: {}",
                status.as_u16(),
                text.chars().take(500).collect::<String>()
            ),
        ));
    }
    let token: OAuthTokenResponse = serde_json::from_str(&text).map_err(|error| {
        ProviderCallError::fatal(format!("Invalid Agent Platform token response: {error}"))
    })?;
    if token.access_token.trim().is_empty() {
        return Err(ProviderCallError::fatal(
            "Agent Platform token response did not include access_token",
        ));
    }
    let expires_at = now + token.expires_in.unwrap_or(JWT_LIFETIME_SECONDS);
    token_cache().lock().await.insert(
        key,
        CachedToken {
            access_token: token.access_token.clone(),
            expires_at,
        },
    );
    Ok(token.access_token)
}

fn retryable_reqwest_error(error: reqwest::Error, context: &str) -> ProviderCallError {
    let message = format!("{context}: {error}");
    if let Some(status) = error.status() {
        ProviderCallError::http(status.as_u16(), message)
    } else if error.is_timeout() || error.is_connect() || error.is_request() || error.is_body() {
        ProviderCallError::retryable(message)
    } else {
        ProviderCallError::fatal(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_service_account_json_and_formats_key() {
        let parsed = parse_service_account_json(
            r#"{
                "project_id": "vertex-project",
                "client_email": "svc@vertex-project.iam.gserviceaccount.com",
                "private_key": "-----BEGIN PRIVATE KEY-----\\nabc\\n-----END PRIVATE KEY-----\\n"
            }"#,
        )
        .expect("parse");
        assert_eq!(parsed.project_id, "vertex-project");
        assert_eq!(
            parsed.client_email,
            "svc@vertex-project.iam.gserviceaccount.com"
        );
        let formatted = format_private_key(&parsed.private_key).expect("format");
        assert!(formatted.contains("-----BEGIN PRIVATE KEY-----\nabc"));
    }

    #[test]
    fn builds_agent_platform_resource_urls() {
        assert_eq!(
            service_base_url(DEFAULT_BASE_URL, "project-1", DEFAULT_LOCATION),
            "https://aiplatform.googleapis.com/v1/projects/project-1/locations/global"
        );
        assert_eq!(
            service_base_url(DEFAULT_BASE_URL, "project-1", "us-central1"),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/project-1/locations/us-central1"
        );
        assert_eq!(
            publisher_models_url(DEFAULT_BASE_URL, "us-central1", "google"),
            "https://us-central1-aiplatform.googleapis.com/v1beta1/publishers/google/models"
        );
        assert_eq!(
            generate_content_url(
                DEFAULT_BASE_URL,
                "project-1",
                DEFAULT_LOCATION,
                "models/gemini-2.5-flash"
            ),
            "https://aiplatform.googleapis.com/v1/projects/project-1/locations/global/publishers/google/models/gemini-2.5-flash:generateContent"
        );
    }
}
