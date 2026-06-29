use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Client, Method};
use serde_json::{json, Value};

use crate::domain::{
    ConnectivityResult, CredentialKind, ModelCapabilities, ModelOption, ProviderCallError,
    ProviderProtocol, ProviderRuntimeConfig, ThinkingLevel,
};
use crate::vertex_ai;

#[derive(Clone)]
pub struct RuntimeAdapter {
    client: Client,
    config: ProviderRuntimeConfig,
}

impl RuntimeAdapter {
    pub fn new(client: Client, config: ProviderRuntimeConfig) -> Self {
        Self { client, config }
    }

    pub async fn list_models(&self) -> Result<Vec<ModelOption>, String> {
        let (url, method) = match self.config.protocol {
            ProviderProtocol::OpenaiResponses
            | ProviderProtocol::Deepseek
            | ProviderProtocol::OpenaiCompatible => (
                openai_endpoint(&self.config.base_url, "models"),
                Method::GET,
            ),
            ProviderProtocol::Gemini => (
                append_endpoint_suffix(&endpoint_base_url(&self.config.base_url), "v1beta/models"),
                Method::GET,
            ),
            ProviderProtocol::AgentPlatform => {
                let vertex = vertex_ai::runtime_config(&self.config)?;
                (
                    format!(
                        "{}?pageSize=100&listAllVersions=true",
                        vertex_ai::publisher_models_url(
                            &self.config.base_url,
                            &vertex.location,
                            "google",
                        )
                    ),
                    Method::GET,
                )
            }
        };
        let value = self
            .request_json(method, url, None)
            .await
            .map_err(|error| error.to_string())?;
        Ok(parse_models(self.config.protocol, &value))
    }

    pub async fn test_connectivity(&self) -> Result<String, String> {
        if self.config.selected_model.trim().is_empty() {
            return Err("Model is required before testing connectivity".into());
        }
        let (url, body) = self.build_prompt_request("", "Reply with OK.", 64)?;
        let value = self
            .request_json(Method::POST, url, Some(body))
            .await
            .map_err(|error| error.to_string())?;
        let text = extract_response_text(self.config.protocol, &value);
        if text.trim().is_empty() {
            return Err("Connectivity succeeded, but the model returned an empty response".into());
        }
        Ok(text)
    }

    pub async fn generate_flashcard(
        &self,
        flashcard_prompt: &str,
        source_entry: &str,
    ) -> Result<String, ProviderCallError> {
        if self.config.selected_model.trim().is_empty() {
            return Err(ProviderCallError::fatal(
                "Model is required before generating flashcards",
            ));
        }
        let (url, body) = self
            .build_prompt_request(flashcard_prompt, source_entry, 4096)
            .map_err(ProviderCallError::fatal)?;
        let value = self.request_json(Method::POST, url, Some(body)).await?;
        let text = extract_response_text(self.config.protocol, &value);
        if text.trim().is_empty() {
            return Err(ProviderCallError::fatal(
                "The model returned an empty flashcard",
            ));
        }
        Ok(text)
    }

    #[cfg(test)]
    pub fn build_generation_request(
        &self,
        flashcard_prompt: &str,
        source_entry: &str,
    ) -> Result<(String, Value), String> {
        self.build_prompt_request(flashcard_prompt, source_entry, 4096)
    }

    fn build_prompt_request(
        &self,
        system_prompt: &str,
        user_text: &str,
        max_output_tokens: u32,
    ) -> Result<(String, Value), String> {
        let model = self.config.selected_model.trim();
        match self.config.protocol {
            ProviderProtocol::OpenaiResponses => Ok((
                openai_endpoint(&self.config.base_url, "responses"),
                build_openai_responses_body(
                    model,
                    system_prompt,
                    user_text,
                    max_output_tokens,
                    self.config.thinking_level,
                    self.config.web_enabled,
                ),
            )),
            ProviderProtocol::Deepseek | ProviderProtocol::OpenaiCompatible => Ok((
                openai_endpoint(&self.config.base_url, "chat/completions"),
                build_chat_completions_body(
                    self.config.protocol,
                    model,
                    system_prompt,
                    user_text,
                    max_output_tokens,
                    self.config.thinking_level,
                    self.config.web_enabled,
                ),
            )),
            ProviderProtocol::Gemini => Ok((
                gemini_generate_url(&self.config.base_url, model),
                build_gemini_body(
                    ProviderProtocol::Gemini,
                    model,
                    system_prompt,
                    user_text,
                    max_output_tokens,
                    self.config.thinking_level,
                    self.config.web_enabled,
                ),
            )),
            ProviderProtocol::AgentPlatform => {
                let vertex = vertex_ai::runtime_config(&self.config)?;
                Ok((
                    vertex_ai::generate_content_url(
                        &self.config.base_url,
                        &vertex.project_id,
                        &vertex.location,
                        model,
                    ),
                    build_gemini_body(
                        ProviderProtocol::AgentPlatform,
                        model,
                        system_prompt,
                        user_text,
                        max_output_tokens,
                        self.config.thinking_level,
                        self.config.web_enabled,
                    ),
                ))
            }
        }
    }

    async fn headers(&self) -> Result<HeaderMap, ProviderCallError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        match self.config.protocol {
            ProviderProtocol::AgentPlatform => {
                let token = vertex_ai::access_token(&self.client, &self.config).await?;
                headers.insert(
                    AUTHORIZATION,
                    HeaderValue::from_str(&format!("Bearer {token}")).map_err(|error| {
                        ProviderCallError::fatal(format!("Invalid Agent Platform token: {error}"))
                    })?,
                );
            }
            ProviderProtocol::Gemini => {
                if let Some(credential) = self.credential() {
                    match self.config.credential_kind {
                        CredentialKind::GeminiAuthApiKey => {
                            headers.insert(
                                AUTHORIZATION,
                                HeaderValue::from_str(&format!("Bearer {credential}")).map_err(
                                    |error| {
                                        ProviderCallError::fatal(format!(
                                            "Invalid Auth API Key: {error}"
                                        ))
                                    },
                                )?,
                            );
                        }
                        _ => {
                            headers.insert(
                                "x-goog-api-key",
                                HeaderValue::from_str(credential).map_err(|error| {
                                    ProviderCallError::fatal(format!(
                                        "Invalid Gemini API Key: {error}"
                                    ))
                                })?,
                            );
                        }
                    }
                }
            }
            ProviderProtocol::OpenaiResponses
            | ProviderProtocol::Deepseek
            | ProviderProtocol::OpenaiCompatible => {
                if let Some(credential) = self.credential() {
                    headers.insert(
                        AUTHORIZATION,
                        HeaderValue::from_str(&format!("Bearer {credential}")).map_err(
                            |error| ProviderCallError::fatal(format!("Invalid API Key: {error}")),
                        )?,
                    );
                }
            }
        }
        Ok(headers)
    }

    fn credential(&self) -> Option<&str> {
        self.config
            .credential
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    async fn request_json(
        &self,
        method: Method,
        url: String,
        body: Option<Value>,
    ) -> Result<Value, ProviderCallError> {
        let mut request = self
            .client
            .request(method, url)
            .headers(self.headers().await?);
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request.send().await.map_err(provider_reqwest_error)?;
        let status = response.status();
        let text = response.text().await.map_err(provider_reqwest_error)?;
        if !status.is_success() {
            return Err(ProviderCallError::http(
                status.as_u16(),
                format!(
                    "HTTP {}: {}",
                    status.as_u16(),
                    text.chars().take(500).collect::<String>()
                ),
            ));
        }
        serde_json::from_str(&text)
            .map_err(|error| ProviderCallError::fatal(format!("Invalid JSON response: {error}")))
    }
}

pub fn provider_reqwest_error(error: reqwest::Error) -> ProviderCallError {
    let message = error.to_string();
    if let Some(status) = error.status() {
        ProviderCallError::http(status.as_u16(), message)
    } else if error.is_timeout() || error.is_connect() || error.is_request() || error.is_body() {
        ProviderCallError::retryable(message)
    } else {
        ProviderCallError::fatal(message)
    }
}

pub async fn test_provider_connectivity(
    client: Client,
    config: ProviderRuntimeConfig,
) -> ConnectivityResult {
    let start = std::time::Instant::now();
    match RuntimeAdapter::new(client, config)
        .test_connectivity()
        .await
    {
        Ok(text) => ConnectivityResult {
            success: true,
            latency_ms: start.elapsed().as_millis(),
            response_text: text,
            error: None,
        },
        Err(error) => ConnectivityResult {
            success: false,
            latency_ms: start.elapsed().as_millis(),
            response_text: String::new(),
            error: Some(error),
        },
    }
}

pub fn model_capabilities(protocol: ProviderProtocol, model: &str) -> ModelCapabilities {
    let thinking_options = match protocol {
        ProviderProtocol::Deepseek if supports_thinking(protocol, model) => {
            vec![ThinkingLevel::None, ThinkingLevel::High, ThinkingLevel::Max]
        }
        ProviderProtocol::OpenaiResponses
        | ProviderProtocol::Gemini
        | ProviderProtocol::AgentPlatform
            if supports_thinking(protocol, model) =>
        {
            vec![
                ThinkingLevel::None,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
            ]
        }
        _ => vec![ThinkingLevel::None],
    };
    ModelCapabilities {
        thinking_options,
        web_supported: supports_web(protocol, model),
    }
}

pub fn supports_web(protocol: ProviderProtocol, model: &str) -> bool {
    match protocol {
        ProviderProtocol::OpenaiResponses => !model.trim().is_empty(),
        ProviderProtocol::Gemini | ProviderProtocol::AgentPlatform => {
            model.to_ascii_lowercase().contains("gemini")
        }
        ProviderProtocol::Deepseek | ProviderProtocol::OpenaiCompatible => false,
    }
}

fn supports_thinking(protocol: ProviderProtocol, model: &str) -> bool {
    let model = model
        .trim()
        .trim_start_matches("models/")
        .to_ascii_lowercase();
    match protocol {
        ProviderProtocol::OpenaiResponses => {
            model.starts_with("gpt-5")
                || model.starts_with("o1")
                || model.starts_with("o3")
                || model.starts_with("o4")
                || model.starts_with("gpt-oss")
        }
        ProviderProtocol::Gemini | ProviderProtocol::AgentPlatform => {
            model.starts_with("gemini-2.5") || model.starts_with("gemini-3")
        }
        ProviderProtocol::Deepseek => model.starts_with("deepseek-v4"),
        ProviderProtocol::OpenaiCompatible => false,
    }
}

fn build_openai_responses_body(
    model: &str,
    system_prompt: &str,
    user_text: &str,
    max_output_tokens: u32,
    thinking_level: ThinkingLevel,
    web_enabled: bool,
) -> Value {
    let mut input = Vec::new();
    if !system_prompt.trim().is_empty() {
        input.push(json!({
            "role": "system",
            "content": [{"type": "input_text", "text": system_prompt.trim()}]
        }));
    }
    input.push(json!({
        "role": "user",
        "content": [{"type": "input_text", "text": user_text}]
    }));
    let mut body = json!({
        "model": model,
        "input": input,
        "max_output_tokens": max_output_tokens
    });
    if thinking_level != ThinkingLevel::None
        && supports_thinking(ProviderProtocol::OpenaiResponses, model)
    {
        body["reasoning"] = json!({
            "effort": thinking_level.as_str(),
            "summary": "auto"
        });
    }
    if web_enabled && supports_web(ProviderProtocol::OpenaiResponses, model) {
        body["tools"] = json!([{ "type": "web_search" }]);
    }
    body
}

fn build_chat_completions_body(
    protocol: ProviderProtocol,
    model: &str,
    system_prompt: &str,
    user_text: &str,
    max_output_tokens: u32,
    thinking_level: ThinkingLevel,
    _web_enabled: bool,
) -> Value {
    let mut messages = Vec::new();
    if !system_prompt.trim().is_empty() {
        messages.push(json!({"role": "system", "content": system_prompt.trim()}));
    }
    messages.push(json!({"role": "user", "content": user_text}));
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "max_tokens": max_output_tokens
    });
    if protocol == ProviderProtocol::Deepseek && model.to_ascii_lowercase().contains("reasoner") {
        body["temperature"] = json!(0);
    }
    if protocol == ProviderProtocol::Deepseek && supports_thinking(protocol, model) {
        body["thinking"] = json!({
            "type": if thinking_level == ThinkingLevel::None { "disabled" } else { "enabled" }
        });
        if thinking_level != ThinkingLevel::None {
            body["reasoning_effort"] = json!(deepseek_reasoning_effort(thinking_level));
        }
    }
    body
}

fn deepseek_reasoning_effort(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Max => "max",
        ThinkingLevel::Low | ThinkingLevel::Medium | ThinkingLevel::High => "high",
        ThinkingLevel::None => "high",
    }
}

fn build_gemini_body(
    protocol: ProviderProtocol,
    model: &str,
    system_prompt: &str,
    user_text: &str,
    max_output_tokens: u32,
    thinking_level: ThinkingLevel,
    web_enabled: bool,
) -> Value {
    let mut body = json!({
        "contents": [{
            "role": "user",
            "parts": [{ "text": user_text }]
        }],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens
        }
    });
    if !system_prompt.trim().is_empty() {
        body["systemInstruction"] = json!({
            "parts": [{ "text": system_prompt.trim() }]
        });
    }
    if supports_thinking(protocol, model) {
        body["generationConfig"]["thinkingConfig"] = json!({
            "thinkingBudget": gemini_thinking_budget(thinking_level)
        });
    }
    if web_enabled && supports_web(protocol, model) {
        body["tools"] = if protocol == ProviderProtocol::AgentPlatform {
            json!([{ "googleSearch": {} }])
        } else {
            json!([{ "google_search": {} }])
        };
    }
    body
}

fn gemini_thinking_budget(level: ThinkingLevel) -> Value {
    match level {
        ThinkingLevel::None => json!(0),
        ThinkingLevel::Low => json!(1024),
        ThinkingLevel::Medium => json!(-1),
        ThinkingLevel::High => json!(24576),
        ThinkingLevel::Max => json!(24576),
    }
}

fn parse_models(protocol: ProviderProtocol, value: &Value) -> Vec<ModelOption> {
    let items = match protocol {
        ProviderProtocol::OpenaiResponses
        | ProviderProtocol::Deepseek
        | ProviderProtocol::OpenaiCompatible => value
            .get("data")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        ProviderProtocol::Gemini => value
            .get("models")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        ProviderProtocol::AgentPlatform => value
            .get("publisherModels")
            .or_else(|| value.get("models"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
    };
    let mut models = Vec::new();
    for item in items {
        let mut id = item
            .get("id")
            .or_else(|| item.get("name"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if protocol == ProviderProtocol::Gemini || protocol == ProviderProtocol::AgentPlatform {
            id = vertex_ai::model_id(&id);
        }
        if id.is_empty() {
            continue;
        }
        if protocol == ProviderProtocol::AgentPlatform
            && !id.to_ascii_lowercase().starts_with("gemini")
        {
            continue;
        }
        let label = item
            .get("display_name")
            .or_else(|| item.get("displayName"))
            .and_then(Value::as_str)
            .unwrap_or(&id)
            .to_string();
        models.push(ModelOption { id, label });
    }
    models.sort_by(|left, right| left.id.cmp(&right.id));
    models
}

fn extract_response_text(protocol: ProviderProtocol, value: &Value) -> String {
    match protocol {
        ProviderProtocol::OpenaiResponses => value
            .get("output_text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                value
                    .get("output")
                    .and_then(Value::as_array)
                    .and_then(|items| {
                        items.iter().find_map(|item| {
                            item.get("content")
                                .and_then(Value::as_array)
                                .and_then(|parts| {
                                    parts
                                        .iter()
                                        .find_map(|part| part.get("text").and_then(Value::as_str))
                                })
                        })
                    })
                    .map(str::to_string)
            })
            .unwrap_or_default(),
        ProviderProtocol::Gemini | ProviderProtocol::AgentPlatform => {
            extract_gemini_response_text(value)
        }
        ProviderProtocol::Deepseek | ProviderProtocol::OpenaiCompatible => value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    }
}

fn extract_gemini_response_text(value: &Value) -> String {
    value
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter(|part| {
                    !part
                        .get("thought")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                })
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .filter(|text| !text.trim().is_empty())
        .or_else(|| {
            value
                .pointer("/candidates/0/content/parts/0/text")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_default()
}

fn endpoint_base_url(base_url: &str) -> String {
    base_url
        .split('#')
        .next()
        .unwrap_or(base_url)
        .trim()
        .trim_end_matches('/')
        .to_string()
}

fn append_endpoint_suffix(base_url: &str, suffix: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        suffix.trim_start_matches('/')
    )
}

fn openai_endpoint(base_url: &str, suffix: &str) -> String {
    let base = endpoint_base_url(base_url);
    if is_versioned_base_url(&base) {
        append_endpoint_suffix(&base, suffix)
    } else {
        append_endpoint_suffix(&format!("{base}/v1"), suffix)
    }
}

fn is_versioned_base_url(base_url: &str) -> bool {
    url::Url::parse(base_url)
        .ok()
        .map(|url| {
            let path = url.path().trim_end_matches('/');
            path.ends_with("/v1") || path.ends_with("/v1beta") || path.ends_with("/v1beta1")
        })
        .unwrap_or_else(|| {
            let path = base_url.trim_end_matches('/');
            path.ends_with("/v1") || path.ends_with("/v1beta") || path.ends_with("/v1beta1")
        })
}

fn gemini_generate_url(base_url: &str, model: &str) -> String {
    append_endpoint_suffix(
        &endpoint_base_url(base_url),
        &format!(
            "v1beta/models/{}:generateContent",
            vertex_ai::model_id(model)
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn runtime(protocol: ProviderProtocol, base_url: String, model: &str) -> ProviderRuntimeConfig {
        ProviderRuntimeConfig {
            protocol,
            base_url,
            credential_kind: CredentialKind::Bearer,
            credential: None,
            selected_model: model.into(),
            thinking_level: ThinkingLevel::High,
            web_enabled: true,
            vertex: vertex_ai::default_config(),
        }
    }

    #[test]
    fn builds_openai_responses_reasoning_and_web_body() {
        let body = build_openai_responses_body(
            "gpt-5-mini",
            "sys",
            "apple",
            4096,
            ThinkingLevel::High,
            true,
        );
        assert_eq!(body.pointer("/reasoning/effort"), Some(&json!("high")));
        assert_eq!(body.pointer("/tools/0/type"), Some(&json!("web_search")));
        assert_eq!(body.pointer("/input/0/role"), Some(&json!("system")));
        assert_eq!(
            body.pointer("/input/1/content/0/text"),
            Some(&json!("apple"))
        );
    }

    #[test]
    fn openai_responses_none_thinking_omits_reasoning_body() {
        let body = build_openai_responses_body(
            "gpt-5-mini",
            "sys",
            "apple",
            4096,
            ThinkingLevel::None,
            false,
        );
        assert!(body.get("reasoning").is_none());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn openai_compatible_fallback_omits_thinking_and_web() {
        let body = build_chat_completions_body(
            ProviderProtocol::OpenaiCompatible,
            "custom-model",
            "sys",
            "apple",
            4096,
            ThinkingLevel::High,
            true,
        );
        assert!(body.get("reasoning").is_none());
        assert!(body.get("tools").is_none());
        assert_eq!(body.pointer("/messages/0/role"), Some(&json!("system")));
        assert_eq!(body.pointer("/messages/1/content"), Some(&json!("apple")));
    }

    #[test]
    fn deepseek_v4_builds_official_thinking_body() {
        let high = build_chat_completions_body(
            ProviderProtocol::Deepseek,
            "deepseek-v4-flash",
            "sys",
            "apple",
            4096,
            ThinkingLevel::High,
            false,
        );
        assert_eq!(high.pointer("/thinking/type"), Some(&json!("enabled")));
        assert_eq!(high.pointer("/reasoning_effort"), Some(&json!("high")));

        let max = build_chat_completions_body(
            ProviderProtocol::Deepseek,
            "deepseek-v4-pro",
            "sys",
            "apple",
            4096,
            ThinkingLevel::Max,
            false,
        );
        assert_eq!(max.pointer("/thinking/type"), Some(&json!("enabled")));
        assert_eq!(max.pointer("/reasoning_effort"), Some(&json!("max")));

        let off = build_chat_completions_body(
            ProviderProtocol::Deepseek,
            "deepseek-v4-flash",
            "sys",
            "apple",
            4096,
            ThinkingLevel::None,
            false,
        );
        assert_eq!(off.pointer("/thinking/type"), Some(&json!("disabled")));
        assert!(off.get("reasoning_effort").is_none());
    }

    #[test]
    fn builds_gemini_and_agent_platform_web_bodies() {
        let gemini = build_gemini_body(
            ProviderProtocol::Gemini,
            "gemini-2.5-flash",
            "",
            "apple",
            4096,
            ThinkingLevel::Low,
            true,
        );
        assert_eq!(
            gemini.pointer("/generationConfig/thinkingConfig/thinkingBudget"),
            Some(&json!(1024))
        );
        assert!(gemini
            .pointer("/generationConfig/thinkingConfig/includeThoughts")
            .is_none());
        assert_eq!(gemini.pointer("/tools/0/google_search"), Some(&json!({})));

        let agent = build_gemini_body(
            ProviderProtocol::AgentPlatform,
            "gemini-2.5-flash",
            "",
            "apple",
            4096,
            ThinkingLevel::High,
            true,
        );
        assert_eq!(agent.pointer("/tools/0/googleSearch"), Some(&json!({})));
    }

    #[test]
    fn generation_request_uses_flashcard_prompt_as_system_prompt() {
        let adapter = RuntimeAdapter::new(
            Client::new(),
            runtime(
                ProviderProtocol::OpenaiResponses,
                "https://api.openai.com".into(),
                "gpt-5-mini",
            ),
        );
        let (_, body) = adapter
            .build_generation_request("Flashcard format", "flabbergasted")
            .expect("request");
        assert_eq!(
            body.pointer("/input/0/content/0/text"),
            Some(&json!("Flashcard format"))
        );
        assert_eq!(
            body.pointer("/input/1/content/0/text"),
            Some(&json!("flabbergasted"))
        );
    }

    #[test]
    fn gemini_response_text_skips_thought_parts() {
        let value = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"thought": true, "text": "internal thought"},
                        {"text": "# Gemini\n\nfinal card"}
                    ]
                }
            }]
        });
        assert_eq!(
            extract_response_text(ProviderProtocol::AgentPlatform, &value),
            "# Gemini\n\nfinal card"
        );
    }

    #[test]
    fn parses_provider_model_lists() {
        let openai = parse_models(
            ProviderProtocol::OpenaiResponses,
            &json!({"data": [{"id": "gpt-5-mini"}]}),
        );
        assert_eq!(openai[0].id, "gpt-5-mini");
        let vertex = parse_models(
            ProviderProtocol::AgentPlatform,
            &json!({"publisherModels": [
                {"name": "publishers/google/models/gemini-2.5-flash", "displayName": "Gemini Flash"},
                {"name": "publishers/google/models/textembedding-gecko", "displayName": "Embeddings"}
            ]}),
        );
        assert_eq!(vertex.len(), 1);
        assert_eq!(vertex[0].id, "gemini-2.5-flash");
    }

    #[test]
    fn builds_openai_urls() {
        assert_eq!(
            openai_endpoint("https://api.openai.com", "responses"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            openai_endpoint("https://proxy.example.com/openai/v1", "models"),
            "https://proxy.example.com/openai/v1/models"
        );
    }

    #[tokio::test]
    async fn fetches_models_from_mock_openai_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("address");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = [0_u8; 2048];
            let _ = stream.read(&mut request);
            let body = r#"{"data":[{"id":"mock-model","name":"Mock Model"}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .expect("write");
        });
        let adapter = RuntimeAdapter::new(
            Client::new(),
            runtime(
                ProviderProtocol::OpenaiResponses,
                format!("http://{address}"),
                "mock-model",
            ),
        );
        let models = adapter.list_models().await.expect("models");
        assert_eq!(models[0].id, "mock-model");
        server.join().expect("server");
    }

    #[tokio::test]
    async fn tests_connectivity_with_mock_openai_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("address");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request);
            let body =
                r#"{"output":[{"type":"message","content":[{"type":"output_text","text":"OK"}]}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .expect("write");
        });
        let adapter = RuntimeAdapter::new(
            Client::new(),
            runtime(
                ProviderProtocol::OpenaiResponses,
                format!("http://{address}"),
                "gpt-5-mini",
            ),
        );
        let text = adapter.test_connectivity().await.expect("connectivity");
        assert_eq!(text, "OK");
        server.join().expect("server");
    }
}
