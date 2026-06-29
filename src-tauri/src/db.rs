use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};

use crate::adapters::model_capabilities;
use crate::domain::{
    CreateProviderInput, CredentialKind, ImportAgentPlatformServiceAccountInput, ProviderProtocol,
    ProviderRuntimeConfig, ProviderView, SaveCredentialInput, ThinkingLevel, UpdateProviderInput,
    VertexConfig,
};
use crate::flashcards;
use crate::vertex_ai;

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

pub async fn connect(path: &Path) -> Result<SqlitePool, String> {
    let options = SqliteConnectOptions::new()
        .filename(path)
        .create_if_missing(true)
        .foreign_keys(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await
        .map_err(|error| error.to_string())?;
    migrate(&pool).await?;
    seed_builtin_providers(&pool).await?;
    Ok(pool)
}

async fn migrate(pool: &SqlitePool) -> Result<(), String> {
    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS providers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            protocol TEXT NOT NULL,
            base_url TEXT NOT NULL,
            credential_kind TEXT NOT NULL,
            credential TEXT,
            selected_model TEXT NOT NULL DEFAULT '',
            system_prompt TEXT NOT NULL DEFAULT '',
            thinking_level TEXT NOT NULL DEFAULT 'none',
            web_enabled INTEGER NOT NULL DEFAULT 0,
            vertex_project_id TEXT NOT NULL DEFAULT '',
            vertex_location TEXT NOT NULL DEFAULT 'global',
            vertex_client_email TEXT NOT NULL DEFAULT '',
            is_builtin INTEGER NOT NULL DEFAULT 0,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"#,
    )
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;
    flashcards::migrate(pool).await?;
    Ok(())
}

async fn seed_builtin_providers(pool: &SqlitePool) -> Result<(), String> {
    let providers = [
        (
            "builtin_openai",
            "OpenAI",
            ProviderProtocol::OpenaiResponses,
            "https://api.openai.com",
            CredentialKind::Bearer,
            "gpt-5-mini",
        ),
        (
            "builtin_gemini",
            "Gemini",
            ProviderProtocol::Gemini,
            "https://generativelanguage.googleapis.com",
            CredentialKind::GeminiApiKey,
            "gemini-2.5-flash",
        ),
        (
            "builtin_deepseek",
            "DeepSeek",
            ProviderProtocol::Deepseek,
            "https://api.deepseek.com",
            CredentialKind::Bearer,
            "deepseek-v4-flash",
        ),
        (
            "builtin_agent_platform",
            "Agent Platform",
            ProviderProtocol::AgentPlatform,
            vertex_ai::DEFAULT_BASE_URL,
            CredentialKind::ServiceAccount,
            "gemini-2.5-flash",
        ),
    ];
    for (index, (id, name, protocol, base_url, credential_kind, selected_model)) in
        providers.iter().enumerate()
    {
        sqlx::query(
            "INSERT INTO providers (
                id, name, protocol, base_url, credential_kind, selected_model,
                vertex_location, is_builtin, sort_order
             )
             VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
             ON CONFLICT(id) DO NOTHING",
        )
        .bind(id)
        .bind(name)
        .bind(protocol.as_str())
        .bind(base_url)
        .bind(credential_kind.as_str())
        .bind(selected_model)
        .bind(vertex_ai::DEFAULT_LOCATION)
        .bind(index as i64)
        .execute(pool)
        .await
        .map_err(|error| error.to_string())?;
    }
    sqlx::query(
        "UPDATE providers
         SET selected_model = ?, updated_at = CURRENT_TIMESTAMP
         WHERE id = ? AND selected_model = ?",
    )
    .bind("deepseek-v4-flash")
    .bind("builtin_deepseek")
    .bind("deepseek-chat")
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;
    Ok(())
}

pub async fn list_providers(pool: &SqlitePool) -> Result<Vec<ProviderView>, String> {
    let rows = sqlx::query(
        "SELECT * FROM providers ORDER BY is_builtin DESC, sort_order ASC, created_at ASC",
    )
    .fetch_all(pool)
    .await
    .map_err(|error| error.to_string())?;
    rows.iter().map(provider_view_from_row).collect()
}

pub async fn get_provider(pool: &SqlitePool, id: &str) -> Result<ProviderView, String> {
    let row = sqlx::query("SELECT * FROM providers WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "Provider not found".to_string())?;
    provider_view_from_row(&row)
}

pub async fn runtime_config(
    pool: &SqlitePool,
    provider_id: &str,
) -> Result<ProviderRuntimeConfig, String> {
    let row = sqlx::query("SELECT * FROM providers WHERE id = ?")
        .bind(provider_id)
        .fetch_optional(pool)
        .await
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "Provider not found".to_string())?;
    let protocol = ProviderProtocol::from_db(&row.get::<String, _>("protocol"))?;
    let credential_kind = CredentialKind::from_db(&row.get::<String, _>("credential_kind"))?;
    Ok(ProviderRuntimeConfig {
        protocol,
        base_url: row.get("base_url"),
        credential_kind,
        credential: row.get("credential"),
        selected_model: row.get("selected_model"),
        thinking_level: ThinkingLevel::from_db(&row.get::<String, _>("thinking_level")),
        web_enabled: row.get::<i64, _>("web_enabled") != 0,
        vertex: vertex_from_row(&row),
    })
}

pub async fn create_provider(
    pool: &SqlitePool,
    input: CreateProviderInput,
) -> Result<ProviderView, String> {
    let name = input.name.trim();
    if name.is_empty() {
        return Err("Provider name is required".into());
    }
    let id = next_id("provider");
    let base_url = default_base_url(input.protocol);
    let credential_kind = default_credential_kind(input.protocol);
    let selected_model = default_model(input.protocol);
    let sort_order: i64 =
        sqlx::query_scalar("SELECT COALESCE(MAX(sort_order), 0) + 1 FROM providers")
            .fetch_one(pool)
            .await
            .map_err(|error| error.to_string())?;
    sqlx::query(
        "INSERT INTO providers (
            id, name, protocol, base_url, credential_kind, selected_model,
            vertex_location, is_builtin, sort_order
         )
         VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
    )
    .bind(&id)
    .bind(name)
    .bind(input.protocol.as_str())
    .bind(base_url)
    .bind(credential_kind.as_str())
    .bind(selected_model)
    .bind(vertex_ai::DEFAULT_LOCATION)
    .bind(sort_order)
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;
    get_provider(pool, &id).await
}

pub async fn delete_provider(pool: &SqlitePool, id: &str) -> Result<(), String> {
    let provider = get_provider(pool, id).await?;
    if provider.is_builtin {
        return Err("预设提供商不能删除".into());
    }
    let result = sqlx::query("DELETE FROM providers WHERE id = ? AND is_builtin = 0")
        .bind(id)
        .execute(pool)
        .await
        .map_err(|error| error.to_string())?;
    if result.rows_affected() == 0 {
        return Err("Provider not found".into());
    }
    sqlx::query(
        "UPDATE app_settings SET value = '', updated_at = CURRENT_TIMESTAMP WHERE key = 'selectedProviderId' AND value = ?",
    )
    .bind(id)
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;
    Ok(())
}

pub async fn update_provider(
    pool: &SqlitePool,
    input: UpdateProviderInput,
) -> Result<ProviderView, String> {
    let existing = runtime_config(pool, &input.id).await?;
    let name = input.name.trim();
    let base_url = input.base_url.trim();
    if name.is_empty() {
        return Err("Provider name is required".into());
    }
    if base_url.is_empty() {
        return Err("Base URL is required".into());
    }
    let selected_model = input.selected_model.trim();
    let caps = model_capabilities(existing.protocol, selected_model);
    let thinking_level = if caps.thinking_options.contains(&input.thinking_level) {
        input.thinking_level
    } else {
        ThinkingLevel::None
    };
    let web_enabled = input.web_enabled && caps.web_supported;
    sqlx::query(
        "UPDATE providers SET
            name = ?,
            base_url = ?,
            credential_kind = ?,
            selected_model = ?,
            system_prompt = ?,
            thinking_level = ?,
            web_enabled = ?,
            vertex_project_id = ?,
            vertex_location = ?,
            vertex_client_email = ?,
            updated_at = CURRENT_TIMESTAMP
         WHERE id = ?",
    )
    .bind(name)
    .bind(base_url)
    .bind(input.credential_kind.as_str())
    .bind(selected_model)
    .bind(input.system_prompt.trim())
    .bind(thinking_level.as_str())
    .bind(if web_enabled { 1_i64 } else { 0_i64 })
    .bind(input.vertex_project_id.trim())
    .bind(
        input
            .vertex_location
            .trim()
            .if_empty(vertex_ai::DEFAULT_LOCATION),
    )
    .bind(input.vertex_client_email.trim())
    .bind(&input.id)
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;
    get_provider(pool, &input.id).await
}

pub async fn save_credential(
    pool: &SqlitePool,
    input: SaveCredentialInput,
) -> Result<ProviderView, String> {
    let credential = input
        .credential
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    sqlx::query("UPDATE providers SET credential = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?")
        .bind(credential)
        .bind(&input.provider_id)
        .execute(pool)
        .await
        .map_err(|error| error.to_string())?;
    get_provider(pool, &input.provider_id).await
}

pub async fn import_agent_platform_service_account(
    pool: &SqlitePool,
    input: ImportAgentPlatformServiceAccountInput,
) -> Result<ProviderView, String> {
    let parsed = vertex_ai::parse_service_account_json(&input.service_account_json)?;
    let existing = runtime_config(pool, &input.provider_id).await?;
    if existing.protocol != ProviderProtocol::AgentPlatform {
        return Err("Service Account JSON can only be imported for Agent Platform".into());
    }
    let project_id = if parsed.project_id.trim().is_empty() {
        existing.vertex.project_id
    } else {
        parsed.project_id
    };
    let location = input
        .location
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(&existing.vertex.location)
        .to_string();
    sqlx::query(
        "UPDATE providers SET
            credential = ?,
            credential_kind = ?,
            vertex_project_id = ?,
            vertex_location = ?,
            vertex_client_email = ?,
            updated_at = CURRENT_TIMESTAMP
         WHERE id = ?",
    )
    .bind(parsed.private_key)
    .bind(CredentialKind::ServiceAccount.as_str())
    .bind(project_id.trim())
    .bind(location.trim().if_empty(vertex_ai::DEFAULT_LOCATION))
    .bind(parsed.client_email.trim())
    .bind(&input.provider_id)
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;
    get_provider(pool, &input.provider_id).await
}

fn provider_view_from_row(row: &sqlx::sqlite::SqliteRow) -> Result<ProviderView, String> {
    let protocol = ProviderProtocol::from_db(&row.get::<String, _>("protocol"))?;
    let credential_kind = CredentialKind::from_db(&row.get::<String, _>("credential_kind"))?;
    let selected_model: String = row.get("selected_model");
    Ok(ProviderView {
        id: row.get("id"),
        name: row.get("name"),
        protocol,
        base_url: row.get("base_url"),
        credential_kind,
        credential_mask: row
            .get::<Option<String>, _>("credential")
            .as_deref()
            .map(mask_secret),
        selected_model: selected_model.clone(),
        system_prompt: row.get("system_prompt"),
        thinking_level: ThinkingLevel::from_db(&row.get::<String, _>("thinking_level")),
        web_enabled: row.get::<i64, _>("web_enabled") != 0,
        is_builtin: row.get::<i64, _>("is_builtin") != 0,
        vertex: vertex_from_row(row),
        capabilities: model_capabilities(protocol, &selected_model),
        updated_at: row.get("updated_at"),
    })
}

fn vertex_from_row(row: &sqlx::sqlite::SqliteRow) -> VertexConfig {
    VertexConfig {
        project_id: row.get("vertex_project_id"),
        location: row
            .get::<String, _>("vertex_location")
            .if_empty(vertex_ai::DEFAULT_LOCATION),
        client_email: row.get("vertex_client_email"),
    }
}

fn mask_secret(secret: &str) -> String {
    let trimmed = secret.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let suffix = trimmed
        .chars()
        .rev()
        .take(4)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("**** {suffix}")
}

fn next_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{nanos:x}{counter:x}")
}

fn default_base_url(protocol: ProviderProtocol) -> &'static str {
    match protocol {
        ProviderProtocol::OpenaiResponses => "https://api.openai.com",
        ProviderProtocol::Gemini => "https://generativelanguage.googleapis.com",
        ProviderProtocol::Deepseek => "https://api.deepseek.com",
        ProviderProtocol::AgentPlatform => vertex_ai::DEFAULT_BASE_URL,
        ProviderProtocol::OpenaiCompatible => "https://api.example.com",
    }
}

fn default_credential_kind(protocol: ProviderProtocol) -> CredentialKind {
    match protocol {
        ProviderProtocol::Gemini => CredentialKind::GeminiApiKey,
        ProviderProtocol::AgentPlatform => CredentialKind::ServiceAccount,
        ProviderProtocol::OpenaiResponses
        | ProviderProtocol::Deepseek
        | ProviderProtocol::OpenaiCompatible => CredentialKind::Bearer,
    }
}

fn default_model(protocol: ProviderProtocol) -> &'static str {
    match protocol {
        ProviderProtocol::OpenaiResponses => "gpt-5-mini",
        ProviderProtocol::Gemini | ProviderProtocol::AgentPlatform => "gemini-2.5-flash",
        ProviderProtocol::Deepseek => "deepseek-v4-flash",
        ProviderProtocol::OpenaiCompatible => "",
    }
}

trait EmptyDefault {
    fn if_empty(self, default: &str) -> String;
}

impl EmptyDefault for &str {
    fn if_empty(self, default: &str) -> String {
        if self.trim().is_empty() {
            default.to_string()
        } else {
            self.trim().to_string()
        }
    }
}

impl EmptyDefault for String {
    fn if_empty(self, default: &str) -> String {
        if self.trim().is_empty() {
            default.to_string()
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_db_path() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("flashcards-maker-test-{nanos}.sqlite3"))
    }

    #[tokio::test]
    async fn deletes_custom_provider_and_keeps_builtin_providers() {
        let path = temp_db_path();
        let pool = connect(&path).await.expect("connect");
        let custom = create_provider(
            &pool,
            CreateProviderInput {
                name: "Custom".into(),
                protocol: ProviderProtocol::OpenaiCompatible,
            },
        )
        .await
        .expect("create provider");
        crate::flashcards::update_settings(
            &pool,
            crate::domain::UpdateFlashcardSettingsInput {
                flashcard_prompt: "prompt".into(),
                output_directory: "".into(),
                selected_provider_id: custom.id.clone(),
                concurrency_limit: crate::domain::DEFAULT_CONCURRENCY_LIMIT,
                retry_count: crate::domain::DEFAULT_RETRY_COUNT,
            },
        )
        .await
        .expect("settings");

        delete_provider(&pool, "builtin_openai")
            .await
            .expect_err("builtin providers cannot be deleted");
        delete_provider(&pool, &custom.id).await.expect("delete custom");

        assert!(get_provider(&pool, &custom.id).await.is_err());
        assert!(get_provider(&pool, "builtin_openai").await.is_ok());
        let settings = crate::flashcards::get_settings(&pool).await.expect("settings");
        assert_ne!(settings.selected_provider_id, custom.id);
        pool.close().await;
        let _ = std::fs::remove_file(path);
    }
    #[tokio::test]
    async fn seeds_builtin_providers_and_masks_credentials() {
        let path = temp_db_path();
        let pool = connect(&path).await.expect("connect");
        let providers = list_providers(&pool).await.expect("providers");
        assert_eq!(providers.iter().filter(|item| item.is_builtin).count(), 4);
        save_credential(
            &pool,
            SaveCredentialInput {
                provider_id: "builtin_openai".into(),
                credential: Some("sk-test-secret".into()),
            },
        )
        .await
        .expect("save credential");
        let openai = get_provider(&pool, "builtin_openai").await.expect("openai");
        assert_eq!(openai.credential_mask.as_deref(), Some("**** cret"));
        pool.close().await;
        let _ = std::fs::remove_file(path);
    }
}
