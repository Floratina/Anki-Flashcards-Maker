use serde::{Deserialize, Serialize};

pub const DEFAULT_CONCURRENCY_LIMIT: u32 = 10;
pub const MIN_CONCURRENCY_LIMIT: u32 = 1;
pub const MAX_CONCURRENCY_LIMIT: u32 = 50;
pub const DEFAULT_RETRY_COUNT: u32 = 5;
pub const MIN_RETRY_COUNT: u32 = 0;
pub const MAX_RETRY_COUNT: u32 = 10;

fn default_concurrency_limit() -> u32 {
    DEFAULT_CONCURRENCY_LIMIT
}

fn default_retry_count() -> u32 {
    DEFAULT_RETRY_COUNT
}

pub fn clamp_concurrency_limit(value: u32) -> u32 {
    value.clamp(MIN_CONCURRENCY_LIMIT, MAX_CONCURRENCY_LIMIT)
}

pub fn clamp_retry_count(value: u32) -> u32 {
    value.clamp(MIN_RETRY_COUNT, MAX_RETRY_COUNT)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderCallError {
    message: String,
    retryable: bool,
    status: Option<u16>,
}

impl ProviderCallError {
    pub fn fatal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
            status: None,
        }
    }

    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
            status: None,
        }
    }

    pub fn http(status: u16, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: status == 429 || (500..=599).contains(&status),
            status: Some(status),
        }
    }

    pub fn is_retryable(&self) -> bool {
        self.retryable
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for ProviderCallError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ProviderCallError {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderProtocol {
    OpenaiResponses,
    Gemini,
    Deepseek,
    AgentPlatform,
    OpenaiCompatible,
}

impl ProviderProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenaiResponses => "openai-responses",
            Self::Gemini => "gemini",
            Self::Deepseek => "deepseek",
            Self::AgentPlatform => "agent-platform",
            Self::OpenaiCompatible => "openai-compatible",
        }
    }

    pub fn from_db(value: &str) -> Result<Self, String> {
        match value {
            "openai-responses" => Ok(Self::OpenaiResponses),
            "gemini" => Ok(Self::Gemini),
            "deepseek" => Ok(Self::Deepseek),
            "agent-platform" => Ok(Self::AgentPlatform),
            "openai-compatible" => Ok(Self::OpenaiCompatible),
            _ => Err(format!("Unsupported provider protocol: {value}")),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CredentialKind {
    Bearer,
    GeminiApiKey,
    GeminiAuthApiKey,
    ServiceAccount,
    None,
}

impl CredentialKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bearer => "bearer",
            Self::GeminiApiKey => "gemini-api-key",
            Self::GeminiAuthApiKey => "gemini-auth-api-key",
            Self::ServiceAccount => "service-account",
            Self::None => "none",
        }
    }

    pub fn from_db(value: &str) -> Result<Self, String> {
        match value {
            "bearer" => Ok(Self::Bearer),
            "gemini-api-key" => Ok(Self::GeminiApiKey),
            "gemini-auth-api-key" => Ok(Self::GeminiAuthApiKey),
            "service-account" => Ok(Self::ServiceAccount),
            "none" => Ok(Self::None),
            _ => Err(format!("Unsupported credential kind: {value}")),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ThinkingLevel {
    None,
    Low,
    Medium,
    High,
    Max,
}

impl ThinkingLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    pub fn from_db(value: &str) -> Self {
        match value {
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "max" => Self::Max,
            _ => Self::None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
#[allow(dead_code)]
pub enum WebMode {
    Off,
    On,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct VertexConfig {
    pub project_id: String,
    pub location: String,
    pub client_email: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelCapabilities {
    pub thinking_options: Vec<ThinkingLevel>,
    pub web_supported: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderView {
    pub id: String,
    pub name: String,
    pub protocol: ProviderProtocol,
    pub base_url: String,
    pub credential_kind: CredentialKind,
    pub credential_mask: Option<String>,
    pub selected_model: String,
    pub system_prompt: String,
    pub thinking_level: ThinkingLevel,
    pub web_enabled: bool,
    pub is_builtin: bool,
    pub vertex: VertexConfig,
    pub capabilities: ModelCapabilities,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelOption {
    pub id: String,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectivityResult {
    pub success: bool,
    pub latency_ms: u128,
    pub response_text: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FlashcardSettings {
    pub flashcard_prompt: String,
    pub output_directory: String,
    pub selected_provider_id: String,
    #[serde(default = "default_concurrency_limit")]
    pub concurrency_limit: u32,
    #[serde(default = "default_retry_count")]
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum StagedCardStatus {
    Ready,
    Failed,
    Written,
}

impl StagedCardStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Failed => "failed",
            Self::Written => "written",
        }
    }

    pub fn from_db(value: &str) -> Self {
        match value {
            "failed" => Self::Failed,
            "written" => Self::Written,
            _ => Self::Ready,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StagedCardView {
    pub id: String,
    pub input_index: i64,
    pub source_entry: String,
    pub filename: String,
    pub staged_path: String,
    pub status: StagedCardStatus,
    pub warnings: Vec<String>,
    pub error: Option<String>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StagedCardContent {
    pub card: StagedCardView,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateFlashcardsInput {
    pub entries_text: String,
    pub provider_id: String,
    pub flashcard_prompt: String,
    #[serde(default = "default_concurrency_limit")]
    pub concurrency_limit: u32,
    #[serde(default = "default_retry_count")]
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateFlashcardsResult {
    pub cards: Vec<StagedCardView>,
    pub generated: usize,
    pub failed: usize,
    pub cancelled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateFlashcardsProgress {
    pub total: usize,
    pub completed: usize,
    pub in_progress: usize,
    pub generated: usize,
    pub failed: usize,
    pub current_entry: Option<String>,
    pub cancelled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateFlashcardSettingsInput {
    pub flashcard_prompt: String,
    pub output_directory: String,
    pub selected_provider_id: String,
    #[serde(default = "default_concurrency_limit")]
    pub concurrency_limit: u32,
    #[serde(default = "default_retry_count")]
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveStagedCardInput {
    pub id: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WriteStagedCardsResult {
    pub written: usize,
    pub files: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProviderRuntimeConfig {
    pub protocol: ProviderProtocol,
    pub base_url: String,
    pub credential_kind: CredentialKind,
    pub credential: Option<String>,
    pub selected_model: String,
    pub thinking_level: ThinkingLevel,
    pub web_enabled: bool,
    pub vertex: VertexConfig,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateProviderInput {
    pub name: String,
    pub protocol: ProviderProtocol,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateProviderInput {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub credential_kind: CredentialKind,
    pub selected_model: String,
    pub system_prompt: String,
    pub thinking_level: ThinkingLevel,
    pub web_enabled: bool,
    pub vertex_project_id: String,
    pub vertex_location: String,
    pub vertex_client_email: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveCredentialInput {
    pub provider_id: String,
    pub credential: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportAgentPlatformServiceAccountInput {
    pub provider_id: String,
    pub service_account_json: String,
    pub location: Option<String>,
}
