use std::path::PathBuf;

use tokio::sync::watch;

use reqwest::Client;
use sqlx::SqlitePool;
use tauri::{AppHandle, Emitter, State};
use tauri_plugin_dialog::DialogExt;

use crate::adapters::{test_provider_connectivity, RuntimeAdapter};
use crate::db;
use crate::domain::{
    ConnectivityResult, CreateProviderInput, FlashcardSettings, GenerateFlashcardsInput,
    GenerateFlashcardsResult, ImportAgentPlatformServiceAccountInput, ModelOption, ProviderView,
    SaveCredentialInput, SaveStagedCardInput, StagedCardContent, StagedCardView,
    UpdateFlashcardSettingsInput, UpdateProviderInput, WriteStagedCardsResult,
};
use crate::flashcards;

pub struct AppState {
    pub pool: SqlitePool,
    pub client: Client,
    pub staging_root: PathBuf,
    pub generation_cancel_tx: watch::Sender<bool>,
}

const GENERATION_PROGRESS_EVENT: &str = "flashcards-generation-progress";

#[tauri::command]
pub async fn list_providers(state: State<'_, AppState>) -> Result<Vec<ProviderView>, String> {
    db::list_providers(&state.pool).await
}

#[tauri::command]
pub async fn create_provider(
    state: State<'_, AppState>,
    input: CreateProviderInput,
) -> Result<ProviderView, String> {
    db::create_provider(&state.pool, input).await
}

#[tauri::command]
pub async fn update_provider(
    state: State<'_, AppState>,
    input: UpdateProviderInput,
) -> Result<ProviderView, String> {
    db::update_provider(&state.pool, input).await
}

#[tauri::command]
pub async fn delete_provider(state: State<'_, AppState>, id: String) -> Result<(), String> {
    db::delete_provider(&state.pool, &id).await
}

#[tauri::command]
pub async fn save_provider_credential(
    state: State<'_, AppState>,
    input: SaveCredentialInput,
) -> Result<ProviderView, String> {
    db::save_credential(&state.pool, input).await
}

#[tauri::command]
pub async fn import_agent_platform_service_account(
    state: State<'_, AppState>,
    input: ImportAgentPlatformServiceAccountInput,
) -> Result<ProviderView, String> {
    db::import_agent_platform_service_account(&state.pool, input).await
}

#[tauri::command]
pub async fn fetch_provider_models(
    state: State<'_, AppState>,
    provider_id: String,
) -> Result<Vec<ModelOption>, String> {
    let config = db::runtime_config(&state.pool, &provider_id).await?;
    RuntimeAdapter::new(state.client.clone(), config)
        .list_models()
        .await
}

#[tauri::command]
pub async fn test_model_connectivity(
    state: State<'_, AppState>,
    provider_id: String,
) -> Result<ConnectivityResult, String> {
    let config = db::runtime_config(&state.pool, &provider_id).await?;
    Ok(test_provider_connectivity(state.client.clone(), config).await)
}

#[tauri::command]
pub async fn get_flashcard_settings(
    state: State<'_, AppState>,
) -> Result<FlashcardSettings, String> {
    flashcards::get_settings(&state.pool).await
}

#[tauri::command]
pub async fn update_flashcard_settings(
    state: State<'_, AppState>,
    input: UpdateFlashcardSettingsInput,
) -> Result<FlashcardSettings, String> {
    flashcards::update_settings(&state.pool, input).await
}

#[tauri::command]
pub async fn export_config(state: State<'_, AppState>) -> Result<String, String> {
    flashcards::export_config(&state.pool).await
}

#[tauri::command]
pub async fn import_config(state: State<'_, AppState>, config_json: String) -> Result<(), String> {
    flashcards::import_config(&state.pool, config_json).await
}

#[tauri::command]
pub async fn pick_output_directory(app: AppHandle) -> Result<Option<String>, String> {
    let picked =
        tauri::async_runtime::spawn_blocking(move || app.dialog().file().blocking_pick_folder())
            .await
            .map_err(|error| error.to_string())?;
    picked
        .map(|path| {
            let path_buf: PathBuf = path
                .try_into()
                .map_err(|error| format!("Unable to resolve selected directory: {error}"))?;
            Ok(path_buf.to_string_lossy().to_string())
        })
        .transpose()
}

#[tauri::command]
pub async fn generate_flashcards(
    app: AppHandle,
    state: State<'_, AppState>,
    input: GenerateFlashcardsInput,
) -> Result<GenerateFlashcardsResult, String> {
    let _ = state.generation_cancel_tx.send(false);
    let progress_app = app.clone();
    let cancel_rx = state.generation_cancel_tx.subscribe();
    let result = flashcards::generate_flashcards_with_progress(
        &state.pool,
        state.client.clone(),
        &state.staging_root,
        input,
        move |progress| {
            let _ = progress_app.emit(GENERATION_PROGRESS_EVENT, progress);
        },
        cancel_rx,
    )
    .await;
    let _ = state.generation_cancel_tx.send(false);
    result
}

#[tauri::command]
pub async fn cancel_flashcard_generation(state: State<'_, AppState>) -> Result<(), String> {
    let _ = state.generation_cancel_tx.send(true);
    Ok(())
}

#[tauri::command]
pub async fn list_staged_cards(state: State<'_, AppState>) -> Result<Vec<StagedCardView>, String> {
    flashcards::list_staged_cards(&state.pool).await
}

#[tauri::command]
pub async fn read_staged_card(
    state: State<'_, AppState>,
    id: String,
) -> Result<StagedCardContent, String> {
    flashcards::read_staged_card(&state.pool, &id).await
}

#[tauri::command]
pub async fn save_staged_card(
    state: State<'_, AppState>,
    input: SaveStagedCardInput,
) -> Result<StagedCardView, String> {
    flashcards::save_staged_card(&state.pool, input).await
}

#[tauri::command]
pub async fn delete_staged_card(state: State<'_, AppState>, id: String) -> Result<(), String> {
    flashcards::delete_staged_card(&state.pool, &id).await
}

#[tauri::command]
pub async fn delete_all_staged_cards(state: State<'_, AppState>) -> Result<usize, String> {
    flashcards::delete_all_staged_cards(&state.pool).await
}

#[tauri::command]
pub async fn write_staged_cards(
    state: State<'_, AppState>,
) -> Result<WriteStagedCardsResult, String> {
    flashcards::write_staged_cards(&state.pool).await
}
