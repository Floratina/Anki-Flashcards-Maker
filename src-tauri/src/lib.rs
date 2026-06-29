mod adapters;
mod commands;
mod db;
mod domain;
mod flashcards;
mod vertex_ai;

use std::path::PathBuf;

use commands::AppState;
use tauri::Manager;

fn portable_data_root() -> Result<PathBuf, String> {
    let exe_path = std::env::current_exe()
        .map_err(|error| format!("Unable to resolve executable path: {error}"))?;
    let exe_dir = exe_path
        .parent()
        .ok_or_else(|| "Unable to resolve executable directory".to_string())?;
    Ok(exe_dir.join("flashcards-maker-data"))
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let app_data = portable_data_root()?;
            std::fs::create_dir_all(&app_data)
                .map_err(|error| format!("Unable to create app data directory: {error}"))?;
            let db_path = app_data.join("providers.sqlite3");
            if !db_path.exists() {
                if let Ok(legacy_data) = app.path().app_data_dir() {
                    let legacy_db_path = legacy_data.join("providers.sqlite3");
                    if legacy_db_path.exists() {
                        std::fs::copy(&legacy_db_path, &db_path).map_err(|error| {
                            format!("Unable to migrate existing app data: {error}")
                        })?;
                    }
                }
            }
            let pool = tauri::async_runtime::block_on(db::connect(&db_path))?;
            let client = reqwest::Client::builder()
                .user_agent("flashcards-maker/0.1.0")
                .build()
                .map_err(|error| format!("Unable to build HTTP client: {error}"))?;
            let staging_root = app_data.join(".flashcards_staging");
            let (generation_cancel_tx, _) = tokio::sync::watch::channel(false);
            app.manage(AppState {
                pool,
                client,
                staging_root,
                generation_cancel_tx,
            });
            if let Some(window) = app.get_webview_window("main") {
                if let Ok(icon) =
                    tauri::image::Image::from_bytes(include_bytes!("../icons/icon.png"))
                {
                    let _ = window.set_icon(icon);
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_providers,
            commands::create_provider,
            commands::update_provider,
            commands::delete_provider,
            commands::save_provider_credential,
            commands::import_agent_platform_service_account,
            commands::fetch_provider_models,
            commands::test_model_connectivity,
            commands::get_flashcard_settings,
            commands::update_flashcard_settings,
            commands::export_config,
            commands::import_config,
            commands::pick_output_directory,
            commands::generate_flashcards,
            commands::cancel_flashcard_generation,
            commands::list_staged_cards,
            commands::read_staged_card,
            commands::save_staged_card,
            commands::delete_staged_card,
            commands::delete_all_staged_cards,
            commands::write_staged_cards,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Flashcards Maker");
}
