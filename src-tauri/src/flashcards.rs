use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};
use tokio::sync::watch;
use tokio::task::JoinSet;
use tokio::time::sleep;

use crate::adapters::RuntimeAdapter;
use crate::db;
use crate::domain::{
    clamp_concurrency_limit, clamp_retry_count, CredentialKind, FlashcardSettings,
    GenerateFlashcardsInput, GenerateFlashcardsProgress, GenerateFlashcardsResult,
    ProviderCallError, ProviderProtocol, ProviderView, SaveStagedCardInput, StagedCardContent,
    StagedCardStatus, StagedCardView, UpdateFlashcardSettingsInput, VertexConfig,
    WriteStagedCardsResult, DEFAULT_CONCURRENCY_LIMIT, DEFAULT_RETRY_COUNT, MAX_CONCURRENCY_LIMIT,
    MAX_RETRY_COUNT, MIN_CONCURRENCY_LIMIT, MIN_RETRY_COUNT,
};
use crate::vertex_ai;

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

const DEFAULT_FLASHCARD_PROMPT: &str =
    "根据用户输入生成一张 Markdown 词卡。只输出 Markdown 正文，不要寒暄，不要代码块。";

pub async fn migrate(pool: &SqlitePool) -> Result<(), String> {
    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"#,
    )
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;

    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS staged_flashcards (
            id TEXT PRIMARY KEY,
            source_entry TEXT NOT NULL,
            filename TEXT NOT NULL,
            staged_path TEXT NOT NULL,
            input_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            warnings_json TEXT NOT NULL DEFAULT '[]',
            error TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"#,
    )
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;
    ensure_staged_input_index_column(pool).await?;
    Ok(())
}

pub async fn get_settings(pool: &SqlitePool) -> Result<FlashcardSettings, String> {
    let flashcard_prompt = read_setting(pool, "flashcardPrompt")
        .await?
        .unwrap_or_else(|| DEFAULT_FLASHCARD_PROMPT.to_string());
    let output_directory = read_setting(pool, "outputDirectory")
        .await?
        .unwrap_or_default();
    let selected_provider_id = read_setting(pool, "selectedProviderId")
        .await?
        .filter(|value| !value.trim().is_empty())
        .or(first_provider_id(pool).await?)
        .unwrap_or_default();
    let concurrency_limit = read_number_setting(
        pool,
        "concurrencyLimit",
        DEFAULT_CONCURRENCY_LIMIT,
        MIN_CONCURRENCY_LIMIT,
        MAX_CONCURRENCY_LIMIT,
    )
    .await?;
    let retry_count = read_number_setting(
        pool,
        "retryCount",
        DEFAULT_RETRY_COUNT,
        MIN_RETRY_COUNT,
        MAX_RETRY_COUNT,
    )
    .await?;
    Ok(FlashcardSettings {
        flashcard_prompt,
        output_directory,
        selected_provider_id,
        concurrency_limit,
        retry_count,
    })
}

async fn first_provider_id(pool: &SqlitePool) -> Result<Option<String>, String> {
    sqlx::query_scalar::<_, String>(
        "SELECT id FROM providers ORDER BY is_builtin DESC, sort_order ASC, created_at ASC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .map_err(|error| error.to_string())
}

pub async fn update_settings(
    pool: &SqlitePool,
    input: UpdateFlashcardSettingsInput,
) -> Result<FlashcardSettings, String> {
    write_setting(pool, "flashcardPrompt", input.flashcard_prompt.trim()).await?;
    write_setting(pool, "outputDirectory", input.output_directory.trim()).await?;
    write_setting(
        pool,
        "selectedProviderId",
        input.selected_provider_id.trim(),
    )
    .await?;
    write_setting(
        pool,
        "concurrencyLimit",
        &clamp_concurrency_limit(input.concurrency_limit).to_string(),
    )
    .await?;
    write_setting(
        pool,
        "retryCount",
        &clamp_retry_count(input.retry_count).to_string(),
    )
    .await?;
    get_settings(pool).await
}

pub async fn export_config(pool: &SqlitePool) -> Result<String, String> {
    let settings = get_settings(pool).await?;
    let providers = export_providers(pool).await?;
    let value = AppConfigExport {
        version: 1,
        settings,
        providers,
    };
    serde_json::to_string_pretty(&value).map_err(|error| error.to_string())
}

pub async fn import_config(pool: &SqlitePool, config_json: String) -> Result<(), String> {
    let value: AppConfigImport = serde_json::from_str(&config_json)
        .map_err(|error| format!("Config JSON is invalid: {error}"))?;
    update_settings(pool, value.settings).await?;
    for provider in value.providers {
        import_provider(pool, provider).await?;
    }
    Ok(())
}

#[cfg(test)]
async fn generate_flashcards(
    pool: &SqlitePool,
    client: Client,
    staging_root: &Path,
    input: GenerateFlashcardsInput,
) -> Result<GenerateFlashcardsResult, String> {
    let (_cancel_tx, cancel_rx) = watch::channel(false);
    generate_flashcards_with_progress(pool, client, staging_root, input, |_| {}, cancel_rx).await
}

pub async fn generate_flashcards_with_progress<F>(
    pool: &SqlitePool,
    client: Client,
    staging_root: &Path,
    input: GenerateFlashcardsInput,
    mut report_progress: F,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<GenerateFlashcardsResult, String>
where
    F: FnMut(GenerateFlashcardsProgress) + Send,
{
    let entries = parse_entries(&input.entries_text);
    let total = entries.len();
    if entries.is_empty() {
        return Err("请输入至少一个非空条目。".into());
    }

    let current = get_settings(pool).await?;
    update_settings(
        pool,
        UpdateFlashcardSettingsInput {
            flashcard_prompt: input.flashcard_prompt.clone(),
            output_directory: current.output_directory,
            selected_provider_id: input.provider_id.clone(),
            concurrency_limit: input.concurrency_limit,
            retry_count: input.retry_count,
        },
    )
    .await?;

    reset_staging(pool, staging_root).await?;
    fs::create_dir_all(staging_root)
        .map_err(|error| format!("Unable to create staging directory: {error}"))?;

    let runtime = db::runtime_config(pool, &input.provider_id).await?;
    let adapter = RuntimeAdapter::new(client, runtime);
    let mut used_names = HashSet::new();
    let jobs = entries
        .into_iter()
        .enumerate()
        .map(|(index, entry)| {
            let id = next_id("card");
            let filename = unique_filename(&entry, &mut used_names);
            let staged_path = staging_root.join(&filename);
            GenerationJob {
                input_index: index as i64,
                id,
                entry,
                filename,
                staged_path,
            }
        })
        .collect::<Vec<_>>();
    let concurrency_limit = clamp_concurrency_limit(input.concurrency_limit) as usize;
    let retry_count = clamp_retry_count(input.retry_count) as usize;
    let prompt = input.flashcard_prompt.clone();
    let mut pending_jobs = jobs.into_iter();
    let mut tasks = JoinSet::new();
    let mut cards = Vec::new();
    let mut completed = 0;
    let mut in_progress = 0;
    let mut generated = 0;
    let mut failed = 0;
    let mut cancelled = false;

    report_progress(generation_progress(
        total,
        completed,
        in_progress,
        generated,
        failed,
        None,
        cancelled,
    ));

    fill_generation_tasks(
        &mut tasks,
        &mut pending_jobs,
        &adapter,
        &prompt,
        retry_count,
        concurrency_limit,
        &mut in_progress,
    );
    report_progress(generation_progress(
        total,
        completed,
        in_progress,
        generated,
        failed,
        None,
        cancelled,
    ));

    while in_progress > 0 {
        if *cancel_rx.borrow() {
            cancelled = true;
            abort_generation_tasks(&mut tasks, &mut in_progress).await;
            report_progress(generation_progress(
                total,
                completed,
                in_progress,
                generated,
                failed,
                None,
                cancelled,
            ));
            break;
        }

        let joined = tokio::select! {
            joined = tasks.join_next() => joined,
            changed = cancel_rx.changed() => {
                if changed.is_ok() && *cancel_rx.borrow() {
                    cancelled = true;
                    abort_generation_tasks(&mut tasks, &mut in_progress).await;
                    report_progress(generation_progress(
                        total,
                        completed,
                        in_progress,
                        generated,
                        failed,
                        None,
                        cancelled,
                    ));
                    break;
                }
                continue;
            }
        };
        let Some(joined) = joined else {
            in_progress = 0;
            break;
        };

        in_progress = in_progress.saturating_sub(1);
        let result = joined.map_err(|error| format!("Generation task failed: {error}"))?;
        let job = result.job;
        report_progress(generation_progress(
            total,
            completed,
            in_progress,
            generated,
            failed,
            Some(job.entry.clone()),
            cancelled,
        ));

        match result.output {
            Ok(raw) => {
                let cleaned = clean_model_output(&raw);
                fs::write(&job.staged_path, &cleaned.content)
                    .map_err(|error| format!("Unable to write staged markdown: {error}"))?;
                insert_staged_card(
                    pool,
                    &job.id,
                    job.input_index,
                    &job.entry,
                    &job.filename,
                    &job.staged_path,
                    &cleaned.content,
                    StagedCardStatus::Ready,
                    &cleaned.warnings,
                    None,
                )
                .await?;
            }
            Err(error) => {
                insert_staged_card(
                    pool,
                    &job.id,
                    job.input_index,
                    &job.entry,
                    &job.filename,
                    &job.staged_path,
                    "",
                    StagedCardStatus::Failed,
                    &[],
                    Some(error.message()),
                )
                .await?;
            }
        }
        let card = get_staged_card(pool, &job.id).await?;
        match card.status {
            StagedCardStatus::Ready => generated += 1,
            StagedCardStatus::Failed => failed += 1,
            StagedCardStatus::Written => {}
        }
        cards.push(card);
        completed += 1;
        report_progress(generation_progress(
            total,
            completed,
            in_progress,
            generated,
            failed,
            Some(job.entry),
            cancelled,
        ));

        if *cancel_rx.borrow() {
            cancelled = true;
            abort_generation_tasks(&mut tasks, &mut in_progress).await;
            report_progress(generation_progress(
                total,
                completed,
                in_progress,
                generated,
                failed,
                None,
                cancelled,
            ));
            break;
        }

        fill_generation_tasks(
            &mut tasks,
            &mut pending_jobs,
            &adapter,
            &prompt,
            retry_count,
            concurrency_limit,
            &mut in_progress,
        );
        report_progress(generation_progress(
            total,
            completed,
            in_progress,
            generated,
            failed,
            None,
            cancelled,
        ));
    }

    cards.sort_by_key(|card| card.input_index);
    report_progress(generation_progress(
        total,
        completed,
        in_progress,
        generated,
        failed,
        None,
        cancelled,
    ));

    Ok(GenerateFlashcardsResult {
        cards,
        generated,
        failed,
        cancelled,
    })
}

fn generation_progress(
    total: usize,
    completed: usize,
    in_progress: usize,
    generated: usize,
    failed: usize,
    current_entry: Option<String>,
    cancelled: bool,
) -> GenerateFlashcardsProgress {
    GenerateFlashcardsProgress {
        total,
        completed,
        in_progress,
        generated,
        failed,
        current_entry,
        cancelled,
    }
}

async fn abort_generation_tasks(
    tasks: &mut JoinSet<GenerationTaskResult>,
    in_progress: &mut usize,
) {
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    *in_progress = 0;
}
#[derive(Debug)]
struct GenerationJob {
    input_index: i64,
    id: String,
    entry: String,
    filename: String,
    staged_path: PathBuf,
}

#[derive(Debug)]
struct GenerationTaskResult {
    job: GenerationJob,
    output: Result<String, ProviderCallError>,
}

fn fill_generation_tasks(
    tasks: &mut JoinSet<GenerationTaskResult>,
    pending_jobs: &mut std::vec::IntoIter<GenerationJob>,
    adapter: &RuntimeAdapter,
    prompt: &str,
    retry_count: usize,
    concurrency_limit: usize,
    in_progress: &mut usize,
) {
    while *in_progress < concurrency_limit {
        let Some(job) = pending_jobs.next() else {
            break;
        };
        let adapter = adapter.clone();
        let prompt = prompt.to_string();
        *in_progress += 1;
        tasks.spawn(async move {
            let output = generate_with_retries(adapter, &prompt, &job.entry, retry_count).await;
            GenerationTaskResult { job, output }
        });
    }
}

async fn generate_with_retries(
    adapter: RuntimeAdapter,
    prompt: &str,
    entry: &str,
    retry_count: usize,
) -> Result<String, ProviderCallError> {
    let max_attempts = retry_count.saturating_add(1);
    for attempt_index in 0..max_attempts {
        match adapter.generate_flashcard(prompt, entry).await {
            Ok(output) => return Ok(output),
            Err(error) if error.is_retryable() && attempt_index + 1 < max_attempts => {
                sleep(retry_delay(attempt_index)).await;
            }
            Err(error) => return Err(error),
        }
    }
    Err(ProviderCallError::fatal(
        "Generation failed without an error",
    ))
}

fn retry_delay(attempt_index: usize) -> Duration {
    let multiplier = 1_u64 << attempt_index.min(3);
    Duration::from_millis((500 * multiplier).min(4000))
}

pub async fn list_staged_cards(pool: &SqlitePool) -> Result<Vec<StagedCardView>, String> {
    let rows = sqlx::query(
        "SELECT * FROM staged_flashcards ORDER BY input_index ASC, created_at ASC, filename ASC",
    )
    .fetch_all(pool)
    .await
    .map_err(|error| error.to_string())?;
    rows.iter().map(staged_card_from_row).collect()
}

pub async fn read_staged_card(pool: &SqlitePool, id: &str) -> Result<StagedCardContent, String> {
    let row = staged_card_row(pool, id).await?;
    let card = staged_card_from_row(&row)?;
    let content =
        fs::read_to_string(&card.staged_path).unwrap_or_else(|_| row.get::<String, _>("content"));
    Ok(StagedCardContent { card, content })
}

pub async fn save_staged_card(
    pool: &SqlitePool,
    input: SaveStagedCardInput,
) -> Result<StagedCardView, String> {
    let card = get_staged_card(pool, &input.id).await?;
    if let Some(parent) = Path::new(&card.staged_path).parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Unable to create staging directory: {error}"))?;
    }
    fs::write(&card.staged_path, &input.content)
        .map_err(|error| format!("Unable to save staged markdown: {error}"))?;
    sqlx::query(
        "UPDATE staged_flashcards SET content = ?, status = ?, error = NULL, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
    )
    .bind(&input.content)
    .bind(StagedCardStatus::Ready.as_str())
    .bind(&input.id)
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;
    get_staged_card(pool, &input.id).await
}

pub async fn delete_staged_card(pool: &SqlitePool, id: &str) -> Result<(), String> {
    let card = get_staged_card(pool, id).await?;
    delete_staged_markdown_file(&card)?;
    sqlx::query("DELETE FROM staged_flashcards WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await
        .map_err(|error| error.to_string())?;
    Ok(())
}

pub async fn delete_all_staged_cards(pool: &SqlitePool) -> Result<usize, String> {
    let cards = list_staged_cards(pool).await?;
    for card in &cards {
        delete_staged_markdown_file(card)?;
    }
    sqlx::query("DELETE FROM staged_flashcards")
        .execute(pool)
        .await
        .map_err(|error| error.to_string())?;
    Ok(cards.len())
}

pub async fn write_staged_cards(pool: &SqlitePool) -> Result<WriteStagedCardsResult, String> {
    let settings = get_settings(pool).await?;
    let output_directory = settings.output_directory.trim();
    if output_directory.is_empty() {
        return Err("请先选择输出路径".into());
    }

    let output_root = PathBuf::from(output_directory);
    fs::create_dir_all(&output_root)
        .map_err(|error| format!("Unable to create output directory: {error}"))?;

    let cards = list_staged_cards(pool).await?;
    let mut written_files = Vec::new();
    let mut reserved = HashSet::new();
    for card in cards
        .into_iter()
        .filter(|card| card.status == StagedCardStatus::Ready)
    {
        let content = read_card_content(pool, &card).await?;
        if content.trim().is_empty() {
            continue;
        }
        let destination = unique_available_path(&output_root, &card.filename, &mut reserved);
        fs::write(&destination, content)
            .map_err(|error| format!("Unable to write output markdown: {error}"))?;
        sqlx::query(
            "UPDATE staged_flashcards SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        )
        .bind(StagedCardStatus::Written.as_str())
        .bind(&card.id)
        .execute(pool)
        .await
        .map_err(|error| error.to_string())?;
        written_files.push(destination.to_string_lossy().to_string());
    }

    Ok(WriteStagedCardsResult {
        written: written_files.len(),
        files: written_files,
    })
}

async fn read_setting(pool: &SqlitePool, key: &str) -> Result<Option<String>, String> {
    sqlx::query_scalar("SELECT value FROM app_settings WHERE key = ?")
        .bind(key)
        .fetch_optional(pool)
        .await
        .map_err(|error| error.to_string())
}

async fn read_number_setting(
    pool: &SqlitePool,
    key: &str,
    default: u32,
    min: u32,
    max: u32,
) -> Result<u32, String> {
    let value = read_setting(pool, key)
        .await?
        .and_then(|raw| raw.trim().parse::<u32>().ok())
        .unwrap_or(default);
    Ok(value.clamp(min, max))
}

async fn write_setting(pool: &SqlitePool, key: &str, value: &str) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO app_settings (key, value, updated_at)
         VALUES (?, ?, CURRENT_TIMESTAMP)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
    )
    .bind(key)
    .bind(value)
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;
    Ok(())
}

async fn reset_staging(pool: &SqlitePool, staging_root: &Path) -> Result<(), String> {
    if staging_root
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == ".flashcards_staging")
        && staging_root.exists()
    {
        fs::remove_dir_all(staging_root)
            .map_err(|error| format!("Unable to reset staging directory: {error}"))?;
    }
    sqlx::query("DELETE FROM staged_flashcards")
        .execute(pool)
        .await
        .map_err(|error| error.to_string())?;
    Ok(())
}

async fn ensure_staged_input_index_column(pool: &SqlitePool) -> Result<(), String> {
    let rows = sqlx::query("PRAGMA table_info(staged_flashcards)")
        .fetch_all(pool)
        .await
        .map_err(|error| error.to_string())?;
    let has_input_index = rows.iter().any(|row| {
        let name: String = row.get("name");
        name == "input_index"
    });
    if !has_input_index {
        sqlx::query(
            "ALTER TABLE staged_flashcards ADD COLUMN input_index INTEGER NOT NULL DEFAULT 0",
        )
        .execute(pool)
        .await
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn parse_entries(entries_text: &str) -> Vec<String> {
    entries_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CleanedOutput {
    content: String,
    warnings: Vec<String>,
}

fn clean_model_output(raw: &str) -> CleanedOutput {
    let mut warnings = Vec::new();
    let mut content = raw
        .trim_start_matches('\u{feff}')
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .trim()
        .to_string();
    if let Some(unwrapped) = unwrap_full_code_fence(&content) {
        content = unwrapped;
        warnings.push("Removed outer code fence".to_string());
    }
    let (trimmed, chatter_warnings) = trim_obvious_chatter(&content);
    content = trimmed;
    warnings.extend(chatter_warnings);
    CleanedOutput { content, warnings }
}

fn unwrap_full_code_fence(content: &str) -> Option<String> {
    let trimmed = content.trim();
    if !trimmed.starts_with("```") || !trimmed.ends_with("```") {
        return None;
    }
    let mut lines: Vec<&str> = trimmed.lines().collect();
    if lines.len() < 2 {
        return None;
    }
    let first = lines.first()?.trim();
    let last = lines.last()?.trim();
    if !first.starts_with("```") || last != "```" {
        return None;
    }
    let language = first.trim_start_matches("```").trim().to_ascii_lowercase();
    if !language.is_empty() && !matches!(language.as_str(), "md" | "markdown" | "text") {
        return None;
    }
    lines.remove(0);
    lines.pop();
    Some(lines.join("\n").trim().to_string())
}

fn trim_obvious_chatter(content: &str) -> (String, Vec<String>) {
    let mut warnings = Vec::new();
    let mut paragraphs = split_paragraphs(content);
    if paragraphs.len() > 1 && is_leading_chatter(&paragraphs[0]) {
        paragraphs.remove(0);
        warnings.push("Removed leading assistant chatter".to_string());
    }
    if paragraphs.len() > 1 && is_trailing_chatter(paragraphs.last().unwrap_or(&String::new())) {
        paragraphs.pop();
        warnings.push("Removed trailing assistant chatter".to_string());
    }
    (paragraphs.join("\n\n").trim().to_string(), warnings)
}

fn split_paragraphs(content: &str) -> Vec<String> {
    content
        .split("\n\n")
        .map(str::trim)
        .filter(|paragraph| !paragraph.is_empty())
        .map(str::to_string)
        .collect()
}

fn is_leading_chatter(paragraph: &str) -> bool {
    let normalized = paragraph.trim().to_ascii_lowercase();
    paragraph.chars().count() <= 80
        && !paragraph.contains('\n')
        && (normalized.starts_with("sure")
            || normalized.starts_with("here is")
            || normalized.starts_with("of course")
            || paragraph.starts_with("好的")
            || paragraph.starts_with("当然")
            || paragraph.starts_with("以下是"))
}

fn is_trailing_chatter(paragraph: &str) -> bool {
    let normalized = paragraph.trim().to_ascii_lowercase();
    paragraph.chars().count() <= 80
        && !paragraph.contains('\n')
        && (normalized.starts_with("hope this helps")
            || paragraph.starts_with("希望")
            || paragraph.starts_with("如果你"))
}

fn unique_filename(entry: &str, used_names: &mut HashSet<String>) -> String {
    let stem = sanitize_filename_stem(entry);
    let mut index = 1;
    loop {
        let candidate = if index == 1 {
            format!("{stem}.md")
        } else {
            format!("{stem}-{index}.md")
        };
        let key = candidate.to_ascii_lowercase();
        if used_names.insert(key) {
            return candidate;
        }
        index += 1;
    }
}

fn sanitize_filename_stem(entry: &str) -> String {
    let mut out = String::new();
    for ch in entry.chars() {
        if matches!(ch, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') || ch.is_control() {
            out.push('-');
        } else {
            out.push(ch);
        }
    }
    let compact = out
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(['.', ' ', '-'])
        .chars()
        .take(80)
        .collect::<String>()
        .trim_matches(['.', ' ', '-'])
        .to_string();
    if compact.is_empty() {
        "card".to_string()
    } else {
        compact
    }
}

fn unique_available_path(
    output_root: &Path,
    filename: &str,
    reserved: &mut HashSet<String>,
) -> PathBuf {
    let stem = filename.trim_end_matches(".md");
    let mut index = 1;
    loop {
        let candidate = if index == 1 {
            output_root.join(format!("{stem}.md"))
        } else {
            output_root.join(format!("{stem}-{index}.md"))
        };
        let key = candidate.to_string_lossy().to_ascii_lowercase();
        if !candidate.exists() && reserved.insert(key) {
            return candidate;
        }
        index += 1;
    }
}

fn is_staged_markdown_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
        && path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == ".flashcards_staging")
}

fn delete_staged_markdown_file(card: &StagedCardView) -> Result<(), String> {
    let staged_path = PathBuf::from(&card.staged_path);
    if !is_staged_markdown_path(&staged_path) || !staged_path.exists() {
        return Ok(());
    }
    fs::remove_file(&staged_path)
        .map_err(|error| format!("Unable to delete staged markdown: {error}"))
}

async fn read_card_content(pool: &SqlitePool, card: &StagedCardView) -> Result<String, String> {
    match fs::read_to_string(&card.staged_path) {
        Ok(content) => Ok(content),
        Err(_) => {
            let row = staged_card_row(pool, &card.id).await?;
            Ok(row.get("content"))
        }
    }
}

async fn insert_staged_card(
    pool: &SqlitePool,
    id: &str,
    input_index: i64,
    source_entry: &str,
    filename: &str,
    staged_path: &Path,
    content: &str,
    status: StagedCardStatus,
    warnings: &[String],
    error: Option<&str>,
) -> Result<(), String> {
    let warnings_json = serde_json::to_string(warnings).map_err(|error| error.to_string())?;
    sqlx::query(
        "INSERT INTO staged_flashcards (
            id, input_index, source_entry, filename, staged_path, content, status, warnings_json, error
         )
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(id)
    .bind(input_index)
    .bind(source_entry)
    .bind(filename)
    .bind(staged_path.to_string_lossy().to_string())
    .bind(content)
    .bind(status.as_str())
    .bind(warnings_json)
    .bind(error)
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;
    Ok(())
}

async fn get_staged_card(pool: &SqlitePool, id: &str) -> Result<StagedCardView, String> {
    let row = staged_card_row(pool, id).await?;
    staged_card_from_row(&row)
}

async fn staged_card_row(pool: &SqlitePool, id: &str) -> Result<sqlx::sqlite::SqliteRow, String> {
    sqlx::query("SELECT * FROM staged_flashcards WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "Staged card not found".to_string())
}

fn staged_card_from_row(row: &sqlx::sqlite::SqliteRow) -> Result<StagedCardView, String> {
    let warnings_json: String = row.get("warnings_json");
    let warnings = serde_json::from_str::<Vec<String>>(&warnings_json).unwrap_or_default();
    Ok(StagedCardView {
        id: row.get("id"),
        input_index: row.get("input_index"),
        source_entry: row.get("source_entry"),
        filename: row.get("filename"),
        staged_path: row.get("staged_path"),
        status: StagedCardStatus::from_db(&row.get::<String, _>("status")),
        warnings,
        error: row.get("error"),
        updated_at: row.get("updated_at"),
    })
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AppConfigExport {
    version: u32,
    settings: FlashcardSettings,
    providers: Vec<ProviderConfigJson>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppConfigImport {
    settings: UpdateFlashcardSettingsInput,
    #[serde(default)]
    providers: Vec<ProviderConfigJson>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProviderConfigJson {
    id: String,
    name: String,
    protocol: ProviderProtocol,
    base_url: String,
    credential_kind: CredentialKind,
    selected_model: String,
    system_prompt: String,
    thinking_level: crate::domain::ThinkingLevel,
    web_enabled: bool,
    is_builtin: bool,
    vertex: VertexConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    credential: Option<String>,
}

async fn export_providers(pool: &SqlitePool) -> Result<Vec<ProviderConfigJson>, String> {
    let providers = db::list_providers(pool).await?;
    Ok(providers.into_iter().map(provider_to_config).collect())
}

fn provider_to_config(provider: ProviderView) -> ProviderConfigJson {
    ProviderConfigJson {
        id: provider.id,
        name: provider.name,
        protocol: provider.protocol,
        base_url: provider.base_url,
        credential_kind: provider.credential_kind,
        selected_model: provider.selected_model,
        system_prompt: provider.system_prompt,
        thinking_level: provider.thinking_level,
        web_enabled: provider.web_enabled,
        is_builtin: provider.is_builtin,
        vertex: provider.vertex,
        credential: None,
    }
}

async fn import_provider(pool: &SqlitePool, provider: ProviderConfigJson) -> Result<(), String> {
    if provider.id.trim().is_empty() || provider.name.trim().is_empty() {
        return Err("Provider config contains an empty id or name".into());
    }
    sqlx::query(
        "INSERT INTO providers (
            id, name, protocol, base_url, credential_kind, credential, selected_model,
            system_prompt, thinking_level, web_enabled, vertex_project_id, vertex_location,
            vertex_client_email, is_builtin, sort_order
         )
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 999)
         ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            protocol = excluded.protocol,
            base_url = excluded.base_url,
            credential_kind = excluded.credential_kind,
            credential = COALESCE(excluded.credential, providers.credential),
            selected_model = excluded.selected_model,
            system_prompt = excluded.system_prompt,
            thinking_level = excluded.thinking_level,
            web_enabled = excluded.web_enabled,
            vertex_project_id = excluded.vertex_project_id,
            vertex_location = excluded.vertex_location,
            vertex_client_email = excluded.vertex_client_email,
            updated_at = CURRENT_TIMESTAMP",
    )
    .bind(provider.id.trim())
    .bind(provider.name.trim())
    .bind(provider.protocol.as_str())
    .bind(provider.base_url.trim())
    .bind(provider.credential_kind.as_str())
    .bind(
        provider
            .credential
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty()),
    )
    .bind(provider.selected_model.trim())
    .bind(provider.system_prompt.trim())
    .bind(provider.thinking_level.as_str())
    .bind(if provider.web_enabled { 1_i64 } else { 0_i64 })
    .bind(provider.vertex.project_id.trim())
    .bind(
        provider
            .vertex
            .location
            .trim()
            .if_empty(vertex_ai::DEFAULT_LOCATION),
    )
    .bind(provider.vertex.client_email.trim())
    .bind(if provider.is_builtin { 1_i64 } else { 0_i64 })
    .execute(pool)
    .await
    .map_err(|error| error.to_string())?;
    Ok(())
}

fn next_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{nanos:x}{counter:x}")
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::Arc;
    use std::time::{Duration as StdDuration, Instant};

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("flashcards-maker-{name}-{nanos}"))
    }

    async fn test_pool() -> (SqlitePool, PathBuf) {
        let path = temp_path("db").with_extension("sqlite3");
        let pool = db::connect(&path).await.expect("connect");
        (pool, path)
    }

    struct MockHttpResponse {
        status: u16,
        body: String,
    }

    fn openai_response_body(text: &str) -> String {
        format!(
            r#"{{"output":[{{"type":"message","content":[{{"type":"output_text","text":"{}"}}]}}]}}"#,
            text.replace('\\', "\\\\").replace('"', "\\\"")
        )
    }

    fn status_reason(status: u16) -> &'static str {
        match status {
            200 => "OK",
            400 => "Bad Request",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            _ => "Status",
        }
    }

    fn spawn_mock_openai_server<F>(
        expected_requests: usize,
        handler: F,
    ) -> (String, std::thread::JoinHandle<()>)
    where
        F: Fn(usize, String) -> MockHttpResponse + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let address = listener.local_addr().expect("address").to_string();
        let handler = Arc::new(handler);
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + StdDuration::from_secs(8);
            let mut accepted = 0;
            let mut workers = Vec::new();
            while accepted < expected_requests && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        accepted += 1;
                        let request_index = accepted;
                        let handler = Arc::clone(&handler);
                        workers.push(std::thread::spawn(move || {
                            let mut request = [0_u8; 8192];
                            let read = stream.read(&mut request).unwrap_or_default();
                            let request_text =
                                String::from_utf8_lossy(&request[..read]).to_string();
                            let response = handler(request_index, request_text);
                            write!(
                                stream,
                                "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                response.status,
                                status_reason(response.status),
                                response.body.len(),
                                response.body
                            )
                            .expect("write");
                        }));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(StdDuration::from_millis(10));
                    }
                    Err(error) => panic!("accept failed: {error}"),
                }
            }
            assert_eq!(accepted, expected_requests, "mock server request count");
            for worker in workers {
                worker.join().expect("worker");
            }
        });
        (format!("http://{address}"), server)
    }

    fn spawn_flexible_mock_openai_server<F>(
        max_requests: usize,
        idle_timeout: StdDuration,
        handler: F,
    ) -> (String, std::thread::JoinHandle<()>)
    where
        F: Fn(usize, String) -> MockHttpResponse + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let address = listener.local_addr().expect("address").to_string();
        let handler = Arc::new(handler);
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + StdDuration::from_secs(8);
            let mut accepted = 0;
            let mut last_accept = Instant::now();
            let mut workers = Vec::new();
            while accepted < max_requests
                && Instant::now() < deadline
                && (accepted == 0 || last_accept.elapsed() < idle_timeout)
            {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        accepted += 1;
                        last_accept = Instant::now();
                        let request_index = accepted;
                        let handler = Arc::clone(&handler);
                        workers.push(std::thread::spawn(move || {
                            let mut request = [0_u8; 8192];
                            let read = stream.read(&mut request).unwrap_or_default();
                            let request_text =
                                String::from_utf8_lossy(&request[..read]).to_string();
                            let response = handler(request_index, request_text);
                            write!(
                                stream,
                                "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                response.status,
                                status_reason(response.status),
                                response.body.len(),
                                response.body
                            )
                            .expect("write");
                        }));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(StdDuration::from_millis(5));
                    }
                    Err(error) => panic!("accept failed: {error}"),
                }
            }
            assert!(accepted > 0, "mock server accepted no requests");
            for worker in workers {
                worker.join().expect("worker");
            }
        });
        (format!("http://{address}"), server)
    }

    async fn point_openai_provider_at(pool: &SqlitePool, base_url: String) {
        db::update_provider(
            pool,
            crate::domain::UpdateProviderInput {
                id: "builtin_openai".into(),
                name: "OpenAI".into(),
                base_url,
                credential_kind: CredentialKind::Bearer,
                selected_model: "gpt-5-mini".into(),
                system_prompt: String::new(),
                thinking_level: crate::domain::ThinkingLevel::None,
                web_enabled: false,
                vertex_project_id: String::new(),
                vertex_location: vertex_ai::DEFAULT_LOCATION.into(),
                vertex_client_email: String::new(),
            },
        )
        .await
        .expect("provider");
    }

    #[test]
    fn cleans_outer_code_fence_and_preserves_markdown() {
        let cleaned = clean_model_output("```markdown\n# Word\n\n- Meaning\n```");
        assert_eq!(cleaned.content, "# Word\n\n- Meaning");
        assert_eq!(cleaned.warnings, vec!["Removed outer code fence"]);

        let preserved = clean_model_output("# Word\n\n```js\nconst ok = true\n```");
        assert!(preserved.content.contains("```js"));
    }

    #[test]
    fn trims_obvious_chatter_only_around_content() {
        let cleaned = clean_model_output("Sure, here is the card:\n\n# apple\n\nHope this helps!");
        assert_eq!(cleaned.content, "# apple");
        assert_eq!(
            cleaned.warnings,
            vec![
                "Removed leading assistant chatter",
                "Removed trailing assistant chatter"
            ]
        );

        let preserved = clean_model_output("# Sure\n\nThis is valid content.");
        assert_eq!(preserved.content, "# Sure\n\nThis is valid content.");
    }

    #[test]
    fn filename_sanitization_and_duplicates_are_stable() {
        let mut used = HashSet::new();
        assert_eq!(unique_filename("a/b:c*?", &mut used), "a-b-c.md");
        assert_eq!(unique_filename("a/b:c*?", &mut used), "a-b-c-2.md");
        assert_eq!(unique_filename("...", &mut used), "card.md");
    }

    #[tokio::test]
    async fn config_export_excludes_secrets_and_import_preserves_them() {
        let (pool, db_path) = test_pool().await;
        db::save_credential(
            &pool,
            crate::domain::SaveCredentialInput {
                provider_id: "builtin_openai".into(),
                credential: Some("sk-test-secret".into()),
            },
        )
        .await
        .expect("credential");
        let exported = export_config(&pool).await.expect("export");
        assert!(!exported.contains("sk-test-secret"));
        assert!(!exported.contains("\"credential\""));
        import_config(&pool, exported).await.expect("import");
        let runtime = db::runtime_config(&pool, "builtin_openai")
            .await
            .expect("runtime");
        assert_eq!(runtime.credential.as_deref(), Some("sk-test-secret"));
        pool.close().await;
        let _ = fs::remove_file(db_path);
    }

    #[tokio::test]
    async fn one_line_generates_one_staged_file() {
        let (pool, db_path) = test_pool().await;
        let staging = temp_path("stage").join(".flashcards_staging");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("address");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request);
            let body = r##"{"output":[{"type":"message","content":[{"type":"output_text","text":"# apple"}]}]}"##;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .expect("write");
        });
        db::update_provider(
            &pool,
            crate::domain::UpdateProviderInput {
                id: "builtin_openai".into(),
                name: "OpenAI".into(),
                base_url: format!("http://{address}"),
                credential_kind: CredentialKind::Bearer,
                selected_model: "gpt-5-mini".into(),
                system_prompt: String::new(),
                thinking_level: crate::domain::ThinkingLevel::None,
                web_enabled: false,
                vertex_project_id: String::new(),
                vertex_location: vertex_ai::DEFAULT_LOCATION.into(),
                vertex_client_email: String::new(),
            },
        )
        .await
        .expect("provider");
        let result = generate_flashcards(
            &pool,
            Client::new(),
            &staging,
            GenerateFlashcardsInput {
                entries_text: "apple".into(),
                provider_id: "builtin_openai".into(),
                flashcard_prompt: "Make card".into(),
                concurrency_limit: 1,
                retry_count: 0,
            },
        )
        .await
        .expect("generate");
        assert_eq!(result.generated, 1);
        assert_eq!(list_staged_cards(&pool).await.expect("list").len(), 1);
        assert!(staging.join("apple.md").exists());
        server.join().expect("server");
        pool.close().await;
        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(staging.parent().unwrap());
    }

    #[tokio::test]
    async fn failed_generation_records_card_and_continues_batch() {
        let (pool, db_path) = test_pool().await;
        let staging = temp_path("stage").join(".flashcards_staging");
        db::update_provider(
            &pool,
            crate::domain::UpdateProviderInput {
                id: "builtin_openai".into(),
                name: "OpenAI".into(),
                base_url: "http://127.0.0.1:9".into(),
                credential_kind: CredentialKind::Bearer,
                selected_model: "gpt-5-mini".into(),
                system_prompt: String::new(),
                thinking_level: crate::domain::ThinkingLevel::None,
                web_enabled: false,
                vertex_project_id: String::new(),
                vertex_location: vertex_ai::DEFAULT_LOCATION.into(),
                vertex_client_email: String::new(),
            },
        )
        .await
        .expect("provider");
        let result = generate_flashcards(
            &pool,
            Client::new(),
            &staging,
            GenerateFlashcardsInput {
                entries_text: "apple\nbanana".into(),
                provider_id: "builtin_openai".into(),
                flashcard_prompt: "Make card".into(),
                concurrency_limit: 1,
                retry_count: 0,
            },
        )
        .await
        .expect("generate");
        assert_eq!(result.generated, 0);
        assert_eq!(result.failed, 2);
        assert_eq!(list_staged_cards(&pool).await.expect("list").len(), 2);
        pool.close().await;
        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(staging.parent().unwrap());
    }

    #[tokio::test]
    async fn cancellation_keeps_completed_cards_and_skips_remaining_entries() {
        let (pool, db_path) = test_pool().await;
        let staging = temp_path("stage").join(".flashcards_staging");
        let (base_url, server) = spawn_mock_openai_server(1, |_, _| MockHttpResponse {
            status: 200,
            body: openai_response_body("# apple"),
        });
        point_openai_provider_at(&pool, base_url).await;
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let result = generate_flashcards_with_progress(
            &pool,
            Client::new(),
            &staging,
            GenerateFlashcardsInput {
                entries_text: "apple\nbanana\ncherry".into(),
                provider_id: "builtin_openai".into(),
                flashcard_prompt: "Make card".into(),
                concurrency_limit: 1,
                retry_count: 0,
            },
            |progress| {
                if progress.completed == 1 {
                    let _ = cancel_tx.send(true);
                }
            },
            cancel_rx,
        )
        .await
        .expect("generate");
        let listed = list_staged_cards(&pool).await.expect("list");
        assert!(result.cancelled);
        assert_eq!(result.generated, 1);
        assert_eq!(result.failed, 0);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].source_entry, "apple");
        assert!(staging.join("apple.md").exists());
        assert!(!staging.join("banana.md").exists());
        assert!(!staging.join("cherry.md").exists());
        server.join().expect("server");
        pool.close().await;
        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(staging.parent().unwrap());
    }
    #[tokio::test]
    async fn concurrent_generation_respects_limit_and_preserves_input_order() {
        let (pool, db_path) = test_pool().await;
        let staging = temp_path("stage").join(".flashcards_staging");
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let active_for_server = Arc::clone(&active);
        let max_for_server = Arc::clone(&max_active);
        let (base_url, server) =
            spawn_flexible_mock_openai_server(24, StdDuration::from_millis(700), move |_, _| {
                let current = active_for_server.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                max_for_server.fetch_max(current, AtomicOrdering::SeqCst);
                std::thread::sleep(StdDuration::from_millis(80));
                active_for_server.fetch_sub(1, AtomicOrdering::SeqCst);
                MockHttpResponse {
                    status: 200,
                    body: openai_response_body("# card"),
                }
            });
        point_openai_provider_at(&pool, base_url).await;
        let entries = (0..12)
            .map(|index| format!("entry-{index:02}"))
            .collect::<Vec<_>>()
            .join("\n");
        let result = generate_flashcards(
            &pool,
            Client::new(),
            &staging,
            GenerateFlashcardsInput {
                entries_text: entries.clone(),
                provider_id: "builtin_openai".into(),
                flashcard_prompt: "Make card".into(),
                concurrency_limit: 10,
                retry_count: 1,
            },
        )
        .await
        .expect("generate");
        let listed = list_staged_cards(&pool).await.expect("list");
        assert_eq!(result.generated, 12, "{listed:?}");
        assert!(max_active.load(AtomicOrdering::SeqCst) <= 10);
        assert!(max_active.load(AtomicOrdering::SeqCst) > 1);
        let expected_entries = entries.lines().collect::<Vec<_>>();
        assert_eq!(
            listed
                .iter()
                .map(|card| card.source_entry.as_str())
                .collect::<Vec<_>>(),
            expected_entries
        );
        server.join().expect("server");
        pool.close().await;
        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(staging.parent().unwrap());
    }

    #[tokio::test]
    async fn retryable_http_429_can_succeed_on_retry() {
        let (pool, db_path) = test_pool().await;
        let staging = temp_path("stage").join(".flashcards_staging");
        let (base_url, server) = spawn_mock_openai_server(2, |request_index, _| {
            if request_index == 1 {
                MockHttpResponse {
                    status: 429,
                    body: r#"{"error":"slow down"}"#.into(),
                }
            } else {
                MockHttpResponse {
                    status: 200,
                    body: openai_response_body("# apple"),
                }
            }
        });
        point_openai_provider_at(&pool, base_url).await;
        let result = generate_flashcards(
            &pool,
            Client::new(),
            &staging,
            GenerateFlashcardsInput {
                entries_text: "apple".into(),
                provider_id: "builtin_openai".into(),
                flashcard_prompt: "Make card".into(),
                concurrency_limit: 1,
                retry_count: 1,
            },
        )
        .await
        .expect("generate");
        assert_eq!(result.generated, 1);
        assert_eq!(result.failed, 0);
        server.join().expect("server");
        pool.close().await;
        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(staging.parent().unwrap());
    }

    #[tokio::test]
    async fn non_retryable_http_400_fails_without_retry_delay() {
        let (pool, db_path) = test_pool().await;
        let staging = temp_path("stage").join(".flashcards_staging");
        let (base_url, server) = spawn_mock_openai_server(1, |_, _| MockHttpResponse {
            status: 400,
            body: r#"{"error":"bad request"}"#.into(),
        });
        point_openai_provider_at(&pool, base_url).await;
        let start = Instant::now();
        let result = generate_flashcards(
            &pool,
            Client::new(),
            &staging,
            GenerateFlashcardsInput {
                entries_text: "apple".into(),
                provider_id: "builtin_openai".into(),
                flashcard_prompt: "Make card".into(),
                concurrency_limit: 1,
                retry_count: 5,
            },
        )
        .await
        .expect("generate");
        assert_eq!(result.generated, 0);
        assert_eq!(result.failed, 1);
        assert!(start.elapsed() < StdDuration::from_millis(450));
        server.join().expect("server");
        pool.close().await;
        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(staging.parent().unwrap());
    }

    #[tokio::test]
    async fn old_config_import_gets_concurrency_and_retry_defaults() {
        let (pool, db_path) = test_pool().await;
        import_config(
            &pool,
            r#"{
                "settings": {
                    "flashcardPrompt": "Prompt",
                    "outputDirectory": "",
                    "selectedProviderId": "builtin_openai"
                },
                "providers": []
            }"#
            .into(),
        )
        .await
        .expect("import");
        let settings = get_settings(&pool).await.expect("settings");
        assert_eq!(settings.concurrency_limit, DEFAULT_CONCURRENCY_LIMIT);
        assert_eq!(settings.retry_count, DEFAULT_RETRY_COUNT);
        pool.close().await;
        let _ = fs::remove_file(db_path);
    }

    #[tokio::test]
    async fn write_staged_cards_never_overwrites_existing_files() {
        let (pool, db_path) = test_pool().await;
        let out = temp_path("out");
        fs::create_dir_all(&out).expect("out");
        fs::write(out.join("apple.md"), "existing").expect("existing");
        update_settings(
            &pool,
            UpdateFlashcardSettingsInput {
                flashcard_prompt: "prompt".into(),
                output_directory: out.to_string_lossy().to_string(),
                selected_provider_id: "builtin_openai".into(),
                concurrency_limit: 10,
                retry_count: 5,
            },
        )
        .await
        .expect("settings");
        let staging = temp_path("stage").join(".flashcards_staging");
        fs::create_dir_all(&staging).expect("stage");
        let staged_path = staging.join("apple.md");
        fs::write(&staged_path, "# apple").expect("staged");
        insert_staged_card(
            &pool,
            "card_1",
            0,
            "apple",
            "apple.md",
            &staged_path,
            "# apple",
            StagedCardStatus::Ready,
            &[],
            None,
        )
        .await
        .expect("insert");
        let result = write_staged_cards(&pool).await.expect("write");
        assert_eq!(result.written, 1);
        assert_eq!(
            fs::read_to_string(out.join("apple.md")).unwrap(),
            "existing"
        );
        assert_eq!(
            fs::read_to_string(out.join("apple-2.md")).unwrap(),
            "# apple"
        );
        pool.close().await;
        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(out);
        let _ = fs::remove_dir_all(staging.parent().unwrap());
    }

    #[tokio::test]
    async fn delete_staged_card_removes_file_and_row() {
        let (pool, db_path) = test_pool().await;
        let staging = temp_path("stage").join(".flashcards_staging");
        fs::create_dir_all(&staging).expect("stage");
        let staged_path = staging.join("apple.md");
        fs::write(&staged_path, "# apple").expect("staged");
        insert_staged_card(
            &pool,
            "card_delete",
            0,
            "apple",
            "apple.md",
            &staged_path,
            "# apple",
            StagedCardStatus::Ready,
            &[],
            None,
        )
        .await
        .expect("insert");
        delete_staged_card(&pool, "card_delete")
            .await
            .expect("delete");
        assert!(!staged_path.exists());
        assert!(list_staged_cards(&pool).await.expect("list").is_empty());
        pool.close().await;
        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(staging.parent().unwrap());
    }

    #[tokio::test]
    async fn delete_all_staged_cards_removes_files_and_rows() {
        let (pool, db_path) = test_pool().await;
        let staging = temp_path("stage").join(".flashcards_staging");
        fs::create_dir_all(&staging).expect("stage");
        for (id, entry) in [("card_a", "apple"), ("card_b", "banana")] {
            let filename = format!("{entry}.md");
            let staged_path = staging.join(&filename);
            fs::write(&staged_path, format!("# {entry}")).expect("staged");
            insert_staged_card(
                &pool,
                id,
                0,
                entry,
                &filename,
                &staged_path,
                &format!("# {entry}"),
                StagedCardStatus::Ready,
                &[],
                None,
            )
            .await
            .expect("insert");
        }
        let deleted = delete_all_staged_cards(&pool).await.expect("delete all");
        assert_eq!(deleted, 2);
        assert!(!staging.join("apple.md").exists());
        assert!(!staging.join("banana.md").exists());
        assert!(list_staged_cards(&pool).await.expect("list").is_empty());
        pool.close().await;
        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(staging.parent().unwrap());
    }
}
