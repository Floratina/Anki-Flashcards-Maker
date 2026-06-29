import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  Bot,
  CheckCircle2,
  CloudDownload,
  Database,
  Download,
  Edit3,
  Eye,
  FileText,
  FolderOpen,
  KeyRound,
  LoaderCircle,
  Moon,
  Plus,
  Save,
  Send,
  Settings,
  Sun,
  Trash2,
  Upload,
  XCircle,
} from "lucide-react";

type PageId = "generate" | "drafts" | "providers" | "config";
type ProviderTab = "presets" | "custom";
type ProviderProtocol =
  | "openai-responses"
  | "gemini"
  | "deepseek"
  | "agent-platform"
  | "openai-compatible";
type CredentialKind =
  | "bearer"
  | "gemini-api-key"
  | "gemini-auth-api-key"
  | "service-account"
  | "none";
type ThinkingLevel = "none" | "low" | "medium" | "high" | "max";
type StagedCardStatus = "ready" | "failed" | "written";
type ThemeMode = "light" | "dark";
type ToastVariant = "success" | "error" | "info" | "warning";

interface ToastMessage {
  id: number;
  message: string;
  variant: ToastVariant;
}

interface VertexConfig {
  projectId: string;
  location: string;
  clientEmail: string;
}

interface ModelCapabilities {
  thinkingOptions: ThinkingLevel[];
  webSupported: boolean;
}

interface ProviderView {
  id: string;
  name: string;
  protocol: ProviderProtocol;
  baseUrl: string;
  credentialKind: CredentialKind;
  credentialMask: string | null;
  selectedModel: string;
  systemPrompt: string;
  thinkingLevel: ThinkingLevel;
  webEnabled: boolean;
  isBuiltin: boolean;
  vertex: VertexConfig;
  capabilities: ModelCapabilities;
  updatedAt: string;
}

interface ModelOption {
  id: string;
  label: string;
}

interface ConnectivityResult {
  success: boolean;
  latencyMs: number;
  responseText: string;
  error: string | null;
}

interface FlashcardSettings {
  flashcardPrompt: string;
  outputDirectory: string;
  selectedProviderId: string;
  concurrencyLimit: number;
  retryCount: number;
}

interface StagedCardView {
  id: string;
  sourceEntry: string;
  filename: string;
  stagedPath: string;
  status: StagedCardStatus;
  warnings: string[];
  error: string | null;
  updatedAt: string;
}

interface StagedCardContent {
  card: StagedCardView;
  content: string;
}

interface GenerateFlashcardsResult {
  cards: StagedCardView[];
  generated: number;
  failed: number;
  cancelled: boolean;
}

interface GenerateFlashcardsProgress {
  total: number;
  completed: number;
  inProgress: number;
  generated: number;
  failed: number;
  currentEntry: string | null;
  cancelled: boolean;
}

interface WriteStagedCardsResult {
  written: number;
  files: string[];
}

const PROTOCOL_LABELS: Record<ProviderProtocol, string> = {
  "openai-responses": "OpenAI Responses",
  gemini: "Gemini",
  deepseek: "DeepSeek",
  "agent-platform": "Agent Platform",
  "openai-compatible": "OpenAI Compatible",
};

const CREDENTIAL_LABELS: Record<CredentialKind, string> = {
  bearer: "Bearer API Key",
  "gemini-api-key": "Gemini API Key",
  "gemini-auth-api-key": "Gemini Auth API Key",
  "service-account": "Service Account",
  none: "None",
};

const GENERATION_PROGRESS_EVENT = "flashcards-generation-progress";
const DEFAULT_CONCURRENCY_LIMIT = 10;
const MIN_CONCURRENCY_LIMIT = 1;
const MAX_CONCURRENCY_LIMIT = 50;
const DEFAULT_RETRY_COUNT = 5;
const MIN_RETRY_COUNT = 0;
const MAX_RETRY_COUNT = 10;

const THINKING_LABELS: Record<ThinkingLevel, string> = {
  none: "None",
  low: "Low",
  medium: "Medium",
  high: "High",
  max: "Max",
};

const CUSTOM_PROTOCOLS: ProviderProtocol[] = [
  "openai-responses",
  "gemini",
  "deepseek",
  "agent-platform",
  "openai-compatible",
];

const VERTEX_LOCATIONS = [
  "global",
  "us-central1",
  "us-east1",
  "us-east4",
  "us-west1",
  "us-west4",
  "europe-west1",
  "europe-west2",
  "europe-west4",
  "asia-east1",
  "asia-northeast1",
  "asia-southeast1",
];

const emptySettings: FlashcardSettings = {
  flashcardPrompt: "",
  outputDirectory: "",
  selectedProviderId: "",
  concurrencyLimit: DEFAULT_CONCURRENCY_LIMIT,
  retryCount: DEFAULT_RETRY_COUNT,
};

function getErrorMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

function isStagedCardNotFound(message: string): boolean {
  return message.toLowerCase().includes("staged card not found");
}

function countEntries(value: string): number {
  return value.split(/\r?\n/).filter((line) => line.trim().length > 0).length;
}

function clampNumber(value: number, min: number, max: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.round(value)));
}

function providerIcon(protocol: ProviderProtocol) {
  if (protocol === "agent-platform") return <Database className="size-4" />;
  if (protocol === "deepseek") return <Activity className="size-4" />;
  return <Bot className="size-4" />;
}

function statusLabel(status: StagedCardStatus): string {
  if (status === "failed") return "失败";
  if (status === "written") return "已写入";
  return "待写入";
}

function App() {
  const [page, setPage] = useState<PageId>("generate");
  const [providerTab, setProviderTab] = useState<ProviderTab>("presets");
  const [providers, setProviders] = useState<ProviderView[]>([]);
  const [settings, setSettings] = useState<FlashcardSettings>(emptySettings);
  const [cards, setCards] = useState<StagedCardView[]>([]);
  const [entriesText, setEntriesText] = useState("");
  const [promptDraft, setPromptDraft] = useState("");
  const [selectedCard, setSelectedCard] = useState<StagedCardContent | null>(null);
  const [draftMode, setDraftMode] = useState<"edit" | "preview">("edit");
  const [providerDraft, setProviderDraft] = useState<ProviderView | null>(null);
  const [modelsByProvider, setModelsByProvider] = useState<Record<string, ModelOption[]>>({});
  const [credentialValue, setCredentialValue] = useState("");
  const [privateKeyValue, setPrivateKeyValue] = useState("");
  const [serviceAccountJson, setServiceAccountJson] = useState("");
  const [newProviderName, setNewProviderName] = useState("");
  const [newProviderProtocol, setNewProviderProtocol] =
    useState<ProviderProtocol>("openai-compatible");
  const [exportedConfig, setExportedConfig] = useState("");
  const [importConfigText, setImportConfigText] = useState("");
  const [connectivity, setConnectivity] = useState<ConnectivityResult | null>(null);
  const [generationProgress, setGenerationProgress] =
    useState<GenerateFlashcardsProgress | null>(null);
  const [generationStopRequested, setGenerationStopRequested] = useState(false);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [theme, setTheme] = useState<ThemeMode>(() =>
    localStorage.getItem("flashcards-maker-theme") === "dark" ? "dark" : "light",
  );
  const nextToastId = useRef(0);
  const toastTimers = useRef<Map<number, number>>(new Map());
  const stagedRefreshTimer = useRef<number | null>(null);

  const dismissToast = useCallback((id: number): void => {
    const timer = toastTimers.current.get(id);
    if (timer !== undefined) {
      window.clearTimeout(timer);
      toastTimers.current.delete(id);
    }
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const pushToast = useCallback(
    (message: string, variant: ToastVariant = "info"): void => {
      const clean = message.trim();
      if (!clean) return;
      const id = ++nextToastId.current;
      setToasts((current) => [...current, { id, message: clean, variant }].slice(-6));
      const timer = window.setTimeout(() => dismissToast(id), 6200);
      toastTimers.current.set(id, timer);
    },
    [dismissToast],
  );

  const refreshStagedCards = useCallback(async (): Promise<void> => {
    try {
      const nextCards = await invoke<StagedCardView[]>("list_staged_cards");
      setCards(nextCards);
    } catch (cause) {
      setError(getErrorMessage(cause));
    }
  }, []);

  const queueStagedCardsRefresh = useCallback(
    (delay = 120): void => {
      if (stagedRefreshTimer.current !== null) return;
      stagedRefreshTimer.current = window.setTimeout(() => {
        stagedRefreshTimer.current = null;
        void refreshStagedCards();
      }, delay);
    },
    [refreshStagedCards],
  );

  useEffect(() => {
    void refreshAll();
  }, []);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;

    void listen<GenerateFlashcardsProgress>(GENERATION_PROGRESS_EVENT, (event) => {
      const progress = event.payload;
      setGenerationProgress(progress);
      if (progress.completed > 0 || progress.cancelled) {
        queueStagedCardsRefresh();
      }
    })
      .then((cleanup) => {
        if (disposed) {
          cleanup();
        } else {
          unlisten = cleanup;
        }
      })
      .catch((cause) => {
        setError(getErrorMessage(cause));
      });

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [queueStagedCardsRefresh]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("flashcards-maker-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!notice.trim()) return;
    pushToast(notice, "success");
    setNotice("");
  }, [notice, pushToast]);

  useEffect(() => {
    if (!error.trim()) return;
    pushToast(error, "error");
    setError("");
  }, [error, pushToast]);

  useEffect(
    () => () => {
      toastTimers.current.forEach((timer) => window.clearTimeout(timer));
      toastTimers.current.clear();
    },
    [],
  );

  useEffect(
    () => () => {
      if (stagedRefreshTimer.current !== null) {
        window.clearTimeout(stagedRefreshTimer.current);
        stagedRefreshTimer.current = null;
      }
    },
    [],
  );

  useEffect(() => {
    setPromptDraft(settings.flashcardPrompt);
  }, [settings.flashcardPrompt]);

  useEffect(() => {
    setSelectedCard((current) => {
      if (!current) return null;
      const matchingCard = cards.find((card) => card.id === current.card.id);
      return matchingCard ? { ...current, card: matchingCard } : null;
    });
  }, [cards]);

  useEffect(() => {
    const current =
      providers.find((provider) => provider.id === settings.selectedProviderId) ??
      providers[0] ??
      null;
    setProviderDraft(current ? cloneProvider(current) : null);
    if (current) {
      setProviderTab(current.isBuiltin ? "presets" : "custom");
    }
  }, [providers, settings.selectedProviderId]);

  const selectedProvider = useMemo(
    () =>
      providers.find((provider) => provider.id === settings.selectedProviderId) ??
      providers[0] ??
      null,
    [providers, settings.selectedProviderId],
  );

  const providerModels = useMemo(() => {
    if (!providerDraft) return [];
    const remote = modelsByProvider[providerDraft.id] ?? [];
    const manualModel = providerDraft.selectedModel.trim();
    if (manualModel && !remote.some((model) => model.id === manualModel)) {
      return [{ id: manualModel, label: manualModel }, ...remote];
    }
    return remote;
  }, [modelsByProvider, providerDraft]);

  async function refreshAll(): Promise<void> {
    setBusy("loading");
    try {
      const [nextProviders, nextSettings, nextCards] = await Promise.all([
        invoke<ProviderView[]>("list_providers"),
        invoke<FlashcardSettings>("get_flashcard_settings"),
        invoke<StagedCardView[]>("list_staged_cards"),
      ]);
      setProviders(nextProviders);
      setSettings(nextSettings);
      setCards(nextCards);
      setError("");
    } catch (cause) {
      setGenerationProgress(null);
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function saveSettings(
    next: Partial<FlashcardSettings> = {},
  ): Promise<FlashcardSettings | null> {
    const merged = { ...settings, ...next };
    try {
      const saved = await invoke<FlashcardSettings>("update_flashcard_settings", {
        input: {
          flashcardPrompt: merged.flashcardPrompt,
          outputDirectory: merged.outputDirectory,
          selectedProviderId: merged.selectedProviderId,
          concurrencyLimit: clampNumber(
            merged.concurrencyLimit,
            MIN_CONCURRENCY_LIMIT,
            MAX_CONCURRENCY_LIMIT,
            DEFAULT_CONCURRENCY_LIMIT,
          ),
          retryCount: clampNumber(
            merged.retryCount,
            MIN_RETRY_COUNT,
            MAX_RETRY_COUNT,
            DEFAULT_RETRY_COUNT,
          ),
        },
      });
      setSettings(saved);
      setNotice("配置已保存");
      setError("");
      return saved;
    } catch (cause) {
      setError(getErrorMessage(cause));
      return null;
    }
  }

  async function selectProvider(providerId: string): Promise<void> {
    setSettings((current) => ({ ...current, selectedProviderId: providerId }));
    setConnectivity(null);
    await saveSettings({ selectedProviderId: providerId });
  }

  async function switchProviderTab(nextTab: ProviderTab): Promise<void> {
    setProviderTab(nextTab);
    const first = providers.find((provider) =>
      nextTab === "presets" ? provider.isBuiltin : !provider.isBuiltin,
    );
    if (first) {
      await selectProvider(first.id);
    } else {
      setProviderDraft(null);
    }
  }

  async function generateCards(): Promise<void> {
    const providerId = selectedProvider?.id ?? "";
    if (!providerId) {
      setError("请先配置提供商");
      return;
    }
    const total = countEntries(entriesText);
    setGenerationProgress({
      total,
      completed: 0,
      inProgress: 0,
      generated: 0,
      failed: 0,
      currentEntry: null,
      cancelled: false,
    });
    setGenerationStopRequested(false);
    setSelectedCard(null);
    setCards([]);
    setBusy("generate");
    setError("");
    try {
      const result = await invoke<GenerateFlashcardsResult>("generate_flashcards", {
        input: {
          entriesText,
          providerId,
          flashcardPrompt: promptDraft,
          concurrencyLimit: settings.concurrencyLimit,
          retryCount: settings.retryCount,
        },
      });
      setCards(result.cards);
      setSettings((current) => ({
        ...current,
        flashcardPrompt: promptDraft,
        selectedProviderId: providerId,
        concurrencyLimit: settings.concurrencyLimit,
        retryCount: settings.retryCount,
      }));
      setNotice(
        result.cancelled
          ? `已停止生成：保留 ${result.generated} 个成功草稿，${result.failed} 个失败草稿`
          : `生成完成：${result.generated} 成功，${result.failed} 失败`,
      );
      setPage("drafts");
      if (result.cards.length > 0) {
        void openCard(result.cards[0].id);
      }
    } catch (cause) {
      setGenerationProgress(null);
      setError(getErrorMessage(cause));
      void refreshStagedCards();
    } finally {
      setGenerationStopRequested(false);
      setBusy("");
    }
  }

  async function stopGeneration(): Promise<void> {
    if (busy !== "generate" || generationStopRequested) return;
    setGenerationStopRequested(true);
    try {
      await invoke("cancel_flashcard_generation");
      setNotice("正在停止生成，已完成的草稿会保留");
      setError("");
    } catch (cause) {
      setGenerationStopRequested(false);
      setError(getErrorMessage(cause));
    }
  }

  async function openCard(id: string): Promise<void> {
    setBusy("read-card");
    try {
      const content = await invoke<StagedCardContent>("read_staged_card", { id });
      setSelectedCard(content);
      setDraftMode("edit");
      setError("");
    } catch (cause) {
      const message = getErrorMessage(cause);
      if (isStagedCardNotFound(message)) {
        setSelectedCard(null);
        try {
          const nextCards = await invoke<StagedCardView[]>("list_staged_cards");
          setCards(nextCards);
          setNotice("草稿已不存在，已刷新列表");
          setError("");
        } catch (refreshCause) {
          setError(getErrorMessage(refreshCause));
        }
      } else {
        setError(message);
      }
    } finally {
      setBusy("");
    }
  }

  async function saveCard(): Promise<void> {
    if (!selectedCard) return;
    setBusy("save-card");
    try {
      const card = await invoke<StagedCardView>("save_staged_card", {
        input: { id: selectedCard.card.id, content: selectedCard.content },
      });
      setSelectedCard({ ...selectedCard, card });
      setCards((current) => current.map((item) => (item.id === card.id ? card : item)));
      setNotice("草稿已保存");
      setError("");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function deleteCard(id: string): Promise<void> {
    setBusy("delete-card");
    try {
      await invoke("delete_staged_card", { id });
      setCards((current) => current.filter((card) => card.id !== id));
      if (selectedCard?.card.id === id) {
        setSelectedCard(null);
      }
      setNotice("草稿已删除");
      setError("");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function deleteAllCards(): Promise<void> {
    if (cards.length === 0) return;
    setBusy("delete-all-cards");
    try {
      const deleted = await invoke<number>("delete_all_staged_cards");
      setCards([]);
      setSelectedCard(null);
      setNotice(`已删除 ${deleted} 个草稿`);
      setError("");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function writeCards(): Promise<void> {
    setBusy("write");
    try {
      await saveSettings();
      const result = await invoke<WriteStagedCardsResult>("write_staged_cards");
      const nextCards = await invoke<StagedCardView[]>("list_staged_cards");
      setCards(nextCards);
      setNotice(`已写入 ${result.written} 个 Markdown 文件`);
      setError("");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function pickOutputDirectory(): Promise<void> {
    setBusy("pick-dir");
    try {
      const directory = await invoke<string | null>("pick_output_directory");
      if (directory) {
        await saveSettings({ outputDirectory: directory });
      }
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function exportConfig(): Promise<void> {
    setBusy("export");
    try {
      const json = await invoke<string>("export_config");
      setExportedConfig(json);
      setNotice("配置 JSON 已生成，默认不包含密钥");
      setError("");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function importConfig(): Promise<void> {
    setBusy("import");
    try {
      await invoke("import_config", { configJson: importConfigText });
      await refreshAll();
      setNotice("配置已导入，已有密钥已保留");
      setError("");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function saveProvider(): Promise<ProviderView | null> {
    if (!providerDraft) return null;
    setBusy("provider-save");
    try {
      const clean = sanitizeProvider(providerDraft);
      const updated = await invoke<ProviderView>("update_provider", {
        input: {
          id: clean.id,
          name: clean.name,
          baseUrl: clean.baseUrl,
          credentialKind: clean.credentialKind,
          selectedModel: clean.selectedModel,
          systemPrompt: "",
          thinkingLevel: clean.thinkingLevel,
          webEnabled: clean.webEnabled,
          vertexProjectId: clean.vertex.projectId,
          vertexLocation: clean.vertex.location,
          vertexClientEmail: clean.vertex.clientEmail,
        },
      });
      setProviders((current) =>
        current.map((provider) => (provider.id === updated.id ? updated : provider)),
      );
      setProviderDraft(cloneProvider(updated));
      setNotice("提供商配置已保存");
      setError("");
      return updated;
    } catch (cause) {
      setError(getErrorMessage(cause));
      return null;
    } finally {
      setBusy("");
    }
  }

  async function saveProviderCredential(value: string | null): Promise<void> {
    if (!providerDraft) return;
    setBusy("credential");
    try {
      const updated = await invoke<ProviderView>("save_provider_credential", {
        input: { providerId: providerDraft.id, credential: value },
      });
      setProviders((current) =>
        current.map((provider) => (provider.id === updated.id ? updated : provider)),
      );
      setProviderDraft(cloneProvider(updated));
      setCredentialValue("");
      setPrivateKeyValue("");
      setNotice(value ? "密钥已保存" : "密钥已清除");
      setError("");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function importServiceAccount(): Promise<void> {
    if (!providerDraft) return;
    setBusy("service-account");
    try {
      const updated = await invoke<ProviderView>("import_agent_platform_service_account", {
        input: {
          providerId: providerDraft.id,
          serviceAccountJson,
          location: providerDraft.vertex.location,
        },
      });
      setProviders((current) =>
        current.map((provider) => (provider.id === updated.id ? updated : provider)),
      );
      setProviderDraft(cloneProvider(updated));
      setServiceAccountJson("");
      setNotice("Service Account JSON 已解析并保存");
      setError("");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function fetchModels(): Promise<void> {
    const saved = await saveProvider();
    if (!saved) return;
    setBusy("models");
    try {
      const models = await invoke<ModelOption[]>("fetch_provider_models", {
        providerId: saved.id,
      });
      setModelsByProvider((current) => ({ ...current, [saved.id]: models }));
      setNotice(`已获取 ${models.length} 个模型`);
      setError("");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function testModel(): Promise<void> {
    const saved = await saveProvider();
    if (!saved) return;
    setBusy("test-model");
    try {
      const result = await invoke<ConnectivityResult>("test_model_connectivity", {
        providerId: saved.id,
      });
      setConnectivity(result);
      setNotice(result.success ? "连通性测试成功" : "连通性测试失败");
      setError(result.success ? "" : result.error ?? "连通性测试失败");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function createProvider(): Promise<void> {
    setBusy("new-provider");
    try {
      const created = await invoke<ProviderView>("create_provider", {
        input: { name: newProviderName, protocol: newProviderProtocol },
      });
      setProviders((current) => [...current, created]);
      setProviderTab("custom");
      setSettings((current) => ({ ...current, selectedProviderId: created.id }));
      await saveSettings({ selectedProviderId: created.id });
      setNewProviderName("");
      setNewProviderProtocol("openai-compatible");
      setNotice("提供商已创建");
      setError("");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  async function deleteProvider(id: string): Promise<void> {
    const provider = providers.find((item) => item.id === id);
    if (!provider || provider.isBuiltin) return;
    setBusy("provider-delete");
    try {
      await invoke("delete_provider", { id });
      const nextProviders = await invoke<ProviderView[]>("list_providers");
      const nextSelected =
        nextProviders.find((item) => item.id === settings.selectedProviderId)?.id ??
        nextProviders[0]?.id ??
        "";
      setProviders(nextProviders);
      setProviderDraft(nextProviders.find((item) => item.id === nextSelected) ?? null);
      setSettings((current) => ({ ...current, selectedProviderId: nextSelected }));
      await saveSettings({ selectedProviderId: nextSelected });
      setCredentialValue("");
      setPrivateKeyValue("");
      setServiceAccountJson("");
      setConnectivity(null);
      setNotice("自定义提供商已删除");
      setError("");
    } catch (cause) {
      setError(getErrorMessage(cause));
    } finally {
      setBusy("");
    }
  }

  const navItems: Array<{ id: PageId; label: string; icon: ReactNode }> = [
    { id: "generate", label: "生成词卡", icon: <Send className="size-4" /> },
    { id: "drafts", label: "草稿列表", icon: <FileText className="size-4" /> },
    { id: "providers", label: "提供商设置", icon: <Bot className="size-4" /> },
    { id: "config", label: "配置", icon: <Settings className="size-4" /> },
  ];

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="title-block">
          <h1>Flashcards Maker</h1>
          <p>一行输入生成一个暂存 Markdown 词卡，检查后再写入目标路径。</p>
        </div>
        <nav className="main-nav" aria-label="Main">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={page === item.id ? "nav-button active" : "nav-button"}
              onClick={() => setPage(item.id)}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>
        <button
          className="icon-button theme-toggle"
          title={theme === "dark" ? "切换浅色模式" : "切换深色模式"}
          onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
        >
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </button>
      </header>

      <section className="content-scroll">
        {page === "generate" && (
          <section className="page-grid two">
            <section className="panel">
              <div className="section-heading">
                <Send className="size-4" />
                <span>生成词卡</span>
              </div>
              <label className="field">
                <span>批量条目</span>
                <textarea
                  className="entries-textarea"
                  value={entriesText}
                  placeholder={
                    "每行一个单词、短语或主题，例如：\napple\nflabbergasted\nmake ends meet"
                  }
                  onChange={(event) => setEntriesText(event.target.value)}
                />
              </label>
              <div className="button-row">
                <button
                  className="primary-button"
                  disabled={busy === "generate" || !entriesText.trim()}
                  onClick={() => void generateCards()}
                >
                  {busy === "generate" ? (
                    <LoaderCircle className="size-4 spin" />
                  ) : (
                    <Send className="size-4" />
                  )}
                  生成到暂存区
                </button>
                {busy === "generate" && (
                  <button
                    className="secondary-button danger"
                    disabled={generationStopRequested}
                    onClick={() => void stopGeneration()}
                  >
                    {generationStopRequested ? (
                      <LoaderCircle className="size-4 spin" />
                    ) : (
                      <XCircle className="size-4" />
                    )}
                    {generationStopRequested ? "停止中" : "停止"}
                  </button>
                )}
                <span className="muted-text">{cards.length} 个草稿</span>
              </div>
              {generationProgress && (
                <GenerationProgressBar
                  progress={generationProgress}
                  active={busy === "generate"}
                />
              )}
            </section>

            <section className="panel">
              <div className="section-heading">
                <Bot className="size-4" />
                <span>生成配置</span>
              </div>
              <label className="field">
                <span>使用提供商</span>
                <select
                  value={settings.selectedProviderId}
                  onChange={(event) => void selectProvider(event.target.value)}
                >
                  {providers.map((provider) => (
                    <option key={provider.id} value={provider.id}>
                      {provider.name} · {provider.selectedModel || "未选择模型"}
                    </option>
                  ))}
                </select>
              </label>
              <ProviderSummary provider={selectedProvider} />
              <div className="form-grid compact">
                <label className="field">
                  <span>并发数</span>
                  <input
                    type="number"
                    min={MIN_CONCURRENCY_LIMIT}
                    max={MAX_CONCURRENCY_LIMIT}
                    step={1}
                    value={settings.concurrencyLimit}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        concurrencyLimit: clampNumber(
                          Number(event.target.value),
                          MIN_CONCURRENCY_LIMIT,
                          MAX_CONCURRENCY_LIMIT,
                          DEFAULT_CONCURRENCY_LIMIT,
                        ),
                      }))
                    }
                  />
                </label>
                <label className="field">
                  <span>重试次数</span>
                  <input
                    type="number"
                    min={MIN_RETRY_COUNT}
                    max={MAX_RETRY_COUNT}
                    step={1}
                    value={settings.retryCount}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        retryCount: clampNumber(
                          Number(event.target.value),
                          MIN_RETRY_COUNT,
                          MAX_RETRY_COUNT,
                          DEFAULT_RETRY_COUNT,
                        ),
                      }))
                    }
                  />
                </label>
              </div>
              <label className="field">
                <span>词卡提示词</span>
                <textarea
                  value={promptDraft}
                  onChange={(event) => setPromptDraft(event.target.value)}
                />
              </label>
              <button
                className="secondary-button"
                onClick={() => void saveSettings({ flashcardPrompt: promptDraft })}
              >
                <Save className="size-4" />
                保存提示词
              </button>
            </section>
          </section>
        )}

        {page === "drafts" && (
          <section className="page-grid drafts-grid">
            <section className="panel draft-list-panel">
              <div className="section-heading">
                <FileText className="size-4" />
                <span>暂存 Markdown</span>
              </div>
              <div className="draft-list">
                {cards.length === 0 && <div className="empty-state">还没有暂存词卡</div>}
                {cards.map((card) => (
                  <div
                    key={card.id}
                    className={
                      selectedCard?.card.id === card.id ? "draft-item active" : "draft-item"
                    }
                  >
                    <button className="draft-open-button" onClick={() => void openCard(card.id)}>
                      <strong>{card.filename}</strong>
                      <span>{card.sourceEntry}</span>
                      <small className={`status ${card.status}`}>{statusLabel(card.status)}</small>
                    </button>
                    <button
                      className="icon-button danger"
                      title="删除草稿"
                      onClick={() => void deleteCard(card.id)}
                    >
                      <Trash2 className="size-4" />
                    </button>
                  </div>
                ))}
              </div>
              <div className="output-box">
                <span>输出路径</span>
                <strong>{settings.outputDirectory || "尚未选择"}</strong>
                <button
                  className="secondary-button danger"
                  disabled={busy === "delete-all-cards" || cards.length === 0}
                  onClick={() => void deleteAllCards()}
                >
                  {busy === "delete-all-cards" ? (
                    <LoaderCircle className="size-4 spin" />
                  ) : (
                    <Trash2 className="size-4" />
                  )}
                  清空草稿
                </button>
                <button
                  className="primary-button"
                  disabled={busy === "write" || cards.every((card) => card.status !== "ready")}
                  onClick={() => void writeCards()}
                >
                  {busy === "write" ? (
                    <LoaderCircle className="size-4 spin" />
                  ) : (
                    <FolderOpen className="size-4" />
                  )}
                  写入路径
                </button>
              </div>
            </section>

            <section className="panel editor-panel">
              {!selectedCard ? (
                <div className="empty-state">选择一个草稿后可以编辑或预览</div>
              ) : (
                <>
                  <div className="editor-toolbar">
                    <div>
                      <strong>{selectedCard.card.filename}</strong>
                      <small>{selectedCard.card.stagedPath}</small>
                    </div>
                    <div className="button-row">
                      <button
                        className={draftMode === "edit" ? "secondary-button active" : "secondary-button"}
                        onClick={() => setDraftMode("edit")}
                      >
                        <Edit3 className="size-4" />
                        编辑
                      </button>
                      <button
                        className={
                          draftMode === "preview" ? "secondary-button active" : "secondary-button"
                        }
                        onClick={() => setDraftMode("preview")}
                      >
                        <Eye className="size-4" />
                        预览
                      </button>
                      <button className="primary-button" onClick={() => void saveCard()}>
                        <Save className="size-4" />
                        保存
                      </button>
                      <button
                        className="secondary-button danger"
                        onClick={() => void deleteCard(selectedCard.card.id)}
                      >
                        <Trash2 className="size-4" />
                        删除
                      </button>
                    </div>
                  </div>
                  {selectedCard.card.error && (
                    <div className="warning-list error">{selectedCard.card.error}</div>
                  )}
                  {selectedCard.card.warnings.length > 0 && (
                    <div className="warning-list">
                      {selectedCard.card.warnings.map((warning) => (
                        <span key={warning}>{warning}</span>
                      ))}
                    </div>
                  )}
                  {draftMode === "edit" ? (
                    <textarea
                      className="markdown-editor"
                      value={selectedCard.content}
                      onChange={(event) =>
                        setSelectedCard({ ...selectedCard, content: event.target.value })
                      }
                    />
                  ) : (
                    <article className="markdown-preview">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {selectedCard.content}
                      </ReactMarkdown>
                    </article>
                  )}
                </>
              )}
            </section>
          </section>
        )}

        {page === "providers" && (
          <ProviderPage
            tab={providerTab}
            providers={providers}
            settings={settings}
            draft={providerDraft}
            providerModels={providerModels}
            credentialValue={credentialValue}
            privateKeyValue={privateKeyValue}
            serviceAccountJson={serviceAccountJson}
            connectivity={connectivity}
            busy={busy}
            newProviderName={newProviderName}
            newProviderProtocol={newProviderProtocol}
            onTabChange={(tab) => void switchProviderTab(tab)}
            onSelect={(selectedProviderId) => void selectProvider(selectedProviderId)}
            onDraftChange={(draft) => setProviderDraft(sanitizeProvider(draft))}
            onCredentialChange={setCredentialValue}
            onPrivateKeyChange={setPrivateKeyValue}
            onServiceAccountJsonChange={setServiceAccountJson}
            onSaveProvider={() => void saveProvider()}
            onSaveCredential={() => void saveProviderCredential(credentialValue)}
            onClearCredential={() => void saveProviderCredential(null)}
            onSavePrivateKey={() => void saveProviderCredential(privateKeyValue)}
            onClearPrivateKey={() => void saveProviderCredential(null)}
            onImportServiceAccount={() => void importServiceAccount()}
            onFetchModels={() => void fetchModels()}
            onTestModel={() => void testModel()}
            onNewProviderNameChange={setNewProviderName}
            onNewProviderProtocolChange={setNewProviderProtocol}
            onCreateProvider={() => void createProvider()}
            onDeleteProvider={(id) => void deleteProvider(id)}
          />
        )}

        {page === "config" && (
          <section className="page-grid two">
            <section className="panel">
              <div className="section-heading">
                <FolderOpen className="size-4" />
                <span>输出路径</span>
              </div>
              <label className="field">
                <span>Markdown 写入目录</span>
                <input
                  value={settings.outputDirectory}
                  onChange={(event) =>
                    setSettings((current) => ({
                      ...current,
                      outputDirectory: event.target.value,
                    }))
                  }
                />
              </label>
              <div className="button-row">
                <button className="secondary-button" onClick={() => void pickOutputDirectory()}>
                  <FolderOpen className="size-4" />
                  选择路径
                </button>
                <button className="primary-button" onClick={() => void saveSettings()}>
                  <Save className="size-4" />
                  保存路径
                </button>
              </div>
            </section>

            <section className="panel">
              <div className="section-heading">
                <Settings className="size-4" />
                <span>JSON 导入导出</span>
              </div>
              <div className="button-row">
                <button className="secondary-button" onClick={() => void exportConfig()}>
                  <Download className="size-4" />
                  导出配置 JSON
                </button>
                <button
                  className="primary-button"
                  disabled={!importConfigText.trim()}
                  onClick={() => void importConfig()}
                >
                  <Upload className="size-4" />
                  导入配置 JSON
                </button>
              </div>
              <label className="field">
                <span>导入 JSON</span>
                <textarea
                  className="json-textarea"
                  value={importConfigText}
                  onChange={(event) => setImportConfigText(event.target.value)}
                />
              </label>
              <label className="field">
                <span>导出 JSON（不含密钥）</span>
                <textarea className="json-textarea" readOnly value={exportedConfig} />
              </label>
            </section>
          </section>
        )}
      </section>
      <ToastStack toasts={toasts} onDismiss={dismissToast} />
    </main>
  );
}

function GenerationProgressBar({
  progress,
  active,
}: {
  progress: GenerateFlashcardsProgress;
  active: boolean;
}) {
  const total = Math.max(progress.total, 1);
  const percent = Math.min(100, Math.round((progress.completed / total) * 100));
  const label = progress.cancelled ? "已停止" : active ? "生成中" : "生成进度";

  return (
    <div
      className="generation-progress"
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={progress.total}
      aria-valuenow={progress.completed}
    >
      <div className="progress-meta">
        <strong>{label}</strong>
        <span>
          {progress.completed}/{progress.total}
        </span>
      </div>
      <div className="progress-track">
        <span style={{ width: `${percent}%` }} />
      </div>
      <div className="progress-detail">
        <span>
          {progress.inProgress} 进行中 · {progress.generated} 成功 · {progress.failed} 失败
        </span>
        {active && !progress.cancelled && progress.currentEntry && <small>{progress.currentEntry}</small>}
      </div>
    </div>
  );
}

function ToastStack({
  toasts,
  onDismiss,
}: {
  toasts: ToastMessage[];
  onDismiss: (id: number) => void;
}) {
  if (toasts.length === 0) return null;
  return createPortal(
    <div className="toast-stack" aria-live="polite" aria-atomic="false">
      {toasts.map((toast) => {
        const Icon =
          toast.variant === "error"
            ? XCircle
            : toast.variant === "success"
              ? CheckCircle2
              : Activity;
        return (
          <button
            key={toast.id}
            className={`toast-bubble ${toast.variant}`}
            onClick={() => onDismiss(toast.id)}
            title="点击关闭"
          >
            <Icon className="size-4" />
            <span>{toast.message}</span>
          </button>
        );
      })}
    </div>,
    document.body,
  );
}

function ProviderSummary({ provider }: { provider: ProviderView | null }) {
  if (!provider) {
    return <div className="provider-summary">尚未配置提供商</div>;
  }
  return (
    <div className="provider-summary">
      <span className="provider-icon">{providerIcon(provider.protocol)}</span>
      <div>
        <strong>{provider.name}</strong>
        <small>
          {PROTOCOL_LABELS[provider.protocol]} · {provider.selectedModel || "未选择模型"}
        </small>
      </div>
    </div>
  );
}

function ProviderPage({
  tab,
  providers,
  settings,
  draft,
  providerModels,
  credentialValue,
  privateKeyValue,
  serviceAccountJson,
  connectivity,
  busy,
  newProviderName,
  newProviderProtocol,
  onTabChange,
  onSelect,
  onDraftChange,
  onCredentialChange,
  onPrivateKeyChange,
  onServiceAccountJsonChange,
  onSaveProvider,
  onSaveCredential,
  onClearCredential,
  onSavePrivateKey,
  onClearPrivateKey,
  onImportServiceAccount,
  onFetchModels,
  onTestModel,
  onNewProviderNameChange,
  onNewProviderProtocolChange,
  onCreateProvider,
  onDeleteProvider,
}: {
  tab: ProviderTab;
  providers: ProviderView[];
  settings: FlashcardSettings;
  draft: ProviderView | null;
  providerModels: ModelOption[];
  credentialValue: string;
  privateKeyValue: string;
  serviceAccountJson: string;
  connectivity: ConnectivityResult | null;
  busy: string;
  newProviderName: string;
  newProviderProtocol: ProviderProtocol;
  onTabChange: (tab: ProviderTab) => void;
  onSelect: (id: string) => void;
  onDraftChange: (draft: ProviderView) => void;
  onCredentialChange: (value: string) => void;
  onPrivateKeyChange: (value: string) => void;
  onServiceAccountJsonChange: (value: string) => void;
  onSaveProvider: () => void;
  onSaveCredential: () => void;
  onClearCredential: () => void;
  onSavePrivateKey: () => void;
  onClearPrivateKey: () => void;
  onImportServiceAccount: () => void;
  onFetchModels: () => void;
  onTestModel: () => void;
  onNewProviderNameChange: (value: string) => void;
  onNewProviderProtocolChange: (value: ProviderProtocol) => void;
  onCreateProvider: () => void;
  onDeleteProvider: (id: string) => void;
}) {
  const visibleProviders = providers.filter((provider) =>
    tab === "presets" ? provider.isBuiltin : !provider.isBuiltin,
  );
  const visibleDraft =
    draft && visibleProviders.some((provider) => provider.id === draft.id) ? draft : null;

  return (
    <section className="provider-page">
      <div className="tabs provider-tabs">
        <button
          className={tab === "presets" ? "tab active" : "tab"}
          onClick={() => onTabChange("presets")}
        >
          预设提供商
        </button>
        <button
          className={tab === "custom" ? "tab active" : "tab"}
          onClick={() => onTabChange("custom")}
        >
          新建提供商
        </button>
      </div>

      <div className="page-grid providers">
        <aside className="panel provider-list-panel">
          <div className="section-heading">
            <Bot className="size-4" />
            <span>{tab === "presets" ? "预设提供商" : "自定义提供商"}</span>
          </div>
          <div className="provider-list">
            {visibleProviders.length === 0 && (
              <div className="empty-state">还没有自定义提供商</div>
            )}
            {visibleProviders.map((provider) => (
              <button
                key={provider.id}
                className={
                  provider.id === settings.selectedProviderId
                    ? "provider-item active"
                    : "provider-item"
                }
                onClick={() => onSelect(provider.id)}
              >
                <span className="provider-icon">{providerIcon(provider.protocol)}</span>
                <span className="provider-meta">
                  <strong>{provider.name}</strong>
                  <small>{PROTOCOL_LABELS[provider.protocol]}</small>
                </span>
              </button>
            ))}
          </div>

          {tab === "custom" && (
            <div className="new-provider-box">
              <label className="field">
                <span>新提供商名称</span>
                <input
                  value={newProviderName}
                  onChange={(event) => onNewProviderNameChange(event.target.value)}
                />
              </label>
              <label className="field">
                <span>协议</span>
                <select
                  value={newProviderProtocol}
                  onChange={(event) =>
                    onNewProviderProtocolChange(event.target.value as ProviderProtocol)
                  }
                >
                  {CUSTOM_PROTOCOLS.map((protocol) => (
                    <option key={protocol} value={protocol}>
                      {PROTOCOL_LABELS[protocol]}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="secondary-button"
                disabled={!newProviderName.trim()}
                onClick={onCreateProvider}
              >
                <Plus className="size-4" />
                创建
              </button>
            </div>
          )}
        </aside>

        <section className="panel provider-editor-panel">
          {!visibleDraft ? (
            <div className="empty-state">
              {tab === "custom" ? "创建或选择自定义提供商" : "请选择预设提供商"}
            </div>
          ) : (
            <ProviderEditor
              draft={visibleDraft}
              providerModels={providerModels}
              credentialValue={credentialValue}
              privateKeyValue={privateKeyValue}
              serviceAccountJson={serviceAccountJson}
              connectivity={connectivity}
              busy={busy}
              onDraftChange={onDraftChange}
              onCredentialChange={onCredentialChange}
              onPrivateKeyChange={onPrivateKeyChange}
              onServiceAccountJsonChange={onServiceAccountJsonChange}
              onSaveProvider={onSaveProvider}
              onSaveCredential={onSaveCredential}
              onClearCredential={onClearCredential}
              onSavePrivateKey={onSavePrivateKey}
              onClearPrivateKey={onClearPrivateKey}
              onImportServiceAccount={onImportServiceAccount}
              onFetchModels={onFetchModels}
              onTestModel={onTestModel}
              onDeleteProvider={onDeleteProvider}
            />
          )}
        </section>
      </div>
    </section>
  );
}

function ProviderEditor({
  draft,
  providerModels,
  credentialValue,
  privateKeyValue,
  serviceAccountJson,
  connectivity,
  busy,
  onDraftChange,
  onCredentialChange,
  onPrivateKeyChange,
  onServiceAccountJsonChange,
  onSaveProvider,
  onSaveCredential,
  onClearCredential,
  onSavePrivateKey,
  onClearPrivateKey,
  onImportServiceAccount,
  onFetchModels,
  onTestModel,
  onDeleteProvider,
}: {
  draft: ProviderView;
  providerModels: ModelOption[];
  credentialValue: string;
  privateKeyValue: string;
  serviceAccountJson: string;
  connectivity: ConnectivityResult | null;
  busy: string;
  onDraftChange: (draft: ProviderView) => void;
  onCredentialChange: (value: string) => void;
  onPrivateKeyChange: (value: string) => void;
  onServiceAccountJsonChange: (value: string) => void;
  onSaveProvider: () => void;
  onSaveCredential: () => void;
  onClearCredential: () => void;
  onSavePrivateKey: () => void;
  onClearPrivateKey: () => void;
  onImportServiceAccount: () => void;
  onFetchModels: () => void;
  onTestModel: () => void;
  onDeleteProvider: (id: string) => void;
}) {
  return (
    <>
      <div className="section-heading">
        {providerIcon(draft.protocol)}
        <span>{draft.name}</span>
      </div>
      <div className="form-grid">
        <label className="field">
          <span>名称</span>
          <input
            value={draft.name}
            disabled={draft.isBuiltin}
            onChange={(event) => onDraftChange({ ...draft, name: event.target.value })}
          />
        </label>
        <label className="field">
          <span>协议</span>
          <input disabled value={PROTOCOL_LABELS[draft.protocol]} />
        </label>
        <label className="field wide">
          <span>Base URL</span>
          <input
            value={draft.baseUrl}
            onChange={(event) => onDraftChange({ ...draft, baseUrl: event.target.value })}
          />
        </label>
        {draft.protocol !== "agent-platform" && (
          <label className="field">
            <span>凭据类型</span>
            <select
              value={draft.credentialKind}
              disabled={draft.protocol !== "gemini"}
              onChange={(event) =>
                onDraftChange({ ...draft, credentialKind: event.target.value as CredentialKind })
              }
            >
              {credentialOptions(draft.protocol).map((kind) => (
                <option key={kind} value={kind}>
                  {CREDENTIAL_LABELS[kind]}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {draft.protocol === "agent-platform" ? (
        <AgentPlatformProviderFields
          draft={draft}
          privateKeyValue={privateKeyValue}
          serviceAccountJson={serviceAccountJson}
          onDraftChange={onDraftChange}
          onPrivateKeyChange={onPrivateKeyChange}
          onServiceAccountJsonChange={onServiceAccountJsonChange}
          onSavePrivateKey={onSavePrivateKey}
          onClearPrivateKey={onClearPrivateKey}
          onImportServiceAccount={onImportServiceAccount}
        />
      ) : (
        <div className="credential-block">
          <label className="field">
            <span>API Key</span>
            <input
              type="password"
              autoComplete="off"
              value={credentialValue}
              placeholder={draft.credentialMask ?? "尚未保存 API Key"}
              onChange={(event) => onCredentialChange(event.target.value)}
            />
          </label>
          <div className="button-row">
            <button
              className="secondary-button"
              disabled={!credentialValue.trim()}
              onClick={onSaveCredential}
            >
              <KeyRound className="size-4" />
              保存 API Key
            </button>
            <button className="ghost-button" onClick={onClearCredential}>
              清除
            </button>
          </div>
        </div>
      )}

      <div className="model-row">
        <label className="field model-field">
          <span>模型</span>
          <input
            list={`models-${draft.id}`}
            value={draft.selectedModel}
            onChange={(event) => onDraftChange({ ...draft, selectedModel: event.target.value })}
          />
          <datalist id={`models-${draft.id}`}>
            {providerModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label}
              </option>
            ))}
          </datalist>
        </label>
        <button className="secondary-button" disabled={busy === "models"} onClick={onFetchModels}>
          {busy === "models" ? (
            <LoaderCircle className="size-4 spin" />
          ) : (
            <CloudDownload className="size-4" />
          )}
          获取模型
        </button>
      </div>

      <div className="form-grid compact">
        <label className="field">
          <span>思考强度</span>
          <select
            value={draft.thinkingLevel}
            onChange={(event) =>
              onDraftChange({ ...draft, thinkingLevel: event.target.value as ThinkingLevel })
            }
          >
            {(["none", "low", "medium", "high", "max"] as ThinkingLevel[]).map((level) => (
              <option
                key={level}
                value={level}
                disabled={!draft.capabilities.thinkingOptions.includes(level)}
              >
                {THINKING_LABELS[level]}
              </option>
            ))}
          </select>
        </label>
        <label className="field switch-field">
          <span>联网</span>
          <button
            type="button"
            className={draft.webEnabled && draft.capabilities.webSupported ? "switch on" : "switch"}
            disabled={!draft.capabilities.webSupported}
            onClick={() => onDraftChange({ ...draft, webEnabled: !draft.webEnabled })}
            aria-pressed={draft.webEnabled}
          >
            <span />
          </button>
        </label>
      </div>

      <div className="button-row">
        <button className="primary-button" onClick={onSaveProvider}>
          <Save className="size-4" />
          保存提供商
        </button>
        <button className="secondary-button" onClick={onTestModel}>
          <Activity className="size-4" />
          测试连通性
        </button>
        {connectivity && (
          <span className={connectivity.success ? "test-result ok" : "test-result bad"}>
            {connectivity.success
              ? `${connectivity.latencyMs}ms · ${connectivity.responseText}`
              : connectivity.error}
          </span>
        )}
        {!draft.isBuiltin && (
          <button
            className="secondary-button danger"
            disabled={busy === "provider-delete"}
            onClick={() => onDeleteProvider(draft.id)}
          >
            {busy === "provider-delete" ? (
              <LoaderCircle className="size-4 spin" />
            ) : (
              <Trash2 className="size-4" />
            )}
            删除提供商
          </button>
        )}
      </div>
    </>
  );
}

function AgentPlatformProviderFields({
  draft,
  privateKeyValue,
  serviceAccountJson,
  onDraftChange,
  onPrivateKeyChange,
  onServiceAccountJsonChange,
  onSavePrivateKey,
  onClearPrivateKey,
  onImportServiceAccount,
}: {
  draft: ProviderView;
  privateKeyValue: string;
  serviceAccountJson: string;
  onDraftChange: (draft: ProviderView) => void;
  onPrivateKeyChange: (value: string) => void;
  onServiceAccountJsonChange: (value: string) => void;
  onSavePrivateKey: () => void;
  onClearPrivateKey: () => void;
  onImportServiceAccount: () => void;
}) {
  return (
    <div className="agent-fields">
      <div className="form-grid">
        <label className="field">
          <span>Project ID</span>
          <input
            value={draft.vertex.projectId}
            onChange={(event) =>
              onDraftChange({ ...draft, vertex: { ...draft.vertex, projectId: event.target.value } })
            }
          />
        </label>
        <label className="field">
          <span>Location</span>
          <input
            list="vertex-locations"
            value={draft.vertex.location}
            onChange={(event) =>
              onDraftChange({ ...draft, vertex: { ...draft.vertex, location: event.target.value } })
            }
          />
          <datalist id="vertex-locations">
            {VERTEX_LOCATIONS.map((location) => (
              <option key={location} value={location} />
            ))}
          </datalist>
        </label>
        <label className="field wide">
          <span>Client Email</span>
          <input
            value={draft.vertex.clientEmail}
            onChange={(event) =>
              onDraftChange({
                ...draft,
                vertex: { ...draft.vertex, clientEmail: event.target.value },
              })
            }
          />
        </label>
      </div>
      <label className="field">
        <span>Private Key</span>
        <textarea
          className="short-textarea"
          value={privateKeyValue}
          placeholder={draft.credentialMask ?? "尚未保存 private_key"}
          onChange={(event) => onPrivateKeyChange(event.target.value)}
        />
      </label>
      <div className="button-row">
        <button
          className="secondary-button"
          disabled={!privateKeyValue.trim()}
          onClick={onSavePrivateKey}
        >
          保存 Private Key
        </button>
        <button className="ghost-button" onClick={onClearPrivateKey}>
          清除
        </button>
      </div>
      <label className="field">
        <span>Service Account JSON</span>
        <textarea
          className="json-textarea"
          value={serviceAccountJson}
          onChange={(event) => onServiceAccountJsonChange(event.target.value)}
        />
      </label>
      <button
        className="secondary-button"
        disabled={!serviceAccountJson.trim()}
        onClick={onImportServiceAccount}
      >
        解析并保存 JSON
      </button>
    </div>
  );
}

function cloneProvider(provider: ProviderView): ProviderView {
  return {
    ...provider,
    vertex: { ...provider.vertex },
    capabilities: {
      ...provider.capabilities,
      thinkingOptions: [...provider.capabilities.thinkingOptions],
    },
  };
}

function sanitizeProvider(provider: ProviderView): ProviderView {
  const capabilities = modelCapabilities(provider.protocol, provider.selectedModel);
  return {
    ...provider,
    credentialKind: normalizeCredentialKind(provider.protocol, provider.credentialKind),
    thinkingLevel: capabilities.thinkingOptions.includes(provider.thinkingLevel)
      ? provider.thinkingLevel
      : "none",
    webEnabled: provider.webEnabled && capabilities.webSupported,
    capabilities,
  };
}

function modelCapabilities(protocol: ProviderProtocol, model: string): ModelCapabilities {
  const normalized = model.trim().replace(/^models\//, "").toLowerCase();
  const thinking =
    protocol === "openai-responses" &&
    (normalized.startsWith("gpt-5") ||
      normalized.startsWith("o1") ||
      normalized.startsWith("o3") ||
      normalized.startsWith("o4") ||
      normalized.startsWith("gpt-oss"));
  const geminiThinking =
    (protocol === "gemini" || protocol === "agent-platform") &&
    (normalized.startsWith("gemini-2.5") || normalized.startsWith("gemini-3"));
  const deepseekThinking = protocol === "deepseek" && normalized.startsWith("deepseek-v4");
  return {
    thinkingOptions: deepseekThinking
      ? ["none", "high", "max"]
      : thinking || geminiThinking
        ? ["none", "low", "medium", "high"]
        : ["none"],
    webSupported:
      protocol === "openai-responses" ||
      ((protocol === "gemini" || protocol === "agent-platform") &&
        normalized.includes("gemini")),
  };
}

function normalizeCredentialKind(protocol: ProviderProtocol, current: CredentialKind): CredentialKind {
  if (protocol === "gemini") {
    return current === "gemini-auth-api-key" ? current : "gemini-api-key";
  }
  if (protocol === "agent-platform") return "service-account";
  return "bearer";
}

function credentialOptions(protocol: ProviderProtocol): CredentialKind[] {
  if (protocol === "gemini") return ["gemini-api-key", "gemini-auth-api-key"];
  if (protocol === "agent-platform") return ["service-account"];
  return ["bearer"];
}

export default App;
