import React, { useState, useEffect, useCallback, useRef } from "react";
import "./App.css";

const ipcRenderer = (window as any).isInElectronRenderer
    ? (window as any).nodeRequire("electron").ipcRenderer
    : (window as any).ipcRendererStub;

const App = () => {
    const [selectedFolder, setSelectedFolder] = useState("");
    const [selectedModel, setSelectedModel] = useState("RetinaFace");
    const [confidenceThreshold, setConfidenceThreshold] = useState(0.9);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isStarting, setIsStarting] = useState(false);
    const [progress, setProgress] = useState(0);
    const [progressMessages, setProgressMessages] = useState<string[]>([]);
    const [hasProgressMessages, setHasProgressMessages] = useState(false);
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    // variant -> mode ("face"/"hand"/...), returned by get_models. This is the
    // reliable source the Model dropdown filters on: it travels with the model
    // list itself, so filtering works even when the detector registry (a separate
    // list_detectors call) is unavailable.
    const [modelModes, setModelModes] = useState<Record<string, string>>({});
    type DetectorInfo = { name: string; mode: string; variants: string[]; kind: string };
    const [detectorRegistry, setDetectorRegistry] = useState<Record<string, DetectorInfo>>({});
    // Modalities the app knows about. `MODE_AVAILABILITY` gates whether a real
    // detector exists (a disabled "coming soon" button is shown when `false`).
    // `HIDDEN_MODES` keeps a mode fully wired but out of the UI — the rest of the
    // pipeline (model dropdown, registry lookup, icons) still handles it, we just
    // don't render its button yet. To surface a hidden mode, drop it from
    // HIDDEN_MODES; to add a brand-new mode, append it to KNOWN_MODES too.
    const KNOWN_MODES: ReadonlyArray<string> = ["face", "hand", "speech"];
    const MODE_AVAILABILITY: Readonly<Record<string, boolean>> = {
        face: true,
        hand: true,
        speech: true,
    };
    // Speech is implemented as local transcription and is intentionally kept in
    // the same mode/model registry as the vision backends.
    const HIDDEN_MODES: ReadonlySet<string> = new Set([]);
    const [selectedMode, setSelectedMode] = useState<string>("face");
    // Whisper checkpoint size for Speech mode, mirroring the sizes the
    // requester's reference script exposes. large-v2 matches the quality of
    // the reference outputs; smaller sizes trade accuracy for speed.
    const WHISPER_SIZES: ReadonlyArray<string> = [
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v1", "large-v2", "large-v3",
        "turbo", "large-v3-turbo",
    ];
    const [whisperSize, setWhisperSize] = useState<string>("large-v2");
    const [results, setResults] = useState<any[]>([]); // eslint-disable-line @typescript-eslint/no-unused-vars
    const [resultsFolder, setResultsFolder] = useState("");
    const [completedResultsFolder, setCompletedResultsFolder] = useState("");
    const [isVideoFile, setIsVideoFile] = useState(false);
    const [pythonReady, setPythonReady] = useState(false);
    // Live startup feedback for the loading screen: latest status line + elapsed time.
    const [startupStatus, setStartupStatus] = useState<string>("");
    const [startupElapsedMs, setStartupElapsedMs] = useState<number>(0);
    // Notify-only "you're behind the latest release" banner. `updateInfo` is
    // populated by a one-shot check against GitHub Releases (see main process);
    // `updateBannerDismissed` hides it after the user closes it for that version.
    type UpdateInfo = {
        currentVersion: string;
        latestVersion: string | null;
        updateAvailable: boolean;
        releaseUrl: string | null;
        releaseName: string | null;
        checked: boolean;
    };
    const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
    const [updateBannerDismissed, setUpdateBannerDismissed] = useState<boolean>(false);

    // Send command to Python via IPC
    const sendPythonCommand = useCallback((command: any): Promise<any> => {
        return new Promise((resolve, reject) => {
            if (!ipcRenderer) {
                reject(new Error("IPC not available"));
                return;
            }

            const timeout = setTimeout(() => {
                reject(new Error("Command timeout"));
            }, 120000); // 120 second timeout (2 minutes)

            // Listen for response
            const handleResponse = (event: any, data: any) => {
                clearTimeout(timeout);
                ipcRenderer.removeListener("python-response", handleResponse);
                
                if (data.error) {
                    reject(new Error(data.error));
                } else {
                    resolve(data.response);
                }
            };

            ipcRenderer.on("python-response", handleResponse);
            ipcRenderer.send("python-command", command);
        });
    }, []);

    // Load detector registry (modality + variants + kind, used for grouping/icons)
    const loadDetectorRegistry = useCallback(async () => {
        try {
            const response = await sendPythonCommand({ type: 'list_detectors' });
            console.log("list_detectors response:", response);
            if (response && response.status === 'success' && response.detectors) {
                setDetectorRegistry(response.detectors);
            } else {
                console.warn(
                    "[FALLBACK] loadDetectorRegistry: no usable detector registry in response — " +
                    "Model dropdown will fall back to an ungrouped list that does NOT filter by Mode.",
                    response,
                );
            }
        } catch (error) {
            console.warn("[FALLBACK] loadDetectorRegistry threw; Model dropdown will not filter by Mode:", error);
        }
    }, [sendPythonCommand]);

    // Applied once, on the first successful get_models. Status polling can make
    // loadAvailableModels run again later; re-applying the Face default then
    // would clobber whatever the user has since selected.
    const didApplyDefaultModelRef = useRef(false);

    // Load available models
    const loadAvailableModels = useCallback(async () => {
        try {
            console.log("Loading available models...");
            const response = await sendPythonCommand({ type: 'get_models' });

            if (response.status === 'success') {
                console.log("Available models loaded:", response.models);
                setAvailableModels(response.models);
                // model_modes is the variant->mode map the dropdown filters on.
                // Older backends may omit it; default to an empty map (the UI then
                // falls back to the detector registry, then to an unfiltered list).
                setModelModes(response.model_modes || {});

                if (didApplyDefaultModelRef.current) {
                    return;
                }
                didApplyDefaultModelRef.current = true;

                // Set RetinaFace as default if available, otherwise use best YOLO face model
                if (response.models.includes("RetinaFace") && selectedModel === "RetinaFace") {
                    console.log("RetinaFace is available and already selected as default");
                    setConfidenceThreshold(0.9);
                } else if (response.models.includes("RetinaFace")) {
                    console.log("Auto-selecting RetinaFace as the default model");
                    setSelectedModel("RetinaFace");
                    setConfidenceThreshold(0.9);
                } else if (response.models.includes("yolov8l-face.pt")) {
                    console.log("RetinaFace not available, auto-selecting best YOLO face model: yolov8l-face.pt");
                    setSelectedModel("yolov8l-face.pt");
                    setConfidenceThreshold(0.7);
                } else {
                    console.warn(
                        "[FALLBACK] loadAvailableModels: no default Face model available; " +
                        "leaving the current selection. The one-time alignment effect will pick " +
                        "the first available model.",
                        response.models,
                    );
                }
            } else {
                console.warn(
                    "[FALLBACK] loadAvailableModels: get_models did not succeed; using [\"RetinaFace\"] as a stand-in list.",
                    response,
                );
                setAvailableModels(["RetinaFace"]); // Fallback
            }
        } catch (error) {
            console.warn("[FALLBACK] loadAvailableModels threw; using [\"RetinaFace\"] as a stand-in list:", error);
            setAvailableModels(["RetinaFace"]); // Fallback
        }
    }, [sendPythonCommand, selectedModel]);

    // Check Python status
    const checkPythonStatus = useCallback(() => {
        if (ipcRenderer) {
            ipcRenderer.send("getPythonStatus");
        }
    }, []);

    const fetchResults = useCallback(async () => {
        try {
            const response = await sendPythonCommand({ type: 'get_results' });
            
            if (response.status === 'success') {
                console.log("Final results received:", response.results.length, "detections");
                setResults(response.results);
            } else {
                console.error("Failed to fetch results:", response.message);
            }
        } catch (error) {
            console.error("Error fetching results:", error);
        }
    }, [sendPythonCommand]);

    const handleCompletionEvent = useCallback((data: any) => {
        console.log("Completion event:", data);
        console.log("Completion event data.results_folder:", data.results_folder);
        
        switch (data.status) {
            case 'processing_started':
                console.log("Backend processing started");
                setIsProcessing(true);
                setIsStarting(false);
                setProgress(0);
                break;
                
            case 'image_completed':
                setProgress(data.progress_percent);
                console.log(`Image ${data.image_index}/${data.total_images} completed: ${data.detections_in_image} detections found`);
                break;
                
            case 'frame_completed':
                setProgress(data.progress_percent);
                console.log(`Frame ${data.frame_index} at ${data.timestamp.toFixed(1)}s: ${data.detections_in_frame} faces found`);
                break;

            case 'audio_completed':
                setProgress(data.progress_percent);
                break;
                
            case 'completed':
            case 'finished':
                console.log("Processing completed, fetching final results");
                setIsProcessing(false);
                setIsStarting(false);
                setProgress(100);
                
                // Capture the results folder path
                if (data.results_folder) {
                    console.log("Setting completedResultsFolder to:", data.results_folder);
                    setCompletedResultsFolder(data.results_folder);
                } else {
                    console.log("WARNING: No results_folder in completion event!");
                }
                
                // Fetch final results
                fetchResults();
                break;
                
            case 'error':
                console.error("Processing error:", data.error);
                setIsProcessing(false);
                setIsStarting(false);
                setProgressMessages(prev => [...prev, `❌ Error: ${data.error}`]);
                setHasProgressMessages(true);
                break;
        }
    }, [fetchResults]);

    // One-shot, notify-only update check against GitHub Releases. The main
    // process does the network call and fails silently (offline/error -> no
    // banner); here we just store the result and honour a per-version dismissal
    // so the banner returns only when a newer release lands.
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                if (!ipcRenderer || typeof ipcRenderer.invoke !== "function") return;
                const info: UpdateInfo = await ipcRenderer.invoke("check-for-updates");
                if (cancelled || !info) return;
                setUpdateInfo(info);
                if (info.updateAvailable && info.latestVersion) {
                    const dismissedFor = window.localStorage.getItem("updateBannerDismissedVersion");
                    if (dismissedFor === info.latestVersion) {
                        setUpdateBannerDismissed(true);
                    }
                } else if (!info.checked) {
                    console.warn("[FALLBACK] Update check did not complete (offline or GitHub unreachable); update banner suppressed.");
                }
            } catch (error) {
                console.warn("[FALLBACK] Update check threw; update banner suppressed:", error);
            }
        })();
        return () => { cancelled = true; };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // Open the release page in the user's default browser (main process guards to https).
    const handleOpenRelease = useCallback(() => {
        if (updateInfo && updateInfo.releaseUrl && ipcRenderer && typeof ipcRenderer.invoke === "function") {
            ipcRenderer.invoke("open-external", updateInfo.releaseUrl);
        }
    }, [updateInfo]);

    // Hide the banner and remember the dismissal for this specific latest version.
    const handleDismissUpdateBanner = useCallback(() => {
        if (updateInfo && updateInfo.latestVersion) {
            window.localStorage.setItem("updateBannerDismissedVersion", updateInfo.latestVersion);
        }
        setUpdateBannerDismissed(true);
    }, [updateInfo]);

    // Handle Python events
    useEffect(() => {
        if (!ipcRenderer) return;

        const handlePythonEvent = (event: any, eventData: any) => {
            console.log("Python event received:", eventData);
            
            switch (eventData.type) {
                case 'progress':
                    if (!eventData.data.includes('ℹ️ DEBUG:') && !eventData.data.includes('Processing stopped by user')) {
                        const message = eventData.data;
                        
                        // Check if this is a download progress update (contains "Downloading" and percentage)
                        const isDownloadProgress = message.includes('⏳ Downloading') && message.includes('%');
                        
                        setProgressMessages(prev => {
                            if (isDownloadProgress && prev.length > 0) {
                                // Check if the last message was also a download progress for the same model
                                const lastMessage = prev[prev.length - 1];
                                const currentModelMatch = message.match(/Downloading ([^:]+):/);
                                const lastModelMatch = lastMessage.match(/Downloading ([^:]+):/);
                                const currentModel = currentModelMatch ? currentModelMatch[1] : null;
                                const lastModel = lastModelMatch ? lastModelMatch[1] : null;
                                
                                if (lastMessage.includes('⏳ Downloading') && currentModel === lastModel) {
                                    // Update the last message instead of adding a new one
                                    return [...prev.slice(0, -1), message];
                                }
                            }
                            
                            // For all other messages or initial download message, add normally
                            return [...prev, message];
                        });
                        setHasProgressMessages(true);
                    }
                    break;
                    
                case 'completion':
                    handleCompletionEvent(eventData.data);
                    break;

                case 'stderr':
                    // Backend startup logs are forwarded here. While the app is
                    // still warming up, show a readable "current step" so the
                    // loading screen reflects real progress. Skip noisy [timing]
                    // lines and empty output.
                    if (typeof eventData.data === 'string') {
                        const line = eventData.data.trim();
                        if (line && !line.startsWith('[timing]')) {
                            setStartupStatus(line);
                        }
                    }
                    break;

                default:
                    console.log("Unknown event type:", eventData.type);
            }
        };

        const handlePythonStatus = (event: any, statusData: any) => {
            console.log("Python status:", statusData);
            setPythonReady(statusData.ready);

            // Surface any status message / elapsed time the main process sends
            // (e.g. the slow-startup warning) so the loading screen isn't static.
            if (typeof statusData.message === "string" && statusData.message) {
                setStartupStatus(statusData.message);
            }
            if (typeof statusData.elapsedMs === "number") {
                setStartupElapsedMs(statusData.elapsedMs);
            }

            if (statusData.ready && availableModels.length === 0) {
                // Load models when Python becomes ready
                loadAvailableModels();
                loadDetectorRegistry();
            }
        };

        ipcRenderer.on("python-event", handlePythonEvent);
        ipcRenderer.on("pythonStatus", handlePythonStatus);

        // Check status immediately
        checkPythonStatus();

        return () => {
            ipcRenderer.removeListener("python-event", handlePythonEvent);
            ipcRenderer.removeListener("pythonStatus", handlePythonStatus);
        };
    }, [availableModels.length, loadAvailableModels, loadDetectorRegistry, handleCompletionEvent, checkPythonStatus]);

    // Mode ↔ model alignment is handled by a one-time effect (see below,
    // `didAlignModeRef`) plus handleModeChange. We intentionally do NOT keep a
    // model→mode sync effect here: it would fight explicit Mode clicks (snapping
    // the mode back whenever the target mode had no matching selected model).

    const handleSelectResultsFolder = () => {
        console.log("Prompting user to select results folder");
        if (ipcRenderer) {
            ipcRenderer.removeAllListeners("selected-folder");

            ipcRenderer.send("browse-folder");
            ipcRenderer.once("selected-folder", (event: any, payload: any) => {
                const sel = unpackSelection(payload);
                if (sel) {
                    console.log("User selected results folder:", sel.path);
                    setResultsFolder(sel.path);
                }
            });
        }
    };

    // The Electron main process sends {path, hasVideo} for both folder and
    // single-file browsing. Older code paths may still see a bare string.
    const unpackSelection = (payload: any): { path: string; hasVideo: boolean } | null => {
        if (!payload) return null;
        if (typeof payload === "string") return { path: payload, hasVideo: false };
        return { path: payload.path, hasVideo: !!payload.hasVideo };
    };

    const handleBrowseFolder = () => {
        console.log("User clicked 'Browse Folder' button");
        if (ipcRenderer) {
            ipcRenderer.removeAllListeners("selected-folder");

            ipcRenderer.send("browse-folder");
            ipcRenderer.once("selected-folder", (event: any, payload: any) => {
                const sel = unpackSelection(payload);
                if (sel) {
                    console.log("User selected folder:", sel.path);
                    setSelectedFolder(sel.path);
                    setIsVideoFile(false);
                }
            });
        }
    };

    const handleBrowseFile = () => {
        console.log("User clicked 'Browse File' button");
        if (ipcRenderer) {
            ipcRenderer.removeAllListeners("selected-folder");

            ipcRenderer.send("browse-file", { mode: selectedMode });
            ipcRenderer.once("selected-folder", (event: any, payload: any) => {
                const sel = unpackSelection(payload);
                if (sel) {
                    console.log("User selected file:", sel.path);
                    setSelectedFolder(sel.path);

                    const videoExtensions = ['.mp4', '.avi', '.mov'];
                    const isVideo = videoExtensions.some(ext => sel.path.toLowerCase().endsWith(ext));
                    setIsVideoFile(isVideo);
                    console.log("Video file detected:", isVideo);
                }
            });
        }
    };

    const getDisplayName = (modelName: string): string => {
        if (modelName === "RetinaFace") {
            return "RetinaFace";
        }

        // Hand detection models
        if (modelName === "HandObject-Baseline") {
            return "HandObject (100DOH baseline)";
        }

        // Handle YOLO face models
        if (modelName.includes("yolov8n-face")) {
            return "YOLOv8 Nano (Face)";
        } else if (modelName.includes("yolov8m-face")) {
            return "YOLOv8 Medium (Face)";
        } else if (modelName.includes("yolov8l-face")) {
            return "YOLOv8 Large (Face)";
        } else if (modelName.includes("yolov11m-face")) {
            return "YOLOv11 Medium (Face)";
        } else if (modelName.includes("yolov11l-face")) {
            return "YOLOv11 Large (Face)";
        } else if (modelName.includes("yolov12l-face")) {
            return "YOLOv12 Large (Face)";
        }
        
        // Handle general YOLO models
        if (modelName.includes("yolov8n.pt")) {
            return "YOLOv8 Nano";
        } else if (modelName.includes("yolov8s.pt")) {
            return "YOLOv8 Small";
        } else if (modelName.includes("yolov8m.pt")) {
            return "YOLOv8 Medium";
        } else if (modelName.includes("yolov8l.pt")) {
            return "YOLOv8 Large";
        } else if (modelName.includes("yolov8x.pt")) {
            return "YOLOv8 Extra Large";
        }
        
        // Fallback to original name if no match
        return modelName;
    };

    // Modality icon for the global confidence slider and dropdown grouping.
    const getDetectorIcon = (modalityName: string): string => {
        switch (modalityName) {
            case "face": return "👤";
            case "hand": return "✋";
            case "speech": return "🎤";
            case "pose": return "🧍";
            default: return "🔍";
        }
    };

    // "face_yolo" -> "YOLO", "face_retinaface" -> "Retinaface". For optgroup labels.
    const formatBackendName = (detectorKey: string): string => {
        const parts = detectorKey.split("_");
        if (parts.length < 2) return detectorKey;
        const backend = parts.slice(1).join(" ");
        return backend.charAt(0).toUpperCase() + backend.slice(1);
    };

    // Find which detector owns a given variant (e.g. "yolov8n-face.pt" -> "face_yolo").
    const getDetectorKeyForVariant = (variant: string): string | null => {
        for (const [key, info] of Object.entries(detectorRegistry)) {
            if (info.variants.includes(variant)) return key;
        }
        return null;
    };

    // Modality ("face"/"hand"/...) for a variant. Prefers the model_modes map
    // (travels with the model list, so it's present even when the detector
    // registry isn't); falls back to the registry. Returns "" if unknown.
    const modeOfVariant = (variant: string): string => {
        if (modelModes[variant]) return modelModes[variant];
        for (const info of Object.values(detectorRegistry)) {
            if (info.variants.includes(variant)) return info.mode || info.name;
        }
        return "";
    };

    // When a mode is picked, switch the model dropdown to the first variant
    // belonging to that mode (and recompute confidence default for it).
    const userSelectedModeRef = useRef(false);

    const handleModeChange = (newMode: string) => {
        // Do not let the asynchronous startup alignment below overwrite an
        // explicit user choice while model metadata is still arriving.
        userSelectedModeRef.current = true;
        setSelectedMode(newMode);
        // Keep the Model widget in sync with the Mode: switch to this mode's first
        // available variant. If the mode has no available models, clear the
        // selection so the dropdown shows its empty state rather than a stale
        // model from the previous mode.
        const firstVariantForMode = availableModels.find(v => modeOfVariant(v) === newMode);
        if (firstVariantForMode) {
            if (firstVariantForMode !== selectedModel) {
                handleModelChange(firstVariantForMode);
            }
        } else {
            console.warn(
                `[FALLBACK] handleModeChange("${newMode}"): no available model for this mode ` +
                `(available: ${availableModels.join(", ") || "none"}). Clearing model selection.`,
            );
            setSelectedModel("");
        }
    };

    const handleModelChange = (newModel: string) => {
        console.log("User changed model from", selectedModel, "to", newModel);
        setSelectedModel(newModel);
        
        // Set appropriate confidence thresholds based on model
        if (newModel === "RetinaFace") {
            setConfidenceThreshold(0.9);
        } else if (newModel.includes("HandObject")) {
            setConfidenceThreshold(0.55);
        } else if (newModel.includes("face")) {
            if (newModel.includes("yolov8n-face")) {
                setConfidenceThreshold(0.3);
            } else if (newModel.includes("yolov8m-face")) {
                setConfidenceThreshold(0.5);
            } else if (newModel.includes("yolov8l-face")) {
                setConfidenceThreshold(0.7);
            } else if (newModel.includes("yolov11m-face")) {
                setConfidenceThreshold(0.6);
            } else if (newModel.includes("yolov11l-face")) {
                setConfidenceThreshold(0.8);
            } else if (newModel.includes("yolov12l-face")) {
                setConfidenceThreshold(0.8);
            }
        } else {
            setConfidenceThreshold(0.5);
        }
    };

    // One-time alignment: once models + registry have loaded, land on a Mode +
    // Model that is actually available (e.g. if only Hand models are installed,
    // start on Hand rather than an unavailable Face default). Guarded by a ref so
    // it runs once and never fights an explicit Mode switch.
    const didAlignModeRef = useRef(false);
    useEffect(() => {
        if (didAlignModeRef.current && !userSelectedModeRef.current) return;
        // Need models plus at least one source of modality info (model_modes or
        // the registry) before we can align Mode to Model.
        const haveModeInfo =
            Object.keys(modelModes).length > 0 || Object.keys(detectorRegistry).length > 0;
        if (availableModels.length === 0 || !haveModeInfo) return;

        if (userSelectedModeRef.current) {
            const firstVariantForMode = availableModels.find(v => modeOfVariant(v) === selectedMode);
            if (firstVariantForMode && firstVariantForMode !== selectedModel) {
                handleModelChange(firstVariantForMode);
            }
            didAlignModeRef.current = true;
            // Alignment for this Mode click is done. Clear the flag, otherwise
            // this effect re-runs on every selectedModel change and snaps a
            // manual dropdown pick back to the mode's first variant.
            userSelectedModeRef.current = false;
            return;
        }

        didAlignModeRef.current = true;

        if (availableModels.includes(selectedModel)) {
            // Current model is valid; just make sure the Mode matches it.
            const mode = modeOfVariant(selectedModel);
            if (mode && mode !== selectedMode) setSelectedMode(mode);
        } else {
            // Default model isn't available — pick the first available one and
            // switch the Mode to its modality.
            const firstAvailable = availableModels[0];
            const mode = modeOfVariant(firstAvailable);
            if (mode) setSelectedMode(mode);
            if (firstAvailable) handleModelChange(firstAvailable);
        }
    }, [availableModels, detectorRegistry, modelModes, selectedModel, selectedMode, handleModelChange]);

    // Warn only when we have models but NO modality info at all (neither the
    // model_modes map nor the registry) — the one case where the dropdown can't
    // filter by the selected Mode and falls back to an unfiltered list.
    useEffect(() => {
        const haveModeInfo =
            Object.keys(modelModes).length > 0 || Object.keys(detectorRegistry).length > 0;
        if (availableModels.length > 0 && !haveModeInfo) {
            console.warn(
                "[FALLBACK] Model dropdown: no modality info (model_modes + registry both empty) — " +
                "rendering all available models unfiltered by the selected Mode.",
            );
        }
    }, [availableModels, detectorRegistry, modelModes]);

    // Modal state for the speaker-diarization Hugging Face token. The token
    // itself never lives in renderer state beyond the input field: it is sent
    // to the main process for Keychain-backed storage and cleared immediately.
    const [hfTokenModalOpen, setHfTokenModalOpen] = useState(false);
    const [hfTokenInput, setHfTokenInput] = useState("");
    const [hfTokenError, setHfTokenError] = useState("");
    const [hfTokenPromptToStart, setHfTokenPromptToStart] = useState(false);

    const closeHfTokenModal = () => {
        setHfTokenModalOpen(false);
        setHfTokenInput("");
        setHfTokenError("");
        setHfTokenPromptToStart(false);
    };

    const handleSaveHfToken = async () => {
        const shouldStart = hfTokenPromptToStart;
        try {
            const result = await ipcRenderer.invoke("set-hf-token", hfTokenInput.trim());
            if (result && result.ok) {
                closeHfTokenModal();
                if (shouldStart) startProcessingNow();
            } else {
                setHfTokenError((result && result.message) || "Could not store the token");
            }
        } catch (error) {
            setHfTokenError(String(error));
        }
    };

    const handleSkipHfToken = () => {
        const shouldStart = hfTokenPromptToStart;
        closeHfTokenModal();
        if (shouldStart) startProcessingNow();
    };

    const handleOpenHfTokenHelp = () => {
        if (ipcRenderer && typeof ipcRenderer.invoke === "function") {
            ipcRenderer.invoke("open-external", "https://huggingface.co/settings/tokens");
        }
    };

    const handleStartProcessing = async () => {
        console.log(selectedMode === "speech" ? "User clicked 'Start Transcription' button" : "User clicked 'Start Detection' button");
        console.log("Processing parameters:", {
            folder: selectedFolder,
            model: selectedModel,
            confidence: confidenceThreshold,
            resultsFolder: resultsFolder
        });

        if (!selectedFolder || !pythonReady) return;

        if (!resultsFolder) {
            console.log("No results folder selected, prompting user");
            handleSelectResultsFolder();
            return;
        }

        // Every speech backend can label speakers via the separate pyannote
        // diarization pipeline, but only with a (gated-model) Hugging Face
        // token. If none is configured anywhere, ask once before running.
        if (selectedMode === "speech" &&
                ipcRenderer && typeof ipcRenderer.invoke === "function") {
            try {
                const status = await ipcRenderer.invoke("get-hf-token-status");
                if (status && status.present === false) {
                    setHfTokenPromptToStart(true);
                    setHfTokenModalOpen(true);
                    return;
                }
            } catch (error) {
                console.warn("[FALLBACK] HF token status check failed; starting without speaker diarization:", error);
            }
        }

        startProcessingNow();
    };

    const startProcessingNow = async () => {
        setIsStarting(true);
        setResults([]);
        setProgress(0);
        setProgressMessages([]);
        setHasProgressMessages(true);
        setCompletedResultsFolder(""); // Clear previous results folder
        
        try {
            const detectorKey = getDetectorKeyForVariant(selectedModel);
            const data: any = {
                folder_path: selectedFolder,
                confidence: confidenceThreshold,
                model: selectedModel,  // legacy field, kept for backward compat
                save_results: true,
                results_folder: resultsFolder,
            };
            if (selectedMode === "speech") {
                data.whisper_size = whisperSize;
            }
            if (detectorKey) {
                data.detectors = [{
                    key: detectorKey,
                    variant: selectedModel,
                    confidence: confidenceThreshold,
                }];
            }

            const response = await sendPythonCommand({
                type: 'start_processing',
                data,
            });

            if (response.status === 'success') {
                console.log("Processing started successfully");
                // Processing state will be updated by events
            } else {
                console.error("Failed to start processing:", response.message);
                setProgressMessages(prev => [...prev, `❌ Error: ${response.message}`]);
                setIsStarting(false);
            }
        } catch (error) {
            console.error("Error starting processing:", error);
            setProgressMessages(prev => [...prev, `❌ Error: ${error}`]);
            setIsStarting(false);
        }
    };

    const handleStopProcessing = async () => {
        console.log("User clicked 'Stop Processing' button");
        
        try {
            const response = await sendPythonCommand({ type: 'stop_processing' });
            
            if (response.status === 'success') {
                console.log("Processing stopped successfully");
                setIsProcessing(false);
                setIsStarting(false);
            } else {
                console.error("Failed to stop processing:", response.message);
            }
        } catch (error) {
            console.error("Error stopping processing:", error);
        }
    };

    const handleOpenResultsFolder = () => {
        console.log("User clicked 'Open Results Folder' button");
        if (completedResultsFolder && ipcRenderer) {
            ipcRenderer.send("open-folder", completedResultsFolder);
        }
    };


    if (!pythonReady) {
        return (
            <div className="loading-container">
                <div className="loading-message">
                    Starting up detection engine...
                </div>
                {startupStatus && (
                    <div className="loading-substatus">
                        {startupStatus}
                    </div>
                )}
                {startupElapsedMs > 0 && (
                    <div className="loading-elapsed">
                        {(startupElapsedMs / 1000).toFixed(0)}s
                    </div>
                )}
                <div className="loading-animation">
                    <div className="dot"></div>
                    <div className="dot"></div>
                    <div className="dot"></div>
                </div>
            </div>
        );
    }

    return (
        <div className="App">
            {hfTokenModalOpen && (
                <div className="modal-overlay">
                    <div className="modal-card" role="dialog" aria-modal="true" aria-labelledby="hf-token-title">
                        <h3 id="hf-token-title">
                            <span role="img" aria-label="key">🔑</span> Hugging Face token for speaker labels
                        </h3>
                        <p>
                            Each speech model can tag utterances and words with a
                            speaker (SPEAKER_00, SPEAKER_01, …), but the diarization
                            model is gated: it needs your personal Hugging Face token,
                            and your account must have accepted the pyannote model
                            terms.
                        </p>
                        <p>
                            The token is stored encrypted with your macOS Keychain and
                            never written to disk in plain text.{" "}
                            <button type="button" className="hf-token-link" onClick={handleOpenHfTokenHelp}>
                                Get a token
                            </button>
                        </p>
                        <input
                            type="password"
                            className="hf-token-input"
                            placeholder="hf_..."
                            value={hfTokenInput}
                            onChange={(e) => setHfTokenInput(e.target.value)}
                            autoFocus
                        />
                        {hfTokenError && <div className="hf-token-error">{hfTokenError}</div>}
                        <div className="modal-actions">
                            <button type="button" className="browse-btn" onClick={closeHfTokenModal}>
                                Cancel
                            </button>
                            {hfTokenPromptToStart && (
                                <button type="button" className="browse-btn" onClick={handleSkipHfToken}>
                                    Continue without speakers
                                </button>
                            )}
                            <button
                                type="button"
                                className="start-btn modal-save-btn"
                                onClick={handleSaveHfToken}
                                disabled={!hfTokenInput.trim()}
                            >
                                {hfTokenPromptToStart ? "Save & Continue" : "Save"}
                            </button>
                        </div>
                    </div>
                </div>
            )}
            {updateInfo && updateInfo.updateAvailable && !updateBannerDismissed && (
                <div className="update-banner" role="status">
                    <span className="update-banner-text">
                        <span className="update-banner-icon" role="img" aria-label="Update available">⬆️</span>
                        A new version (v{updateInfo.latestVersion}) is available — you're on v{updateInfo.currentVersion}.
                    </span>
                    <span className="update-banner-actions">
                        <button type="button" className="update-banner-link" onClick={handleOpenRelease}>
                            View release
                        </button>
                        <button
                            type="button"
                            className="update-banner-dismiss"
                            onClick={handleDismissUpdateBanner}
                            aria-label="Dismiss update notification"
                            title="Dismiss"
                        >
                            ✕
                        </button>
                    </span>
                </div>
            )}
            <div className="app-container">
                <div className="left-panel">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
                        <h2 style={{ margin: 0 }}>TinyExplorer Detection App</h2>
                        <img src="dragon.png" alt="App Logo" style={{ width: '32px', height: '32px' }} />
                    </div>
                    
                    {/* Mode comes first: it decides which file types Browse File accepts. */}
                    <div className="control-section">
                        <label>Select Mode:</label>
                        <div className="mode-selector">
                            {KNOWN_MODES.filter(mode => !HIDDEN_MODES.has(mode)).map(mode => {
                                const hasDetector = MODE_AVAILABILITY[mode] === true;
                                const disabled = !hasDetector;
                                const modeLabel = mode.charAt(0).toUpperCase() + mode.slice(1);
                                const title = hasDetector
                                    ? `${modeLabel} detection`
                                    : `${modeLabel} detection — coming soon`;
                                return (
                                    <button
                                        key={mode}
                                        type="button"
                                        className={`mode-btn ${selectedMode === mode ? "active" : ""}`}
                                        disabled={disabled}
                                        title={title}
                                        onClick={() => handleModeChange(mode)}
                                    >
                                        <span role="img" aria-label={`${mode} modality`}>
                                            {getDetectorIcon(mode)}
                                        </span>{" "}
                                        {modeLabel}
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    <div className="control-section">
                        <label>Select File or Folder:</label>
                        <div className="file-input-group">
                            <input
                                type="text"
                                value={selectedFolder}
                                readOnly
                                placeholder="No file or folder selected..."
                                className="file-input"
                            />
                        </div>
                        <div className="button-group">
                            <button onClick={handleBrowseFile} className="browse-btn">Browse File</button>
                            <button onClick={handleBrowseFolder} className="browse-btn">Browse Folder</button>
                        </div>
                    </div>

                    <div className="control-section">
                        <label>Select Model:</label>
                        <select
                            value={selectedModel}
                            onChange={(e) => handleModelChange(e.target.value)}
                            className="model-select"
                        >
                            {(() => {
                                const noneForMode = (
                                    <option value="" disabled>
                                        No models available for this mode
                                    </option>
                                );
                                // Preferred: registry present -> group by detector
                                // (optgroups), filtered to the selected Mode.
                                if (Object.keys(detectorRegistry).length > 0) {
                                    const groups = Object.entries(detectorRegistry)
                                        .filter(([, info]) => (info.mode || info.name) === selectedMode)
                                        .map(([key, info]) => {
                                            const variants = info.variants.filter(v => availableModels.includes(v));
                                            if (variants.length === 0) return null;
                                            return (
                                                <optgroup
                                                    key={key}
                                                    label={formatBackendName(key)}
                                                >
                                                    {variants.map(model => (
                                                        <option key={model} value={model}>
                                                            {getDisplayName(model)}
                                                        </option>
                                                    ))}
                                                </optgroup>
                                            );
                                        })
                                        .filter(Boolean);
                                    return groups.length === 0 ? noneForMode : groups;
                                }
                                // No registry, but model_modes is available:
                                // ungrouped list filtered to the selected Mode.
                                if (Object.keys(modelModes).length > 0) {
                                    const forMode = availableModels.filter(m => modeOfVariant(m) === selectedMode);
                                    if (forMode.length === 0) return noneForMode;
                                    return forMode.map(model => (
                                        <option key={model} value={model}>
                                            {getDisplayName(model)}
                                        </option>
                                    ));
                                }
                                // Last resort: no modality info at all -> unfiltered list.
                                return availableModels.map(model => (
                                    <option key={model} value={model}>
                                        {getDisplayName(model)}
                                    </option>
                                ));
                            })()}
                        </select>
                        {selectedMode === "speech" && (
                            <button
                                type="button"
                                className="hf-token-link"
                                onClick={() => setHfTokenModalOpen(true)}
                                title="Configure the Hugging Face token used for speaker diarization"
                            >
                                <span role="img" aria-label="key">🔑</span> Speaker diarization token…
                            </button>
                        )}
                    </div>

                    {selectedMode === "speech" && (
                        <div className="control-section">
                            <label>Select Model Size:</label>
                            <select
                                value={whisperSize}
                                onChange={(e) => {
                                    console.log("User changed Whisper size from", whisperSize, "to", e.target.value);
                                    setWhisperSize(e.target.value);
                                }}
                                className="whisper-size-select"
                            >
                                {WHISPER_SIZES.map(size => (
                                    <option key={size} value={size}>{size}</option>
                                ))}
                            </select>
                        </div>
                    )}

                    {selectedMode !== "speech" && (
                        <div className="control-section">
                            <label>
                                {(() => {
                                    const detectorKey = getDetectorKeyForVariant(selectedModel);
                                    const info = detectorKey ? detectorRegistry[detectorKey] : null;
                                    return info ? (
                                        <span
                                            role="img"
                                            aria-label={`${info.name} modality`}
                                            className="modality-icon"
                                        >
                                            {getDetectorIcon(info.name)}{" "}
                                        </span>
                                    ) : null;
                                })()}
                                Select Confidence Threshold:
                            </label>
                            <div className="threshold-control">
                                <input
                                    type="range"
                                    min="0.1"
                                    max="1.0"
                                    step="0.01"
                                    value={confidenceThreshold}
                                    onChange={(e) => {
                                        const newValue = parseFloat(e.target.value);
                                        console.log("User adjusted confidence threshold from", confidenceThreshold, "to", newValue);
                                        setConfidenceThreshold(newValue);
                                    }}
                                    className="threshold-slider"
                                />
                                <span className="threshold-value">{confidenceThreshold.toFixed(2)}</span>
                            </div>
                        </div>
                    )}

                    <div className="control-section">
                        <label>Results will be saved to:</label>
                        <div className="file-input-group">
                            <input 
                                type="text" 
                                value={resultsFolder} 
                                readOnly 
                                placeholder="No results folder selected..."
                                className="file-input"
                            />
                            <button onClick={handleSelectResultsFolder} className="browse-btn">Select Results Folder</button>
                        </div>
                        {isVideoFile && (
                            <div className="file-info">
                                <small><span role="img" aria-label="movie camera">🎬</span> Video file detected - will process 1 frame per second</small>
                            </div>
                        )}
                    </div>

                    <div className="control-section">
                        {!isProcessing && !isStarting ? (
                            <button 
                                onClick={handleStartProcessing}
                                disabled={!selectedFolder || !resultsFolder || !pythonReady}
                                className="start-btn"
                            >
                                {selectedMode === "speech" ? "Start Transcription" : "Start Detection"}
                            </button>
                        ) : isStarting ? (
                            <button 
                                disabled
                                className="start-btn starting"
                            >
                                Starting...
                            </button>
                        ) : (
                            <button 
                                onClick={handleStopProcessing}
                                className="stop-btn"
                            >
                                Stop Processing
                            </button>
                        )}
                    </div>

                    {(isProcessing || isStarting) && (
                        <div className="progress-section">
                            <div className="progress-bar">
                                <div 
                                    className="progress-fill" 
                                    style={{ width: `${Math.min(progress, 100)}%` }}
                                />
                            </div>
                            <div className="progress-text">{progress.toFixed(1)}%</div>
                        </div>
                    )}

                </div>

                <div className="right-panel">
                    <div className="results-container">
                        {hasProgressMessages && (
                            <div className="progress-messages">
                                <h3>Progress Updates:</h3>
                                <div className="message-window">
                                    {progressMessages.map((message, index) => (
                                        <div key={index}>{message}</div>
                                    ))}
                                </div>
                                {completedResultsFolder && !isProcessing && !isStarting && (
                                    <div className="control-section" style={{ marginTop: '10px' }}>
                                        <button 
                                            onClick={handleOpenResultsFolder}
                                            className="browse-btn"
                                        >
                                            <span role="img" aria-label="folder">📁</span> Open Results Folder
                                        </button>
                                    </div>
                                )}
                            </div>
                        )}


                        {!hasProgressMessages && (
                            <div className="empty-state">
                                <p>Select a file or folder and start detection to see progress here.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;
