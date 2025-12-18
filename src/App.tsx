import React, { useState, useEffect, useCallback } from "react";
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
    const [results, setResults] = useState<any[]>([]); // eslint-disable-line @typescript-eslint/no-unused-vars
    const [resultsFolder, setResultsFolder] = useState("");
    const [completedResultsFolder, setCompletedResultsFolder] = useState("");
    const [isVideoFile, setIsVideoFile] = useState(false);
    const [pythonReady, setPythonReady] = useState(false);

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

    // Load available models
    const loadAvailableModels = useCallback(async () => {
        try {
            console.log("Loading available models...");
            const response = await sendPythonCommand({ type: 'get_models' });
            
            if (response.status === 'success') {
                console.log("Available models loaded:", response.models);
                setAvailableModels(response.models);
                
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
                }
            } else {
                console.error("Failed to load models:", response.message);
                setAvailableModels(["RetinaFace"]); // Fallback
            }
        } catch (error) {
            console.error("Error loading models:", error);
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
                console.log(`Image ${data.image_index}/${data.total_images} completed: ${data.detections_in_image} faces found`);
                break;
                
            case 'frame_completed':
                setProgress(data.progress_percent);
                console.log(`Frame ${data.frame_index} at ${data.timestamp.toFixed(1)}s: ${data.detections_in_frame} faces found`);
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
                    
                default:
                    console.log("Unknown event type:", eventData.type);
            }
        };

        const handlePythonStatus = (event: any, statusData: any) => {
            console.log("Python status:", statusData);
            setPythonReady(statusData.ready);
            
            if (statusData.ready && availableModels.length === 0) {
                // Load models when Python becomes ready
                loadAvailableModels();
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
    }, [availableModels.length, loadAvailableModels, handleCompletionEvent, checkPythonStatus]);

    const handleSelectResultsFolder = () => {
        console.log("Prompting user to select results folder");
        if (ipcRenderer) {
            ipcRenderer.removeAllListeners("selected-folder");
            
            ipcRenderer.send("browse-folder");
            ipcRenderer.once("selected-folder", (event: any, folderPath: string) => {
                if (folderPath) {
                    console.log("User selected results folder:", folderPath);
                    setResultsFolder(folderPath);
                }
            });
        }
    };

    const handleBrowseFolder = () => {
        console.log("User clicked 'Browse Folder' button");
        if (ipcRenderer) {
            ipcRenderer.removeAllListeners("selected-folder");
            
            ipcRenderer.send("browse-folder");
            ipcRenderer.once("selected-folder", (event: any, folderPath: string) => {
                if (folderPath) {
                    console.log("User selected folder:", folderPath);
                    setSelectedFolder(folderPath);
                    setIsVideoFile(false);
                    
                    // Prompt for results folder
                    setTimeout(() => {
                        handleSelectResultsFolder();
                    }, 100);
                }
            });
        }
    };

    const handleBrowseFile = () => {
        console.log("User clicked 'Browse File' button");
        if (ipcRenderer) {
            ipcRenderer.removeAllListeners("selected-folder");
            
            ipcRenderer.send("browse-file");
            ipcRenderer.once("selected-folder", (event: any, filePath: string) => {
                if (filePath) {
                    console.log("User selected file:", filePath);
                    setSelectedFolder(filePath);
                    
                    // Check if it's a video file
                    const videoExtensions = ['.mp4', '.avi', '.mov'];
                    const isVideo = videoExtensions.some(ext => filePath.toLowerCase().endsWith(ext));
                    setIsVideoFile(isVideo);
                    console.log("Video file detected:", isVideo);
                    
                    // Prompt for results folder
                    setTimeout(() => {
                        handleSelectResultsFolder();
                    }, 100);
                }
            });
        }
    };

    const getDisplayName = (modelName: string): string => {
        if (modelName === "RetinaFace") {
            return "RetinaFace";
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

    const handleModelChange = (newModel: string) => {
        console.log("User changed model from", selectedModel, "to", newModel);
        setSelectedModel(newModel);
        
        // Set appropriate confidence thresholds based on model
        if (newModel === "RetinaFace") {
            setConfidenceThreshold(0.9);
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

    const handleStartProcessing = async () => {
        console.log("User clicked 'Start Detection' button");
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
        
        setIsStarting(true);
        setResults([]);
        setProgress(0);
        setProgressMessages([]);
        setHasProgressMessages(true);
        setCompletedResultsFolder(""); // Clear previous results folder
        
        try {
            const response = await sendPythonCommand({
                type: 'start_processing',
                data: {
                    folder_path: selectedFolder,
                    confidence: confidenceThreshold,
                    model: selectedModel,
                    save_results: true,
                    results_folder: resultsFolder
                }
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
                    Starting up face detection engine...
                </div>
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
            <div className="app-container">
                <div className="left-panel">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
                        <h2 style={{ margin: 0 }}>TinyExplorer Detection App</h2>
                        <img src="dragon.png" alt="App Logo" style={{ width: '32px', height: '32px' }} />
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
                            {availableModels.map(model => (
                                <option key={model} value={model}>
                                    {getDisplayName(model)}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="control-section">
                        <label>Select Confidence Threshold:</label>
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
                                disabled={!selectedFolder || !pythonReady}
                                className="start-btn"
                            >
                                Start Detection
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