// For Linux AppImage runs, append --no-sandbox so double-click launches work on systems
// where the Chromium sandbox cannot initialize (e.g., user namespaces disabled).
// This must be done before importing electron app to take effect.
if (process.platform === "linux" && (process.env.APPIMAGE || process.env.APPDIR)) {
    process.argv.push("--no-sandbox");
    process.argv.push("--disable-dev-shm-usage");
    // Set environment variable as well
    process.env.ELECTRON_DISABLE_SANDBOX = '1';
}

import { app, BrowserWindow, ipcMain, dialog, shell, Tray, Menu, nativeImage } from "electron"; // tslint:disable-line
import * as path from "path";
import * as fs from "fs";
import * as https from "https";

// Wall-clock at main-process entry, used for lightweight startup phase timing.
// Verbose [timing] lines are gated behind the STARTUP_TIMING env var; the phase
// marks below let us measure process-start -> window-shown -> python-ready.
const processStart = Date.now();
const startupTimingEnabled = Boolean(process.env.STARTUP_TIMING);
function logStartupTiming(phase: string) {
    if (startupTimingEnabled) {
        try { console.log(`[timing] ${phase} ${Date.now() - processStart}ms`); } catch (e) {}
    }
}

// Speech detection requires audio, which only video files have. Use this to
// gate the "Speech" mode button in the renderer. We bail early on first match
// and cap the walk to keep huge folders responsive.
const VIDEO_EXTENSIONS: ReadonlyArray<string> = [".mp4", ".avi", ".mov"];
const FOLDER_SCAN_FILE_LIMIT = 5000;
const FOLDER_SCAN_DEPTH_LIMIT = 6;

function selectionHasVideo(selectedPath: string): boolean {
    try {
        const stat = fs.statSync(selectedPath);
        if (stat.isFile()) {
            return VIDEO_EXTENSIONS.some(ext => selectedPath.toLowerCase().endsWith(ext));
        }
        if (!stat.isDirectory()) return false;

        const stack: Array<{ dir: string; depth: number }> = [{ dir: selectedPath, depth: 0 }];
        let inspected = 0;
        while (stack.length > 0) {
            const { dir, depth } = stack.pop()!;
            if (depth > FOLDER_SCAN_DEPTH_LIMIT) continue;
            let entries: fs.Dirent[];
            try {
                entries = fs.readdirSync(dir, { withFileTypes: true });
            } catch {
                continue; // skip unreadable subdirs
            }
            for (const entry of entries) {
                if (++inspected > FOLDER_SCAN_FILE_LIMIT) return false;
                if (entry.isDirectory()) {
                    stack.push({ dir: path.join(dir, entry.name), depth: depth + 1 });
                } else if (entry.isFile()) {
                    const lower = entry.name.toLowerCase();
                    if (VIDEO_EXTENSIONS.some(ext => lower.endsWith(ext))) return true;
                }
            }
        }
        return false;
    } catch {
        return false;
    }
}

// Disable GPU acceleration for better compatibility with remote displays and AppImages
app.disableHardwareAcceleration();

// Mitigate Windows cache permission issues when running from protected folders (e.g., OneDrive)
if (process.platform === 'win32') {
    const os = require('os');
    const userData = path.join(os.homedir(), 'AppData', 'Roaming', 'TinyExplorer Detection App');
    app.setPath('userData', userData);
    app.setPath('cache', path.join(userData, 'Cache'));
}

const isDev = (process.env.NODE_ENV === "development");
let tray: Tray | null = null;
let mainWindow: BrowserWindow | null = null;
let splash: BrowserWindow | null = null;

// --- Notify-only update check ------------------------------------------------
// Surface an info banner in the renderer when a newer *app* release exists on
// GitHub. The repo hosts model-weight releases (tags like `handobj-weights-v1`,
// `v1.0.0-models`) alongside app releases (`v0.3.0`), and GitHub's
// /releases/latest can point at a weights release — so we list all releases and
// consider only tags shaped like a plain semver app version. Any failure
// (offline, HTTP error, rate limit) resolves to "no update" and is logged, never
// thrown, so the banner simply doesn't appear rather than breaking startup.
const GITHUB_OWNER = "cardiff-babylab";
const GITHUB_REPO = "tinyexplorer-detection-app";
// Matches "v0.3.0" but not "v1.0.0-models" / "handobj-weights-v1" / prereleases.
const APP_RELEASE_TAG = /^v(\d+)\.(\d+)\.(\d+)$/;

interface UpdateInfo {
    currentVersion: string;
    latestVersion: string | null;
    updateAvailable: boolean;
    releaseUrl: string | null;
    releaseName: string | null;
    checked: boolean; // false when the check couldn't complete (offline/error)
}

function parseAppVersion(tagOrVersion: string): [number, number, number] | null {
    const normalized = tagOrVersion.trim().startsWith("v") ? tagOrVersion.trim() : `v${tagOrVersion.trim()}`;
    const m = APP_RELEASE_TAG.exec(normalized);
    return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
}

// >0 if a is newer than b, <0 if older, 0 if equal.
function compareVersion(a: [number, number, number], b: [number, number, number]): number {
    for (let i = 0; i < 3; i++) {
        if (a[i] !== b[i]) return a[i] - b[i];
    }
    return 0;
}

function fetchLatestAppRelease(): Promise<{ tag: string; name: string; url: string } | null> {
    return new Promise((resolve) => {
        const req = https.request(
            {
                hostname: "api.github.com",
                path: `/repos/${GITHUB_OWNER}/${GITHUB_REPO}/releases?per_page=30`,
                method: "GET",
                headers: {
                    "User-Agent": `${GITHUB_REPO}-update-check`,
                    Accept: "application/vnd.github+json",
                },
                timeout: 5000,
            },
            (res) => {
                const status = res.statusCode || 0;
                if (status < 200 || status >= 300) {
                    console.warn(`[update-check] GitHub API returned HTTP ${status}; skipping update check.`);
                    res.resume(); // drain so the socket frees
                    resolve(null);
                    return;
                }
                let body = "";
                res.setEncoding("utf8");
                res.on("data", (chunk) => (body += chunk));
                res.on("end", () => {
                    try {
                        const releases = JSON.parse(body);
                        if (!Array.isArray(releases)) {
                            console.warn("[update-check] Unexpected releases payload; skipping update check.");
                            resolve(null);
                            return;
                        }
                        let best: { tag: string; name: string; url: string; ver: [number, number, number] } | null = null;
                        for (const r of releases) {
                            if (!r || r.draft || r.prerelease) continue;
                            const ver = parseAppVersion(String(r.tag_name || ""));
                            if (!ver) continue; // skips weights/models/prerelease-shaped tags
                            if (!best || compareVersion(ver, best.ver) > 0) {
                                best = { tag: String(r.tag_name), name: String(r.name || r.tag_name), url: String(r.html_url), ver };
                            }
                        }
                        resolve(best ? { tag: best.tag, name: best.name, url: best.url } : null);
                    } catch (e) {
                        console.warn("[update-check] Failed to parse releases response:", e);
                        resolve(null);
                    }
                });
            },
        );
        req.on("error", (e) => {
            console.warn("[update-check] Network error during update check (offline?):", (e as Error).message);
            resolve(null);
        });
        req.on("timeout", () => {
            console.warn("[update-check] Update check timed out; skipping.");
            req.destroy();
            resolve(null);
        });
        req.end();
    });
}

async function getUpdateInfo(): Promise<UpdateInfo> {
    const currentVersion = app.getVersion();
    const current = parseAppVersion(currentVersion);
    const latest = await fetchLatestAppRelease();
    if (!latest) {
        return { currentVersion, latestVersion: null, updateAvailable: false, releaseUrl: null, releaseName: null, checked: false };
    }
    const latestVer = parseAppVersion(latest.tag);
    const updateAvailable = Boolean(current && latestVer && compareVersion(latestVer, current) > 0);
    return {
        currentVersion,
        latestVersion: latest.tag.replace(/^v/, ""),
        updateAvailable,
        releaseUrl: latest.url,
        releaseName: latest.name,
        checked: true,
    };
}

// Registered at module scope (not inside createWindow) so it's wired exactly
// once regardless of window re-creation.
ipcMain.handle("check-for-updates", async (): Promise<UpdateInfo> => {
    try {
        return await getUpdateInfo();
    } catch (error) {
        console.warn("[update-check] Unexpected error during update check:", error);
        return { currentVersion: app.getVersion(), latestVersion: null, updateAvailable: false, releaseUrl: null, releaseName: null, checked: false };
    }
});

// Open an external https URL (e.g. the GitHub release page) in the default
// browser. Restricted to https to avoid opening arbitrary schemes.
ipcMain.handle("open-external", async (_event: any, url: string): Promise<boolean> => {
    if (typeof url === "string" && /^https:\/\//i.test(url)) {
        await shell.openExternal(url);
        return true;
    }
    console.warn("[update-check] Refused to open non-https external URL:", url);
    return false;
});

// Minimal, self-contained splash shown the instant the app starts, so the user
// sees immediate feedback instead of a blank screen while the React bundle loads
// and the Python backend warms up. Kept as an inline data URL (no external assets,
// no extra packaged files) so it always renders, even before first paint of the
// main window. Closed once the main window is ready-to-show (or Python is ready).
const SPLASH_HTML = `<!doctype html><html><head><meta charset="utf-8"><style>
  html,body{margin:0;height:100%;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  body{display:flex;flex-direction:column;align-items:center;justify-content:center;
       background:linear-gradient(135deg,#1e293b,#0f172a);color:#e2e8f0;-webkit-user-select:none}
  .title{font-size:18px;font-weight:600;margin-bottom:6px}
  .sub{font-size:12px;color:#94a3b8;margin-bottom:22px}
  .dots{display:flex;gap:8px}
  .dot{width:9px;height:9px;border-radius:50%;background:#38bdf8;animation:b 1.2s infinite ease-in-out}
  .dot:nth-child(2){animation-delay:.15s}.dot:nth-child(3){animation-delay:.3s}
  @keyframes b{0%,80%,100%{opacity:.25;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
</style></head><body>
  <div class="title">TinyExplorer Detection App</div>
  <div class="sub">Starting up&hellip;</div>
  <div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
</body></html>`;

function createSplash() {
    try {
        splash = new BrowserWindow({
            width: 360,
            height: 220,
            frame: false,
            resizable: false,
            center: true,
            show: true,
            skipTaskbar: true,
            webPreferences: {},
        });
        splash.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(SPLASH_HTML));
        // Safety net: never let the splash outlive startup, even if 'ready-to-show'
        // never fires for some reason.
        setTimeout(closeSplash, 30000);
    } catch (e) {
        try { console.error("Failed to create splash window:", e); } catch (e2) {}
        splash = null;
    }
}

function closeSplash() {
    if (splash && !splash.isDestroyed()) {
        try { splash.close(); } catch (e) {}
    }
    splash = null;
}

// Add isQuitting property to app instance
(app as any).isQuitting = false;

// Set command line switches early for AppImage
if (process.platform === "linux" && (process.env.APPIMAGE || process.env.APPDIR)) {
    app.commandLine.appendSwitch("no-sandbox");
    app.commandLine.appendSwitch("disable-dev-shm-usage");
}

app.on("window-all-closed", () => {
    // Don't quit the app when window is closed on Linux/Windows if we have a tray
    if (process.platform !== "darwin" && !tray) {
        app.quit();
    }
});

// Ensure app quits properly
app.on('before-quit', () => {
    (app as any).isQuitting = true;
});

app.on("ready", () => {
    logStartupTiming("app_ready");
    if (isDev) {
        const sourceMapSupport = require("source-map-support"); // tslint:disable-line
        sourceMapSupport.install();
    }

    // Show the splash immediately so the user gets instant visual feedback.
    createSplash();

    // Import Python subprocess handler after app is ready
    require("./with-python-subprocess");

    createWindow();
    createTray();
});


function createTray() {
    // Get the icon path - use PNG for Linux
    // Using path.join for cross-platform compatibility
    let iconPath = isDev 
        ? path.join(__dirname, '..', 'resources', 'icons', '256x256.png')
        : path.join(process.resourcesPath, 'resources', 'icons', '256x256.png');
    
    // Fallback to alternative icon locations if primary doesn't exist
    const fs = require('fs');
    if (!fs.existsSync(iconPath)) {
        // Try alternative paths - more extensive fallback list for packaged apps
        const alternatives = [
            // Standard packaged locations
            path.join(process.resourcesPath, 'app', 'resources', 'icons', '256x256.png'),
            path.join(process.resourcesPath, 'app.asar', 'resources', 'icons', '256x256.png'),
            path.join(process.resourcesPath, 'resources', 'icon.png'),
            path.join(process.resourcesPath, 'app', 'resources', 'icon.png'),
            path.join(process.resourcesPath, 'app.asar', 'resources', 'icon.png'),
            path.join(process.resourcesPath, 'dragon-512.png'),
            // Development mode fallbacks
            path.join(__dirname, '..', 'resources', 'icon.png'),
            path.join(__dirname, '..', 'graphics', 'icons', 'dragon-256.png'),
            path.join(__dirname, '..', 'graphics', 'dragon.png')
        ];
        
        for (const alt of alternatives) {
            if (fs.existsSync(alt)) {
                iconPath = alt;
                console.log(`Using fallback tray icon: ${alt}`);
                break;
            }
        }
        
        // If still no icon found, log the attempted paths for debugging
        if (!fs.existsSync(iconPath)) {
            console.error(`Warning: Tray icon not found at any location. Attempted paths:`, [iconPath, ...alternatives].join(', '));
            console.error(`process.resourcesPath: ${process.resourcesPath}`);
            console.error(`__dirname: ${__dirname}`);
        }
    }
    
    // Create a NativeImage from the icon path
    const trayIcon = nativeImage.createFromPath(iconPath);
    
    // Check if icon was loaded successfully
    if (trayIcon.isEmpty()) {
        console.error('Warning: Tray icon could not be loaded from:', iconPath);
    }
    
    // Resize the icon to appropriate size for system tray (22x22 is common for Ubuntu)
    const resizedIcon = trayIcon.resize({ width: 22, height: 22 });
    
    // Create the tray
    tray = new Tray(resizedIcon);
    
    // Set the tooltip
    tray.setToolTip('TinyExplorer Detection App');
    
    // Create context menu
    const contextMenu = Menu.buildFromTemplate([
        {
            label: 'Show App',
            click: () => {
                if (mainWindow) {
                    mainWindow.show();
                    mainWindow.focus();
                }
            }
        },
        {
            label: 'Hide App',
            click: () => {
                if (mainWindow) {
                    mainWindow.hide();
                }
            }
        },
        { type: 'separator' },
        {
            label: 'Quit',
            click: () => {
                (app as any).isQuitting = true;
                app.quit();
            }
        }
    ]);
    
    // Set the context menu
    tray.setContextMenu(contextMenu);
    
    // Handle left click on tray icon
    tray.on('click', () => {
        if (mainWindow) {
            if (mainWindow.isVisible()) {
                mainWindow.hide();
            } else {
                mainWindow.show();
                mainWindow.focus();
            }
        }
    });
}

function createWindow() {
    // Determine icon path with fallback options
    let windowIconPath = isDev 
        ? path.join(__dirname, '..', 'resources', 'icon.png')
        : path.join(process.resourcesPath, 'app', 'resources', 'icon.png');
    
    // Check if icon exists, use fallback if not
    const fs = require('fs');
    if (!fs.existsSync(windowIconPath)) {
        const fallbackIcon = path.join(__dirname, '..', 'graphics', 'icons', 'dragon-512.png');
        if (fs.existsSync(fallbackIcon)) {
            windowIconPath = fallbackIcon;
        }
    }
    
    mainWindow = new BrowserWindow({
        icon: windowIconPath,
        // Start hidden and reveal on 'ready-to-show' so the user never sees a
        // blank white window while the React bundle loads; the splash covers the
        // gap until then.
        show: false,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    // Reveal the main window once its first paint is ready, then dismiss the splash.
    mainWindow.once("ready-to-show", () => {
        logStartupTiming("window_shown");
        if (mainWindow) {
            mainWindow.show();
            mainWindow.focus();
        }
        closeSplash();
    });

    if (isDev) {
        mainWindow.webContents.openDevTools();
    }
    
    if (isDev) {
        mainWindow.loadURL("http://localhost:3000/index.html");
    } else {
        mainWindow.loadURL(`file://${path.join(__dirname, "/../build/index.html")}`);
    }
    
    // Handle window closed
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
    
    // Prevent app from quitting when window is closed (keep in tray)
    mainWindow.on('close', (event) => {
        if (!(app as any).isQuitting) {
            event.preventDefault();
            mainWindow?.hide();
        }
        return false;
    });

    // Handle folder browsing
    ipcMain.on("browse-folder", async (event: any) => {
        const result = await dialog.showOpenDialog(mainWindow!, {
            properties: ["openDirectory"]
        });
        if (!result.canceled && result.filePaths && result.filePaths.length > 0) {
            const p = result.filePaths[0];
            event.sender.send("selected-folder", { path: p, hasVideo: selectionHasVideo(p) });
        } else {
            event.sender.send("selected-folder", null);
        }
    });

    // Handle file browsing
    ipcMain.on("browse-file", async (event: any) => {
        const result = await dialog.showOpenDialog(mainWindow!, {
            properties: ["openFile"],
            filters: [
                { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'] },
                { name: 'Videos', extensions: ['mp4', 'avi', 'mov'] },
                { name: 'Audio', extensions: ['wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'mkv'] },
                { name: 'All Files', extensions: ['*'] }
            ]
        });
        if (!result.canceled && result.filePaths && result.filePaths.length > 0) {
            const p = result.filePaths[0];
            event.sender.send("selected-folder", { path: p, hasVideo: selectionHasVideo(p) });
        } else {
            event.sender.send("selected-folder", null);
        }
    });

    // Handle CSV file saving
    ipcMain.on("save-csv", async (event: any) => {
        const result = await dialog.showSaveDialog(mainWindow!, {
            filters: [
                { name: 'CSV Files', extensions: ['csv'] },
                { name: 'All Files', extensions: ['*'] }
            ],
            defaultPath: 'detection_results.csv'
        });
        if (!result.canceled && result.filePath) {
            event.sender.send("selected-save-path", result.filePath);
        } else {
            event.sender.send("selected-save-path", null);
        }
    });

    // Handle opening folder in system file manager
    ipcMain.on("open-folder", async (event: any, folderPath: string) => {
        try {
            const result = await shell.openPath(folderPath);
            if (result) {
                console.log("Successfully opened folder:", folderPath);
            } else {
                console.error("Failed to open folder:", folderPath);
            }
        } catch (error) {
            console.error("Error opening folder:", error);
        }
    });
}
