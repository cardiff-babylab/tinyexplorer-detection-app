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

// Disable GPU acceleration for better compatibility with remote displays and AppImages
app.disableHardwareAcceleration();

// Mitigate Windows cache permission issues when running from protected folders (e.g., OneDrive)
if (process.platform === 'win32') {
    const os = require('os');
    const userData = path.join(os.homedir(), 'AppData', 'Roaming', 'TinyExplorer DetectionApp');
    app.setPath('userData', userData);
    app.setPath('cache', path.join(userData, 'Cache'));
}

const isDev = (process.env.NODE_ENV === "development");
let tray: Tray | null = null;
let mainWindow: BrowserWindow | null = null;

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
    if (isDev) {
        const sourceMapSupport = require("source-map-support"); // tslint:disable-line
        sourceMapSupport.install();
    }
    
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
    tray.setToolTip('TinyExplorer DetectionApp');
    
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
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
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
            event.sender.send("selected-folder", result.filePaths[0]);
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
                { name: 'All Files', extensions: ['*'] }
            ]
        });
        if (!result.canceled && result.filePaths && result.filePaths.length > 0) {
            event.sender.send("selected-folder", result.filePaths[0]);
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
            defaultPath: 'face_detection_results.csv'
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
