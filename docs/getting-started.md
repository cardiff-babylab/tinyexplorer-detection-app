# Getting Started

Welcome to the TinyExplorer DetectionApp user guide. This section helps you install the application and become familiar with the interface.

## Installation

Choose your operating system below for specific installation instructions:

=== "macOS"

    ### macOS Installation Guide
    
    Since TinyExplorer DetectionApp is not signed with an Apple Developer certificate, macOS requires additional steps to ensure the app can run safely on your system.
    
    #### Step 1: Download the Application
    1. Visit the [Releases page](https://github.com/cardiff-babylab/tinyexplorer-detectionapp/releases)
    2. Download the latest `.dmg` file for macOS
    3. Open the downloaded `.dmg` file
    4. Drag **TinyExplorer DetectionApp** to your **Applications** folder
    
    #### Step 2: Remove Quarantine (Required)
    
    !!! info "Why is this necessary?"
        macOS quarantines unsigned applications downloaded from the internet for security. Since we don't have an Apple Developer certificate (which costs $99/year), you need to manually approve the app. This is safe as long as you downloaded it from our official GitHub releases.
    
    1. Open **Terminal** (found in Applications → Utilities)
    2. Type the following command but **don't press Enter yet**:
       ```bash
       sudo xattr -r -d com.apple.quarantine 
       ```
    3. Add a space after the command
    4. **Drag and drop** the TinyExplorer app from your Applications folder into the Terminal window
       - This will automatically add the correct path to your app
       - The complete command should look like:
         ```bash
         sudo xattr -r -d com.apple.quarantine /Applications/TinyExplorer\ FaceDetectionApp.app
         ```
    5. Press **Enter** and type your Mac password when prompted
    
    !!! tip "What does this command do?"
        - `sudo` - Runs the command with administrator privileges
        - `xattr` - Modifies file attributes
        - `-r` - Applies recursively to all files in the app
        - `-d com.apple.quarantine` - Removes the quarantine flag
        - The path points to your installed application
    
    #### Step 3: Launch the Application
    1. Go to your **Applications** folder
    2. Double-click **TinyExplorer DetectionApp**
    3. If you see a security warning, click **Open** to proceed
    4. The app will now launch successfully!
    
    !!! success "Installation Complete!"
        The app is now installed and ready to use. You only need to do this once.

=== "Windows"

    ### Windows Installation Guide

    Since TinyExplorer DetectionApp is not signed with an official developer certificate, Windows may show a security warning during installation. This is normal and safe as long as you downloaded the installer from our official GitHub releases.

    #### Step 1: Download the Installer
    1. Visit the [Releases page](https://github.com/cardiff-babylab/tinyexplorer-detectionapp/releases)
    2. Download the latest `tinyexplorer-detectionapp-x.y.z-setup.exe` file

    #### Step 2: Run the Installer
    1. Double-click the downloaded `.exe` file to launch the installer
    2. You may see a **Windows Defender SmartScreen** warning (blue screen) saying "Windows protected your PC"

    !!! info "Why does this warning appear?"
        This warning appears because the app is not signed with an official Microsoft developer certificate. This is safe as long as you downloaded the installer from our official GitHub releases.

    3. Click **"More info"** on the warning screen
    4. Click **"Run anyway"** to proceed with the installation

    #### Step 3: Complete Installation
    1. The installer will ask where to install the application
        - Default location: `C:\Users\<YourUsername>\AppData\Local\Programs\TinyExplorer DetectionApp`
        - You can keep the default or choose a different location
    2. Click **Install** to begin the installation

    !!! warning "Installation takes time"
        The installation process may take several minutes and the progress bar might appear stuck at times. **Please be patient** — the installer is copying all necessary files including the Python environment. Do not close the installer window.

    3. Once complete, click **Finish** to close the installer

    #### Step 4: Launch the Application
    1. Find **TinyExplorer DetectionApp** in your Start menu or on your Desktop
    2. Double-click to launch the application
    3. The app is now ready to use!

    !!! success "Installation Complete!"
        The app is now installed and ready to use.

=== "Ubuntu/Linux"

    ### Ubuntu/Linux Installation Guide
    
    !!! warning "Coming Soon"
        Linux AppImage/deb package is currently in development. Please check back later or build from source using the instructions below.
    
    #### Build from Source (Temporary Solution)
    1. Install Node.js and npm:
       ```bash
       sudo apt update
       sudo apt install nodejs npm
       ```
    2. Clone the repository:
       ```bash
       git clone https://github.com/cardiff-babylab/tinyexplorer-detectionapp.git
       cd tinyexplorer-detectionapp
       ```
    3. Install dependencies:
       ```bash
       npm install
       ```
    4. Start the application:
       ```bash
       npm run start
       ```

## First Launch

When you first launch the application, you'll see the main interface with all the controls needed for face detection.

## Interface Overview

<div align="center">
  <img src="../assets/screenshots/app-main-interface.png" alt="TinyExplorer DetectionApp Interface" />
  <br>
  <em>Main application interface showing file selection, model options, and confidence threshold controls</em>
</div>

The main interface contains:

- **File/Folder Selection** – choose images or videos for processing
- **Model Selection** – pick the face detection model
- **Confidence Threshold** – adjust detection sensitivity
- **Start** – begin the recognition process
- **Results Display** – view detection outputs

## See Also
- Basic Usage: installation options and local build steps — see [Basic Usage](index.md#basic-usage).
- Supported File Types: list of compatible images and videos — see [Supported File Formats](main-features.md#supported-file-formats).
