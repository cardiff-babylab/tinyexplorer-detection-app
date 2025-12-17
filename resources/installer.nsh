; Include required libraries
!include "LogicLib.nsh"

; Custom logging macro - defined at top level
!macro ConsoleLog Text
  ${If} ${Silent}
    System::Call 'kernel32::GetStdHandle(i -11)i.r0'
    System::Call 'kernel32::WriteConsole(i r0, t "${Text}$\r$\n", i ${NSIS_MAX_STRLEN}, *i .r1, i 0)'
  ${EndIf}
  DetailPrint "${Text}"
!macroend

!define ConsoleLog "!insertmacro ConsoleLog"

!macro customHeader
  ; Console output is enabled through other mechanisms
  ; MUI_VERBOSE is already defined by the system, so we don't redefine it
!macroend

!macro customInit
  ; Log installation start
  ${ConsoleLog} "TinyExplorer DetectionApp installation starting..."
  ${ConsoleLog} "Version: ${VERSION}"
  ${ConsoleLog} "Architecture: x64"
  
  ; Check for silent mode
  ${If} ${Silent}
    ${ConsoleLog} "Running in silent mode with console output"
  ${Else}
    ${ConsoleLog} "Running in GUI mode"
  ${EndIf}
!macroend

!macro customInstall
  ${ConsoleLog} "Installing application files..."
  
  ; Log Python environment setup
  ${ConsoleLog} "Setting up Python environments..."
  ${ConsoleLog} "  - YOLO environment"
  ${ConsoleLog} "  - RetinaFace environment"
  
  ; Log model directory creation
  ${ConsoleLog} "Creating model storage directories..."
  CreateDirectory "$LOCALAPPDATA\TinyExplorerDetection"
  CreateDirectory "$LOCALAPPDATA\TinyExplorerDetection\models"
  
  ${ConsoleLog} "Setting up shortcuts..."
!macroend

!macro customInstallMode
  ; Force per-user installation by default for better permissions
  ${If} ${Silent}
    ${ConsoleLog} "Installation mode: Per-user"
  ${EndIf}
!macroend

!macro customUnInit
  ${ConsoleLog} "TinyExplorer DetectionApp uninstallation starting..."
!macroend

!macro customUnInstall
  ${ConsoleLog} "Removing application files..."
  ${ConsoleLog} "Cleaning up model cache..."
  
  ; Optional: Remove model cache (ask user in GUI mode)
  ${If} ${Silent}
    ${ConsoleLog} "Preserving model cache at $LOCALAPPDATA\TinyExplorerDetection\models"
  ${Else}
    MessageBox MB_YESNO "Do you want to remove downloaded models?$\n$\nThis will delete all cached face detection models." IDYES removeModels IDNO keepModels
    removeModels:
      RMDir /r "$LOCALAPPDATA\TinyExplorerDetection"
      ${ConsoleLog} "Model cache removed"
      Goto done
    keepModels:
      ${ConsoleLog} "Model cache preserved"
    done:
  ${EndIf}
!macroend

; Progress reporting for file extraction
!macro customExtractFiles
  ${ConsoleLog} "Extracting files... This may take a few minutes."
  
  ; Report progress at intervals
  SetDetailsPrint both
  SetDetailsView show
!macroend

; Allow command-line parameters for silent installation
!macro customCommandLineArguments
  ; Support for logging to file
  ${GetParameters} $R0
  ${GetOptions} $R0 "/LOG=" $R1
  ${If} $R1 != ""
    ${ConsoleLog} "Logging to file: $R1"
    LogSet on
    LogText "Installation log for TinyExplorer DetectionApp"
  ${EndIf}
  
  ; Support for custom installation directory in silent mode
  ${GetOptions} $R0 "/D=" $R2
  ${If} $R2 != ""
    ${ConsoleLog} "Custom installation directory: $R2"
    StrCpy $INSTDIR "$R2"
  ${EndIf}
!macroend