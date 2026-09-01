# Runs ON the Azure Windows VM via `az vm run-command invoke` (as SYSTEM).
# Downloads the released installer, installs silently to C:\TinyExplorer, then
# probes the bundled speech env for the Windows-only failures diagnosed on
# 2026-09-01: the ctranslate2+torch duplicate-OpenMP abort (kills WhisperX)
# and baseline CPU transcription speed (OpenAI Whisper vs Faster Whisper).

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Version = "0.3.0"
$Url = "https://github.com/cardiff-babylab/tinyexplorer-detection-app/releases/download/v$Version/tinyexplorer-detection-app-$Version-setup.exe"
$Setup = "$env:TEMP\tinyexplorer-setup.exe"
$InstallDir = "C:\TinyExplorer"
$Py = "$InstallDir\resources\pythondist\yolo-env\python.exe"

Write-Output "=== 1/5 Download installer ==="
if (-not (Test-Path $Py)) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Invoke-WebRequest -Uri $Url -OutFile $Setup -UseBasicParsing
    Write-Output ("downloaded {0:N0} MB in {1:N0}s" -f ((Get-Item $Setup).Length / 1MB), $sw.Elapsed.TotalSeconds)

    Write-Output "=== 2/5 Silent install ==="
    # NSIS: /S silent, /D=<dir> must be the last argument.
    Start-Process -FilePath $Setup -ArgumentList "/S", "/D=$InstallDir" -Wait
} else {
    Write-Output "already installed, skipping download/install"
}
if (-not (Test-Path $Py)) { Write-Output "FATAL: $Py not found after install"; exit 1 }
& $Py --version

Write-Output "=== 3/5 OMP duplicate-runtime probe (expected to die with OMP Error #15) ==="
# Run in a child python so an abort() doesn't kill this script.
$out = & $Py -c "import ctranslate2, torch; print('IMPORT_OK torch', torch.__version__)" 2>&1
Write-Output "exit=$LASTEXITCODE"
Write-Output ($out | Out-String)

Write-Output "=== 4/5 Same probe with KMP_DUPLICATE_LIB_OK=TRUE (proposed fix) ==="
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$sw = [Diagnostics.Stopwatch]::StartNew()
$out = & $Py -c "import ctranslate2, torch; import whisperx; print('WHISPERX_IMPORT_OK')" 2>&1
Write-Output ("exit=$LASTEXITCODE in {0:N0}s (cold import incl. Defender scan)" -f $sw.Elapsed.TotalSeconds)
Write-Output ($out | Out-String)

Write-Output "=== 5/5 CPU speed calibration: tiny model on 20s synthetic audio ==="
# In-memory audio (the app decodes via PyAV to a float32 array, so this matches
# the app's code path and avoids needing an ffmpeg binary).
$bench = @"
import time, numpy as np
sr = 16000
t = np.linspace(0, 20, 20 * sr, dtype=np.float32)
audio = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

t0 = time.time()
from faster_whisper import WhisperModel
m = WhisperModel('tiny', device='cpu', compute_type='int8')
segs, info = m.transcribe(audio, word_timestamps=True)
list(segs)
print('faster-whisper tiny: %.1fs' % (time.time() - t0))

t0 = time.time()
import whisper
m2 = whisper.load_model('tiny')
m2.transcribe(audio, word_timestamps=True, fp16=False)
print('openai-whisper tiny: %.1fs' % (time.time() - t0))
"@
$out = & $Py -c $bench 2>&1
Write-Output "exit=$LASTEXITCODE"
Write-Output ($out | Out-String)
Write-Output "=== done ==="
