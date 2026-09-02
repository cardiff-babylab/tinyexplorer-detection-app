# Speech-backend benchmark for the Azure Windows test VM (see azure-wintest.sh).
# Run via: az vm run-command invoke ... --scripts "@scripts/azure-wintest-bench.ps1" \
#   --parameters "backend=faster|openai|whisperx" "sizes=tiny base small ..."
# Generates ~64 s of 16 kHz speech with Windows TTS, then for each size does a
# COLD pass (fresh process, weight download, first Defender scan) and a warm
# measured pass. Results append to C:\bench\results-all.txt.

param([string]$backend = "faster", [string]$sizes = "tiny")
$ProgressPreference = "SilentlyContinue"
$Py = "C:\TinyExplorer\resources\pythondist\yolo-env\python.exe"
New-Item -ItemType Directory -Path "C:\bench" -Force | Out-Null

# ~62 s of real speech via Windows TTS (16 kHz mono PCM)
if (-not (Test-Path "C:\bench\speech60.wav")) {
    Add-Type -AssemblyName System.Speech
    $syn = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    $syn.SetOutputToWaveFile("C:\bench\speech60.wav", $fmt)
    $text = "The quick brown fox jumps over the lazy dog while researchers watch from the observation room. " +
            "In developmental psychology, careful measurement of infant attention provides insight into early learning. " +
            "The laboratory equipment records both audio and video during each experimental session. " +
            "Participants arrive in the morning and are greeted by trained research assistants. " +
            "Every recording is transcribed and annotated before the statistical analysis begins. " +
            "The detection application processes faces, hands, and speech from the recorded sessions. " +
            "Automatic transcription saves many hours of manual work for the research team. " +
            "Timestamps allow researchers to align spoken words with visual attention measures. " +
            "The final dataset combines all modalities into a single comprehensive table. " +
            "Reliable software tools are essential for reproducible science in modern laboratories. " +
            "Careful benchmarking reveals how each model size trades accuracy against processing speed."
    $syn.Speak($text)
    $syn.Dispose()
}
$wav = Get-Item "C:\bench\speech60.wav"
Write-Output ("speech60.wav: {0:N0} KB" -f ($wav.Length / 1KB))

$pycode = @"
import sys, time, wave
import numpy as np
backend, size, wavpath, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
w = wave.open(wavpath, 'rb')
sr = w.getframerate()
audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
dur = len(audio) / sr
t0 = time.time()
if backend == 'faster':
    from faster_whisper import WhisperModel
    m = WhisperModel(size, device='cpu', compute_type='int8')
    load = time.time() - t0
    if mode == 'bench':
        t1 = time.time()
        segs, _ = m.transcribe(audio, word_timestamps=True)
        txt = ' '.join(s.text for s in segs)
        tr = time.time() - t1
elif backend == 'openai':
    import whisper
    m = whisper.load_model(size)
    load = time.time() - t0
    if mode == 'bench':
        t1 = time.time()
        r = m.transcribe(audio, word_timestamps=True, fp16=False)
        txt = r['text']
        tr = time.time() - t1
else:
    import whisperx
    m = whisperx.load_model(size, device='cpu', compute_type='int8')
    load = time.time() - t0
    if mode == 'bench':
        t1 = time.time()
        r = m.transcribe(audio, batch_size=8)
        txt = ' '.join(s['text'] for s in r['segments'])
        tr = time.time() - t1
if mode == 'bench':
    print('RESULT %s %s dur=%.0fs load=%.1f transcribe=%.1f words=%d' % (backend, size, dur, load, tr, len(txt.split())))
else:
    print('WARM %s %s load=%.1f' % (backend, size, load))
"@
Set-Content -Path "C:\bench\bench.py" -Value $pycode

foreach ($size in $sizes.Split(" ")) {
    # pass 1 = COLD START: fresh process, downloads weights, first Defender
    # scan, cold disk cache. pass 2 = warm, the measured steady-state run.
    $cold = & $Py "C:\bench\bench.py" $backend $size "C:\bench\speech60.wav" warm 2>$null
    $coldline = ($cold | Select-String "^WARM").Line
    if (-not $coldline) { $coldline = "WARM $backend $size FAILED" }
    $coldline = $coldline -replace "^WARM", "COLD"
    Write-Output $coldline
    Add-Content -Path "C:\bench\results-all.txt" -Value $coldline
    $out = & $Py "C:\bench\bench.py" $backend $size "C:\bench\speech60.wav" bench 2>$null
    $line = ($out | Select-String "^RESULT").Line
    if (-not $line) { $line = "RESULT $backend $size FAILED" }
    Write-Output $line
    Add-Content -Path "C:\bench\results-all.txt" -Value $line
}
Write-Output "=== batch done: $backend [$sizes] ==="
