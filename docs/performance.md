# Performance

This page reports measured speech-transcription performance of the bundled
Windows build, to help you pick a backend and model size for your recordings.

All numbers were measured on 2026-09-02 on a clean Windows 11 Pro machine
(Azure `Standard_B4s_v2`: 4 vCPU, 16 GB RAM, Microsoft Defender active),
CPU only, using a 64-second spoken-audio clip. Your hardware will differ,
but the *relative* differences between backends and sizes carry over.

## Transcription speed

Time to transcribe the 64-second clip with warm caches (model weights already
downloaded and loaded once before):

| Model          | Faster Whisper | OpenAI Whisper | WhisperX |
|----------------|----------------|----------------|----------|
| tiny           | 3.5 s          | 8.2 s          | 3.9 s    |
| base           | 6.6 s          | 9.9 s          | 6.0 s    |
| small          | 19.2 s         | 23.1 s         | 15.3 s   |
| medium         | 56.5 s         | 63.3 s         | 42.0 s   |
| large-v2       | 97.9 s         | 114.4 s        | 73.4 s   |
| large-v3       | 97.0 s         | 114.9 s        | 72.1 s   |
| large-v3-turbo | 56.4 s         | 200.2 s *      | 51.1 s   |

!!! warning "Avoid OpenAI Whisper with large-v3-turbo on CPU"
    In our test this combination fell into repeated temperature-fallback
    decoding and produced hallucinated extra text (152 words transcribed
    against ~134 actually spoken) while also being the slowest run in the
    entire matrix. Use Faster Whisper or WhisperX if you want the turbo
    model on a machine without a GPU.

How to read the table:

- The default **small** model runs at roughly 3-4x realtime on every backend
  — an hour of audio takes 15-20 minutes.
- **WhisperX is the fastest backend from medium upwards** thanks to batched
  inference, and keeps even large-v3 close to realtime.
- **OpenAI Whisper is consistently the slowest** — it computes in fp32 on
  CPU and may re-decode segments with temperature fallbacks, so its times
  also vary more between runs.

## Model loading

Each processing run first loads the selected model:

| Backend        | Typical warm load time |
|----------------|------------------------|
| Faster Whisper | 1-10 s (size-dependent) |
| OpenAI Whisper | 2-20 s (size-dependent) |
| WhisperX       | 9-20 s (also initialises a voice-activity pipeline) |

## First use of a model (cold start)

The first time you select a model size, its weights are downloaded and caches
are built. On a fast connection this takes:

| Model size     | First-use overhead |
|----------------|--------------------|
| tiny / base    | a few seconds      |
| small          | ~10 s              |
| medium         | 30-60 s            |
| large variants | 1-2.5 min (~3 GB download each) |

WhisperX additionally performs a one-off pipeline initialisation of about
100 s on its very first run.

!!! info "The app is not stuck"
    During a cold start the progress display can sit still while weights
    download in the background — this is the "first use may download model
    weights" stage, not a hang. On slow connections the large models can
    take considerably longer than the times above.

## Requirements on Windows

On a freshly installed Windows the app requires the Microsoft Visual C++
Redistributable (x64); without it the detection and speech backends fail to
start. Installers built after 2026-09-01 detect and install it automatically.
For older builds, install it manually from
[aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe).

## Methodology

The benchmark harness lives in the repository at
[`scripts/azure-wintest-bench.ps1`](https://github.com/cardiff-babylab/tinyexplorer-detection-app/blob/main/scripts/azure-wintest-bench.ps1).
It generates a ~64 s spoken clip with Windows text-to-speech (16 kHz mono),
then for each backend and model size runs one cold pass (fresh process,
weight download, first Defender scan) and one measured warm pass with word
timestamps enabled. Word counts across all runs were compared against the
known text to confirm real transcription took place.
