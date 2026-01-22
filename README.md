# Dictation Local SenseVoice

This repository contains two things:

1. **ASR Benchmark Suite** — Tools for comparing speech recognition models (SenseVoice, Paraformer, Fun-ASR-Nano, ElevenLabs) with YouTube transcripts as ground truth
2. **Dictation App** — A macOS dictation tool using local ASR models with global hotkey support

---

## Quick Start

```bash
git clone https://github.com/Yan-Yu-Lin/Dictation-Local-SenseVoice.git
cd Dictation-Local-SenseVoice
uv sync
```

---

# Part 1: ASR Benchmark Suite

Benchmark tools for comparing ASR models against YouTube's original transcript (ground truth).

## Models Compared

| Model | Type | Notes |
|-------|------|-------|
| **SenseVoice** | Local | Alibaba DAMO, multilingual, MPS/CUDA/CPU |
| **Paraformer** | Local | Chinese-focused, CPU only (MPS hangs) |
| **Fun-ASR-Nano** | Local | LLM-based, 31 languages + dialects |
| **ElevenLabs Scribe v2** | Cloud API | Requires `ELEVENLABS_API_KEY` in `.env` |

## Running Benchmarks

### Basic Benchmark (Your Own Recordings)

```bash
# Record test audio first
uv run python record_audio.py
# Press Cmd+Option+Control+D to start/stop recording

# Run benchmark on recordings
uv run python benchmarks/basic.py
uv run python benchmarks/basic.py --models sensevoice paraformer
```

Output: `benchmark_results/benchmark_results.md`

### YouTube Benchmark (Pre-extracted Segments)

Compares ASR output against YouTube's original Chinese transcript.

```bash
uv run python benchmarks/youtube.py
uv run python benchmarks/youtube.py --models elevenlabs sensevoice
```

Output: `benchmark_results/youtube_benchmark_results.md`

### Full Video Benchmark

Transcribes an entire YouTube video and compares segment-by-segment.

```bash
# Download the test video first (志祺七七 AI影片討論)
yt-dlp -x --audio-format wav -o "youtube_test/video.wav" "https://www.youtube.com/watch?v=56-dpUWm-sA"
yt-dlp --write-subs --sub-lang zh-Hant --skip-download -o "youtube_test/%(id)s" "https://www.youtube.com/watch?v=56-dpUWm-sA"

# Run full benchmark
uv run python benchmarks/youtube_full.py
```

Output: `benchmark_results/full_benchmark_results.md`

## Metrics

- **CER (Character Error Rate)** — Edit distance / reference length
- **RTF (Real-Time Factor)** — Processing time / audio duration (lower = faster)
- **RAM tracking** — Memory usage per model
- **OpenCC normalization** — Simplified → Traditional Chinese for fair comparison

## Benchmark Results

See `benchmark_results/` for detailed comparison reports and `docs/ASR_RESEARCH.md` for research notes.

---

# Part 2: Dictation App (macOS)

A local/offline dictation tool. Press global hotkey → speak → transcription is pasted to active app.

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **macOS** | Supported | Uses `quickmachotkey` for global hotkeys |
| Linux | Not supported | `quickmachotkey` is macOS-only |
| Windows | Not supported | `quickmachotkey` is macOS-only |

## Usage

```bash
# Traditional Chinese output (default)
uv run python dictation.py --chinese tw

# Simplified Chinese output
uv run python dictation.py --chinese cn

# Use different model
uv run python dictation.py --model paraformer --device cpu
```

**Controls:**
- `Cmd+Option+Control+D` — Start/stop recording
- `Ctrl+C` — Exit the app

## Features

- **Fully offline** — No API keys needed
- **Global hotkey** — Works from any app
- **Auto-paste** — Transcription pasted directly
- **Chinese conversion** — Simplified → Traditional (OpenCC)
- **Sound feedback** — macOS system sounds
- **Apple Silicon optimized** — MPS GPU acceleration

## Model Options

| Model | Command | Notes |
|-------|---------|-------|
| SenseVoice (default) | `--model sensevoice` | Best for multilingual |
| Paraformer | `--model paraformer --device cpu` | Chinese-focused, no MPS |
| Fun-ASR-Nano | `--model fun-asr-nano` | LLM-based |

---

# Project Structure

```
├── dictation.py              # Main dictation app
├── record_audio.py           # Audio recorder for benchmarks
│
├── benchmarks/               # Benchmark scripts
│   ├── basic.py              # Compare models on recordings
│   ├── youtube.py            # Compare against YouTube transcript
│   └── youtube_full.py       # Full video benchmark
│
├── models/                   # Model implementations
│   ├── fun_asr_nano.py       # Fun-ASR-Nano custom model
│   └── ctc.py                # CTC module
│
├── benchmark_results/        # Benchmark output reports
├── recordings/               # Your recorded test audio
├── youtube_test/             # YouTube test video + subtitles
└── docs/                     # Research documentation
```

---

# About the Models

## SenseVoice

Open-source multilingual ASR from **Alibaba DAMO Academy**.

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/FunAudioLLM/SenseVoice |
| Paper | https://arxiv.org/abs/2407.04051 |
| Model | https://modelscope.cn/models/iic/SenseVoiceSmall |

**Supported languages:** Chinese, English, Cantonese, Japanese, Korean

## FunASR Framework

All models run on **FunASR**, Alibaba's speech recognition toolkit.

GitHub: https://github.com/modelscope/FunASR

---

# Troubleshooting

**Model download slow** — First run downloads from ModelScope China servers. Be patient.

**MPS errors** — Model auto-fallbacks to CPU. Or use `--device cpu`.

**Hotkey not working** — Grant Terminal accessibility permissions in System Settings.

**No audio input** — Grant Terminal microphone permissions in System Settings.

---

## License

MIT License — see [LICENSE](LICENSE)
