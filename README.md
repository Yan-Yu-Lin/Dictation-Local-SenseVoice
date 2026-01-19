# Dictation Local SenseVoice

Local/offline dictation app using **SenseVoice** model. Press global hotkey to start/stop recording, transcription is automatically pasted to the active app.

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **macOS** | Supported | Uses `quickmachotkey` for global hotkeys, `pyobjc` for Cocoa integration |
| Linux | Not supported | `quickmachotkey` is macOS-only |
| Windows | Not supported | `quickmachotkey` is macOS-only |

> **Note**: This app uses `quickmachotkey` for global hotkey interception, which is a macOS-specific library that hooks into the Carbon Event Manager. Cross-platform support would require significant refactoring to use platform-specific hotkey libraries.

## Features

- **Fully offline** - No API keys needed, runs locally on your machine
- **Global hotkey** - `Cmd+Option+Control+D` (or Hyper+D) works from any app
- **Auto-paste** - Transcription is pasted directly to active app
- **Chinese conversion** - Simplified → Traditional (OpenCC)
- **Sound feedback** - macOS system sounds for start/stop
- **Apple Silicon optimized** - Uses MPS (Metal Performance Shaders) for GPU acceleration

## Usage

```bash
# Traditional Chinese output (default)
uv run python dictation.py --chinese tw

# Simplified Chinese output
uv run python dictation.py --chinese cn
```

**Controls:**
- `Cmd+Option+Control+D` - Start/stop recording
- `Ctrl+C` - Exit the app

---

## Benchmark Tools

This repo includes tools for benchmarking ASR models.

### Audio Recorder (`record_audio.py`)

Records audio clips for benchmark testing using the same global hotkey as the dictation app.

```bash
uv run python record_audio.py
# Press Cmd+Option+Control+D to start/stop recording
# Files saved to ./recordings/ as 16kHz mono WAV
```

### Basic Benchmark (`benchmark.py`)

Compares multiple ASR models on your recorded audio files:

| Model | Type | Notes |
|-------|------|-------|
| ElevenLabs Scribe v2 | Cloud API | Requires `ELEVENLABS_API_KEY` in `.env` |
| SenseVoice | Local | MPS/CUDA/CPU |
| Paraformer | Local | CPU only (MPS hangs) |
| Fun-ASR-Nano | Local | LLM-based, MPS/CUDA/CPU |

```bash
# Run benchmark on all recordings
uv run python benchmark.py

# Specify folder and output
uv run python benchmark.py --folder ./my_recordings --output results.md

# Run specific models only
uv run python benchmark.py --models sensevoice paraformer
```

Output: `results/benchmark_results.md`

### YouTube Benchmark (`benchmark_youtube.py`)

Compares ASR models against YouTube's original transcript (ground truth). Includes advanced metrics:

- **CER (Character Error Rate)** - Edit distance / reference length
- **RTF (Real-Time Factor)** - Processing time / audio duration
- **RAM tracking** - Memory usage per model
- **OpenCC conversion** - Normalizes Simplified → Traditional Chinese for fair comparison

```bash
# Run with default YouTube test segments
uv run python benchmark_youtube.py

# Specify models
uv run python benchmark_youtube.py --models elevenlabs sensevoice
```

Output: `results/youtube_benchmark_results.md`

#### Creating Test Segments

To benchmark with a new YouTube video:

```bash
# Download audio and subtitles
yt-dlp -x --audio-format wav -o "youtube_test/video.wav" "VIDEO_URL"
yt-dlp --write-subs --sub-lang zh-Hant --skip-download -o "youtube_test/%(id)s" "VIDEO_URL"

# Extract segments (example: 33s-50s)
ffmpeg -i youtube_test/video.wav -ss 00:00:33 -to 00:00:50 -ar 16000 -ac 1 youtube_test/segment_01.wav
```

Then update the segment configs in `benchmark_youtube.py`.

---

## About SenseVoice

### Overview

**SenseVoice** is an open-source speech recognition model developed by **Alibaba DAMO Academy (達摩院)**. It's designed for high-accuracy multilingual speech-to-text with additional capabilities like emotion recognition and audio event detection.

### Links

| Resource | URL |
|----------|-----|
| **GitHub Repository** | https://github.com/FunAudioLLM/SenseVoice |
| **Model on ModelScope** | https://modelscope.cn/models/iic/SenseVoiceSmall |
| **FunASR Framework** | https://github.com/modelscope/FunASR |
| **Paper (arXiv)** | https://arxiv.org/abs/2407.04051 |

### Model Variants

| Model | Size | Status |
|-------|------|--------|
| `iic/SenseVoiceSmall` | ~800MB | **Publicly available** (used in this app) |
| `SenseVoiceLarge` | ~2GB | Mentioned in paper but **NOT publicly released** |

### Supported Languages

- `zh` - Chinese (Mandarin)
- `en` - English
- `yue` - Cantonese (粵語)
- `ja` - Japanese
- `ko` - Korean
- `auto` - Auto-detect (default)

### Additional Features

SenseVoice can also detect:
- **Emotions**: Happy, sad, angry, fearful, disgusted, surprised
- **Audio Events**: Laughter, applause, coughing, crying, etc.

These are included in the raw output but stripped by `rich_transcription_postprocess()`.

### Technical Details

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/SenseVoiceSmall",    # Model ID
    trust_remote_code=True,          # Required for custom model code
    vad_model="fsmn-vad",            # Voice Activity Detection model
    vad_kwargs={"max_single_segment_time": 30000},  # Max segment: 30s
    device="mps",                    # mps/cuda/cpu
    disable_update=True,             # Don't check for updates
)

result = model.generate(
    input="audio.wav",
    cache={},
    language="auto",      # Language detection
    use_itn=True,         # Inverse Text Normalization (numbers, dates)
    batch_size_s=60,      # Batch size in seconds
    merge_vad=True,       # Merge VAD segments
    merge_length_s=15,    # Merge threshold
)
```

### Performance

On Apple Silicon (M3 Pro):
- **RAM usage**: ~1-1.5 GB total
- **RTF (Real-Time Factor)**: ~0.04-0.2x (much faster than real-time)
- **First load**: Downloads model (~800MB), caches to `~/.cache/modelscope/`
- **CPU vs GPU**: Both are fast for short dictation clips. A dedicated GPU (like 5070 Ti) would be overkill.

### Framework: FunASR

SenseVoice runs on **FunASR**, Alibaba's open-source speech recognition toolkit.

```
FunASR (Fundamental ASR)
├── ASR Models (transcription)
│   ├── SenseVoice - Multilingual ASR + emotion + events
│   ├── Paraformer - Chinese-focused ASR
│   └── Whisper - OpenAI Whisper wrapper
├── VAD Models (voice activity detection)
│   └── fsmn-vad - Detects speech vs silence, splits audio
├── Punctuation Models
│   └── ct-punc - Chinese punctuation restoration
└── Other Models (emotion, speaker diarization, etc.)
```

GitHub: https://github.com/modelscope/FunASR

---

## Model Comparison: SenseVoice vs Paraformer

This app supports multiple ASR models. Here's what we found from testing:

### SenseVoice (Default, Recommended)

```bash
uv run python dictation.py                    # Uses SenseVoice
uv run python dictation.py --model sensevoice
```

| Feature | Status |
|---------|--------|
| Chinese ASR | ✅ Excellent |
| English ASR | ✅ Good |
| Japanese/Korean/Cantonese | ✅ Good |
| Punctuation | ✅ Built-in |
| Emotion detection | ✅ Built-in |
| Apple Silicon MPS | ✅ Works |

### Paraformer (Chinese-focused alternative)

```bash
uv run python dictation.py --model paraformer --device cpu
```

| Feature | Status |
|---------|--------|
| Chinese ASR | ✅ Excellent |
| English ASR | ⚠️ Works but not optimized |
| Punctuation | ✅ Via `ct-punc` (Chinese only) |
| Apple Silicon MPS | ❌ **Hangs** (known issue, use `--device cpu`) |

### MPS (Apple Silicon GPU) Compatibility

| Model | MPS Status | Notes |
|-------|------------|-------|
| SenseVoice | ✅ Works | Simple CTC architecture |
| Paraformer | ❌ Hangs | CIF + LSTM ops have PyTorch MPS bugs |
| fsmn-vad | ✅ Works | - |
| ct-punc | ✅ Works | - |

**Root cause**: Paraformer uses CIF predictor with LSTM and complex tensor operations (boolean indexing, cumsum) that have known issues on PyTorch MPS backend. See [PyTorch #145374](https://github.com/pytorch/pytorch/issues/145374), [FunASR #2652](https://github.com/modelscope/FunASR/issues/2652).

**Workaround**: Use `--device cpu` for Paraformer. CPU is fast enough for dictation (RTF ~0.04-0.08).

### Recommendation

For multilingual dictation on Mac, **SenseVoice is the better choice**:
- Works on MPS (GPU acceleration)
- Good English + Chinese support
- Built-in punctuation for all languages
- Emotion/event detection as bonus

---

## Comparison with Other STT Solutions

| Solution | Type | Languages | Offline | Speed | Notes |
|----------|------|-----------|---------|-------|-------|
| **SenseVoice** | Local | zh/en/yue/ja/ko | Yes | Fast | This app |
| Whisper | Local | 99+ | Yes | Medium | OpenAI, larger models |
| faster-whisper | Local | 99+ | Yes | Fast | CTranslate2 optimized |
| ElevenLabs Scribe | Cloud | Many | No | Fast | Requires API key |
| Deepgram | Cloud | Many | No | Fast | Requires API key |

---

## Dependencies & Licenses

| Package | Purpose | License |
|---------|---------|---------|
| `funasr` | FunASR framework (STT) | MIT |
| `torch`, `torchaudio` | PyTorch for model inference | BSD-3-Clause |
| `sounddevice` | Audio recording | MIT |
| `opencc` | Chinese character conversion | Apache-2.0 |
| `quickmachotkey` | Global hotkey interception (macOS) | MIT |
| `pyobjc-*` | macOS Cocoa bindings | MIT |
| `pynput` | Keyboard simulation for paste | LGPL-3.0 |
| `pyperclip` | Clipboard access | BSD-3-Clause |

All dependencies use permissive open-source licenses (MIT, BSD, Apache 2.0) or LGPL (pynput - used as a library, not modified).

---

## Troubleshooting

### Model download slow
First run downloads from ModelScope China servers. Be patient or use a VPN.

### MPS errors
If you get MPS (Metal) errors, the model will auto-fallback to CPU.

### Hotkey not working
Make sure Terminal/iTerm has Accessibility permissions in System Settings.

### No audio input
Check microphone permissions for Terminal in System Settings → Privacy & Security → Microphone.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
