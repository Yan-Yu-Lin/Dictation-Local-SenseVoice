# ASR Models & Frameworks Research Report

This document provides a comprehensive overview of the ASR models and frameworks used in this dictation app.

---

## Table of Contents

1. [FunASR Framework](#1-funasr-framework)
2. [SenseVoice (Default Model)](#2-sensevoice-default-model)
3. [Paraformer (Chinese-Focused Alternative)](#3-paraformer-chinese-focused-alternative)
4. [Fun-ASR-Nano (LLM-Based ASR)](#4-fun-asr-nano-llm-based-asr)
5. [FSMN-VAD (Voice Activity Detection)](#5-fsmn-vad-voice-activity-detection)
6. [CT-Punc (Punctuation Restoration)](#6-ct-punc-punctuation-restoration)
7. [How They All Work Together](#7-how-they-all-work-together)
8. [Quick Comparison Table](#8-quick-comparison-table)
9. [Recommended Reading](#9-recommended-reading)

---

## 1. FunASR Framework

**What it is:** FunASR is an open-source, industrial-grade speech recognition toolkit developed by **Alibaba DAMO Academy (达摩院)**. It bridges the gap between academic research and industrial applications.

**Key Features:**
- Speech Recognition (ASR)
- Voice Activity Detection (VAD)
- Punctuation Restoration
- Speaker Verification & Diarization
- Multi-talker ASR
- Unified API via `AutoModel` for loading any model

**Architecture:**
FunASR uses an end-to-end deep learning framework supporting various neural network architectures. The `AutoModel` class provides a unified interface to load different models (SenseVoice, Paraformer, Fun-ASR-Nano, etc.) with consistent `generate()` API.

### Links

| Resource | URL |
|----------|-----|
| **GitHub** | https://github.com/modelscope/FunASR |
| **Documentation** | https://www.funasr.com |
| **PyPI** | https://pypi.org/project/funasr/ |
| **arXiv Paper (2023)** | https://arxiv.org/abs/2305.11013 |
| **ModelScope Models** | https://modelscope.cn/organization/iic |
| **HuggingFace Models** | https://huggingface.co/funasr |

---

## 2. SenseVoice (Default Model)

**What it is:** SenseVoice is a **speech foundation model** with multiple understanding capabilities beyond just ASR. It's part of the **FunAudioLLM** family (alongside CosyVoice for speech generation).

### Capabilities

- **ASR** - Automatic Speech Recognition
- **LID** - Language Identification
- **SER** - Speech Emotion Recognition (happy, sad, angry, fearful, disgusted, surprised)
- **AED** - Audio Event Detection (laughter, applause, coughing, crying, BGM, etc.)

### Architecture

- **Non-autoregressive end-to-end** framework
- Uses **CTC-based** architecture (simple, MPS-compatible)
- **234M parameters** (SenseVoice-Small)
- **Exceptionally low latency**: 70ms for 10 seconds of audio (15x faster than Whisper-Large)

### Languages Supported

| Code | Language |
|------|----------|
| `zh` | Chinese (Mandarin) |
| `en` | English |
| `yue` | Cantonese (粵語) |
| `ja` | Japanese |
| `ko` | Korean |
| `auto` | Auto-detect |

### Performance Highlights

- Trained on **400,000+ hours** of data
- Supports **50+ languages** (SenseVoice-Large)
- Better Chinese/Cantonese recognition than Whisper
- Emotion recognition surpasses current best models

### Why it works on MPS (Apple Silicon)

SenseVoice uses a simple CTC architecture without complex operations like LSTM or CIF that have PyTorch MPS bugs.

### Links

| Resource | URL |
|----------|-----|
| **GitHub** | https://github.com/FunAudioLLM/SenseVoice |
| **arXiv Paper** | https://arxiv.org/abs/2407.04051 |
| **ModelScope** | https://modelscope.cn/models/iic/SenseVoiceSmall |
| **HuggingFace** | https://huggingface.co/FunAudioLLM/SenseVoiceSmall |
| **FunAudioLLM Homepage** | https://funaudiollm.github.io/ |

---

## 3. Paraformer (Chinese-Focused Alternative)

**What it is:** Paraformer is a **non-autoregressive (NAR)** transformer for ASR that achieves **10x+ speedup** over traditional autoregressive models while maintaining comparable accuracy.

### The Innovation - CIF (Continuous Integrate-and-Fire)

Traditional ASR models generate tokens one-by-one (autoregressive), which is slow. Paraformer uses a **CIF predictor** that:

1. Predicts the number of output tokens in advance
2. Generates hidden variables in parallel
3. Uses a **Glancing Language Model (GLM) sampler** to enhance context modeling

This allows parallel generation instead of sequential.

### Architecture (Paraformer v1)

```
Audio Input → Conformer Encoder → CIF Predictor → NAR Decoder → Text Output
                                       ↓
                              (predicts token count +
                               extracts hidden variables)
```

### Paraformer-v2 Improvements

- Replaces CIF with **CTC module** for token embedding extraction
- **14%+ better WER** on English datasets
- More robust in noisy environments

### Why it hangs on MPS

The CIF predictor uses LSTM and complex tensor operations (boolean indexing, cumsum) that have known PyTorch MPS bugs. See [PyTorch #145374](https://github.com/pytorch/pytorch/issues/145374).

**Workaround:** Use `--device cpu` (CPU is fast enough, RTF ~0.04-0.08)

### Model Size

~220M parameters

### Links

| Resource | URL |
|----------|-----|
| **Paraformer Paper (2022)** | https://arxiv.org/abs/2206.08317 |
| **Paraformer-v2 Paper (2024)** | https://arxiv.org/abs/2409.17746 |
| **CIF Predictor Code** | https://github.com/modelscope/FunASR/blob/main/funasr/models/paraformer/cif_predictor.py |
| **ModelScope** | https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch |

---

## 4. Fun-ASR-Nano (LLM-Based ASR)

**What it is:** Fun-ASR-Nano is the newest generation ASR from Tongyi Lab. It's an **LLM-based** end-to-end model trained on **tens of millions of hours** of real speech data.

### Architecture

```
Audio Input → Audio Encoder (0.2B params) → LLM Decoder (0.6B params) → Text Output
                    ↓
            (Transformer layers)
```

The full Fun-ASR model has:
- Audio encoder: 0.7B parameters
- LLM decoder: 7B parameters

Fun-ASR-Nano is the lighter version (~800M total params).

### Key Innovations

- **LLM integration** for powerful contextual understanding
- **Reinforcement learning** to reduce hallucinations (a common LLM problem)
- **Hotword customization** support
- **Code-switching** (mixing languages naturally)

### Languages Supported (31 total)

- Chinese (with **7 dialects**: Wu, Cantonese, Min, Hakka, Gan, Xiang, Jin)
- Chinese with **26 regional accents** (Henan, Sichuan, Guangdong, etc.)
- English, Japanese with regional accents
- Plus many European and Asian languages in the MLT variant

### Special Capabilities

- **Lyric recognition** (songs with music background)
- **Rap speech recognition**
- **Far-field high-noise** scenarios (93% accuracy)

### Links

| Resource | URL |
|----------|-----|
| **GitHub** | https://github.com/FunAudioLLM/Fun-ASR |
| **Technical Report (arXiv)** | https://arxiv.org/abs/2509.12508 |
| **ModelScope** | https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512 |
| **HuggingFace** | https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512 |
| **Homepage** | https://funaudiollm.github.io/funasr/ |

---

## 5. FSMN-VAD (Voice Activity Detection)

**What it is:** FSMN-VAD is an enterprise-grade Voice Activity Detection model that identifies speech vs. silence in audio.

### What FSMN means

- **FSMN** = Feedforward Sequential Memory Network
- A type of neural network that can model sequential data efficiently (like RNNs but faster)

### Purpose in This App

VAD splits long audio into shorter segments before ASR processing:

```
Long Audio → FSMN-VAD → [Segment 1] [Segment 2] [Segment 3] → ASR Model
                              ↓
                    (detects speech start/end times)
```

### Performance

- ~0.4M parameters (very lightweight)
- 70s audio processed in <0.6s on M1 Pro
- RTF of 0.0077 with ONNX runtime
- Trained on 5,000 hours of Mandarin and English

### Configuration

```python
vad_model="fsmn-vad",
vad_kwargs={"max_single_segment_time": 30000}  # 30 seconds max per segment
```

### Links

| Resource | URL |
|----------|-----|
| **HuggingFace** | https://huggingface.co/funasr/fsmn-vad |
| **ModelScope** | https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch |
| **ONNX Version** | https://huggingface.co/funasr/fsmn-vad-onnx |
| **Standalone Repo** | https://github.com/lovemefan/fsmn-vad |

---

## 6. CT-Punc (Punctuation Restoration)

**What it is:** CT-Punc is a punctuation restoration model that adds proper punctuation to ASR output (which typically lacks punctuation).

### What CT means

- **CT** = Controllable Transformer
- Uses transformer architecture for punctuation prediction

### Purpose in This App

Used with Paraformer (which doesn't have built-in punctuation):

```python
punc_model="ct-punc"  # Only needed for Paraformer
```

SenseVoice has punctuation built-in, so it doesn't need this.

### Capabilities

- ~290M parameters
- Supports Chinese and English
- Trained on 100M text samples

### Links

| Resource | URL |
|----------|-----|
| **HuggingFace** | https://huggingface.co/funasr/ct-punc |
| **ModelScope** | https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch |

---

## 7. How They All Work Together

Here's the pipeline in this dictation app:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Audio Input                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FSMN-VAD                                   │
│              (Splits audio into speech segments)                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ASR Model                                   │
│  ┌──────────────┬──────────────┬──────────────────┐             │
│  │ SenseVoice   │  Paraformer  │   Fun-ASR-Nano   │             │
│  │ (CTC-based)  │ (CIF-based)  │   (LLM-based)    │             │
│  │ MPS ✅       │ CPU only ❌   │   MPS ✅         │             │
│  │ Punc built-in│ Needs ct-punc│   Punc built-in  │             │
│  └──────────────┴──────────────┴──────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Post-Processing                               │
│  - rich_transcription_postprocess() strips emotion tags          │
│  - OpenCC converts Simplified → Traditional Chinese              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Final Text Output                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Quick Comparison Table

| Model | Type | Params | MPS | Punctuation | Best For |
|-------|------|--------|-----|-------------|----------|
| **SenseVoice** | CTC (NAR) | 234M | ✅ | Built-in | Multilingual, emotion detection |
| **Paraformer** | CIF (NAR) | 220M | ❌ | Needs ct-punc | Chinese-focused, fast inference |
| **Fun-ASR-Nano** | LLM-based | 800M | ✅ | Built-in | Dialects, accents, noisy environments |

---

## 9. Recommended Reading

If you want to dive deeper, here's the suggested reading order:

1. **FunAudioLLM paper** - Overview of SenseVoice + CosyVoice
   - https://arxiv.org/abs/2407.04051

2. **Paraformer paper** - Understand CIF mechanism and NAR ASR
   - https://arxiv.org/abs/2206.08317

3. **Fun-ASR Technical Report** - Latest LLM-based ASR approach
   - https://arxiv.org/abs/2509.12508

4. **FunASR toolkit paper** - Framework architecture
   - https://arxiv.org/abs/2305.11013

---

## All Links Summary

### GitHub Repositories

| Project | URL |
|---------|-----|
| FunASR | https://github.com/modelscope/FunASR |
| SenseVoice | https://github.com/FunAudioLLM/SenseVoice |
| Fun-ASR | https://github.com/FunAudioLLM/Fun-ASR |
| FSMN-VAD (standalone) | https://github.com/lovemefan/fsmn-vad |

### arXiv Papers

| Paper | URL |
|-------|-----|
| FunAudioLLM (SenseVoice + CosyVoice) | https://arxiv.org/abs/2407.04051 |
| Paraformer (original) | https://arxiv.org/abs/2206.08317 |
| Paraformer-v2 | https://arxiv.org/abs/2409.17746 |
| Fun-ASR Technical Report | https://arxiv.org/abs/2509.12508 |
| FunASR Toolkit | https://arxiv.org/abs/2305.11013 |

### ModelScope

| Model | URL |
|-------|-----|
| SenseVoiceSmall | https://modelscope.cn/models/iic/SenseVoiceSmall |
| Paraformer-large | https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch |
| Fun-ASR-Nano | https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512 |
| FSMN-VAD | https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch |
| CT-Punc | https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch |

### HuggingFace

| Model | URL |
|-------|-----|
| SenseVoiceSmall | https://huggingface.co/FunAudioLLM/SenseVoiceSmall |
| Fun-ASR-Nano | https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512 |
| FSMN-VAD | https://huggingface.co/funasr/fsmn-vad |
| CT-Punc | https://huggingface.co/funasr/ct-punc |

### Other Resources

| Resource | URL |
|----------|-----|
| FunASR Documentation | https://www.funasr.com |
| FunAudioLLM Homepage | https://funaudiollm.github.io/ |
| Fun-ASR Homepage | https://funaudiollm.github.io/funasr/ |
| FunASR PyPI | https://pypi.org/project/funasr/ |

---

*Generated: 2026-01-20*
