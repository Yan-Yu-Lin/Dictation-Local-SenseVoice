#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "funasr",
#     "torch",
#     "numpy",
#     "transformers",
#     "elevenlabs",
#     "pydub",
#     "opencc-python-reimplemented",
#     "python-dotenv",
#     "psutil",
# ]
# ///
"""
YouTube ASR Benchmark v2 - Compare ASR models against YouTube's original transcript.

Features:
- YouTube video metadata (URL, title)
- RAM/GPU memory tracking
- Real-time factor (RTF) calculation
- OpenCC conversion (normalize to Traditional Chinese)
- Character Error Rate (CER) calculation
- Summary comparison table
- Clean output (strip emoji artifacts)
"""

import argparse
import io
import os
import re
import time
import wave
import traceback
from datetime import datetime
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

import torch
import psutil
import opencc
from pydub import AudioSegment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# OpenCC converter for Simplified -> Traditional Chinese
S2T_CONVERTER = opencc.OpenCC('s2t')


# =============================================================================
# Utility Functions
# =============================================================================

def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def clean_transcript(text: str) -> str:
    """Clean transcript text - remove emoji artifacts, normalize whitespace."""
    # Remove common SenseVoice emoji artifacts
    text = re.sub(r'[🎼😊🎵🎶👏😄😢😠]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def convert_to_traditional(text: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese."""
    return S2T_CONVERTER.convert(text)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER) using Levenshtein distance."""
    # Remove spaces for character-level comparison
    ref = reference.replace(' ', '')
    hyp = hypothesis.replace(' ', '')

    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0

    # Levenshtein distance
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    return dp[m][n] / m


@dataclass
class SRTEntry:
    index: int
    start_ms: int
    end_ms: int
    text: str


def parse_timestamp(ts: str) -> int:
    """Convert SRT timestamp to milliseconds."""
    match = re.match(r'(\d+):(\d+):(\d+),(\d+)', ts)
    if not match:
        return 0
    h, m, s, ms = map(int, match.groups())
    return h * 3600000 + m * 60000 + s * 1000 + ms


def parse_srt(srt_path: str) -> list[SRTEntry]:
    """Parse SRT file into entries."""
    entries = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                ts_match = re.match(r'(.+) --> (.+)', lines[1])
                if ts_match:
                    start_ms = parse_timestamp(ts_match.group(1))
                    end_ms = parse_timestamp(ts_match.group(2))
                    text = ' '.join(lines[2:])
                    entries.append(SRTEntry(index, start_ms, end_ms, text))
            except (ValueError, IndexError):
                continue

    return entries


def get_transcript_for_range(entries: list[SRTEntry], start_ms: int, end_ms: int) -> str:
    """Get combined transcript text for a time range."""
    texts = []
    for entry in entries:
        if entry.end_ms > start_ms and entry.start_ms < end_ms:
            texts.append(entry.text)
    return ' '.join(texts)


def get_audio_duration(wav_path: str) -> float:
    """Get duration of a WAV file in seconds."""
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate


def get_audio_info(wav_path: str) -> dict:
    """Get audio file information."""
    with wave.open(wav_path, 'rb') as wf:
        return {
            "duration": wf.getnframes() / wf.getframerate(),
            "sample_rate": wf.getframerate(),
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth() * 8,  # bits
        }


def convert_wav_to_mp3(wav_path: str) -> bytes:
    """Convert WAV file to MP3 bytes for API upload."""
    audio = AudioSegment.from_wav(wav_path)
    buffer = io.BytesIO()
    audio.export(buffer, format="mp3")
    return buffer.getvalue()


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# =============================================================================
# Model Runners
# =============================================================================

@dataclass
class TranscriptionResult:
    text: str
    text_clean: str  # Cleaned version
    text_traditional: str  # Converted to Traditional Chinese
    inference_time: float
    load_time: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    error: str | None = None


class BaseRunner:
    name: str = "Base"
    device: str = "Unknown"
    model_info: str = ""

    def transcribe(self, wav_path: str) -> str:
        raise NotImplementedError


class ElevenLabsRunner(BaseRunner):
    def __init__(self):
        from elevenlabs import ElevenLabs as ElevenLabsClient
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")
        self.client = ElevenLabsClient(api_key=api_key)
        self.name = "ElevenLabs Scribe v2"
        self.device = "Cloud API"
        self.model_info = "Cloud-based, paid API"

    def transcribe(self, wav_path: str) -> str:
        mp3_bytes = convert_wav_to_mp3(wav_path)
        audio_file = io.BytesIO(mp3_bytes)
        audio_file.name = "audio.mp3"
        result = self.client.speech_to_text.convert(
            file=audio_file,
            model_id="scribe_v2",
            language_code=None,
        )
        return result.text


class SenseVoiceRunner(BaseRunner):
    def __init__(self, device: str):
        from funasr import AutoModel
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            device=device,
            disable_update=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
        )
        self.name = "SenseVoice"
        self.device = device.upper()
        self.model_info = "~234M params, multilingual"

    def transcribe(self, wav_path: str) -> str:
        torch.set_num_threads(4)
        res = self.model.generate(
            input=wav_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        torch.set_num_threads(4)
        if res and res[0].get("text"):
            text = res[0]["text"]
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                return rich_transcription_postprocess(text)
            except Exception:
                return text
        return ""


class ParaformerRunner(BaseRunner):
    def __init__(self):
        from funasr import AutoModel
        self.model = AutoModel(
            model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            device="cpu",
            disable_update=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            punc_model="ct-punc",
        )
        self.name = "Paraformer"
        self.device = "CPU"
        self.model_info = "~220M params, Chinese-focused"

    def transcribe(self, wav_path: str) -> str:
        torch.set_num_threads(4)
        res = self.model.generate(
            input=wav_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        torch.set_num_threads(4)
        if res and res[0].get("text"):
            return res[0]["text"]
        return ""


class FunASRNanoRunner(BaseRunner):
    def __init__(self, device: str):
        from funasr import AutoModel
        remote_code_path = os.path.join(PROJECT_ROOT, "models", "fun_asr_nano.py")
        self.model = AutoModel(
            model="FunAudioLLM/Fun-ASR-Nano-2512",
            trust_remote_code=True,
            remote_code=remote_code_path,
            device=device,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
        )
        self.name = "Fun-ASR-Nano"
        self.device = device.upper()
        self.model_info = "~800M params, LLM-based"

    def transcribe(self, wav_path: str) -> str:
        torch.set_num_threads(4)
        res = self.model.generate(
            input=[wav_path],
            cache={},
            batch_size=1,
            language="auto",
            itn=True,
        )
        torch.set_num_threads(4)
        if res and res[0].get("text"):
            return res[0]["text"]
        return ""


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class SegmentConfig:
    wav_path: str
    start_ms: int
    end_ms: int
    name: str


@dataclass
class BenchmarkConfig:
    youtube_url: str = ""
    youtube_title: str = ""
    srt_path: str = ""
    output_file: str = ""


@dataclass
class SegmentResult:
    name: str
    wav_path: str
    duration: float
    start_ms: int
    end_ms: int
    audio_info: dict
    ground_truth: str
    ground_truth_clean: str
    transcriptions: list = field(default_factory=list)


def run_transcription(runner: BaseRunner, wav_path: str) -> TranscriptionResult:
    """Run transcription and collect metrics."""
    memory_before = get_memory_mb()
    memory_peak = memory_before

    start_time = time.time()
    error = None
    text = ""

    try:
        text = runner.transcribe(wav_path)
        memory_peak = max(memory_peak, get_memory_mb())
    except Exception as e:
        error = str(e)
        traceback.print_exc()

    inference_time = time.time() - start_time
    memory_after = get_memory_mb()

    # Clean and convert text
    text_clean = clean_transcript(text)
    text_traditional = convert_to_traditional(text_clean)

    return TranscriptionResult(
        text=text,
        text_clean=text_clean,
        text_traditional=text_traditional,
        inference_time=inference_time,
        load_time=0,  # Already loaded
        memory_before_mb=memory_before,
        memory_after_mb=memory_after,
        memory_peak_mb=memory_peak,
        error=error,
    )


def run_youtube_benchmark(
    config: BenchmarkConfig,
    segments: list[SegmentConfig],
    models: list[str] | None = None
):
    """Run benchmark comparing ASR models with YouTube transcript."""

    # Parse SRT
    print("Parsing SRT file...")
    srt_entries = parse_srt(config.srt_path)
    print(f"Found {len(srt_entries)} subtitle entries")

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        device_desc = "Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        device = "cuda:0"
        device_desc = "NVIDIA GPU (CUDA)"
    else:
        device = "cpu"
        device_desc = "CPU"
    print(f"Device: {device_desc}")

    # Initialize models and track load time/memory
    print("\nLoading models...")
    runners: list[tuple[BaseRunner, float, float]] = []  # (runner, load_time, memory_delta)

    if models is None:
        models = ["elevenlabs", "sensevoice", "paraformer", "fun-asr-nano"]

    for model_name in models:
        try:
            mem_before = get_memory_mb()
            load_start = time.time()

            if model_name == "elevenlabs":
                print("  Loading ElevenLabs...")
                runner = ElevenLabsRunner()
            elif model_name == "sensevoice":
                print("  Loading SenseVoice...")
                runner = SenseVoiceRunner(device)
            elif model_name == "paraformer":
                print("  Loading Paraformer...")
                runner = ParaformerRunner()
            elif model_name == "fun-asr-nano":
                print("  Loading Fun-ASR-Nano...")
                runner = FunASRNanoRunner(device)
            else:
                continue

            load_time = time.time() - load_start
            mem_delta = get_memory_mb() - mem_before
            runners.append((runner, load_time, mem_delta))
            print(f"    Loaded in {load_time:.1f}s, +{mem_delta:.0f}MB RAM")

        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")

    print(f"\nLoaded {len(runners)} model(s)")

    # Run benchmark
    results: list[SegmentResult] = []

    for seg in segments:
        audio_info = get_audio_info(seg.wav_path)
        ground_truth = get_transcript_for_range(srt_entries, seg.start_ms, seg.end_ms)
        ground_truth_clean = clean_transcript(ground_truth)

        print(f"\nProcessing: {seg.name} ({audio_info['duration']:.1f}s)")
        print(f"  Ground truth: {ground_truth_clean[:60]}...")

        seg_result = SegmentResult(
            name=seg.name,
            wav_path=seg.wav_path,
            duration=audio_info['duration'],
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            audio_info=audio_info,
            ground_truth=ground_truth,
            ground_truth_clean=ground_truth_clean,
        )

        for runner, load_time, mem_delta in runners:
            print(f"  {runner.name}...", end=" ", flush=True)

            result = run_transcription(runner, seg.wav_path)
            result.load_time = load_time

            # Calculate CER
            cer = calculate_cer(ground_truth_clean, result.text_traditional)
            rtf = result.inference_time / audio_info['duration']

            print(f"{result.inference_time:.2f}s (RTF: {rtf:.2f}, CER: {cer:.1%})")

            seg_result.transcriptions.append({
                "model": runner.name,
                "device": runner.device,
                "model_info": runner.model_info,
                "result": result,
                "cer": cer,
                "rtf": rtf,
                "load_memory_mb": mem_delta,
            })

        results.append(seg_result)

    # Generate report
    print(f"\nGenerating report: {config.output_file}")
    generate_report(config, results, device_desc)
    print("Done!")


def generate_report(config: BenchmarkConfig, results: list[SegmentResult], device_desc: str):
    """Generate detailed markdown benchmark report."""
    lines = [
        "# YouTube ASR Benchmark Results",
        "",
        "## Benchmark Info",
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Device:** {device_desc}",
    ]

    if config.youtube_url:
        lines.append(f"- **YouTube URL:** {config.youtube_url}")
    if config.youtube_title:
        lines.append(f"- **Video Title:** {config.youtube_title}")

    lines.extend(["", "---", ""])

    # Summary table
    lines.extend([
        "## Summary",
        "",
        "| Model | Avg RTF | Avg CER | Load RAM | Device |",
        "|-------|---------|---------|----------|--------|",
    ])

    # Collect stats per model
    model_stats: dict[str, list] = {}
    for seg_result in results:
        for trans in seg_result.transcriptions:
            model = trans["model"]
            if model not in model_stats:
                model_stats[model] = {"rtf": [], "cer": [], "mem": trans["load_memory_mb"], "device": trans["device"]}
            model_stats[model]["rtf"].append(trans["rtf"])
            model_stats[model]["cer"].append(trans["cer"])

    for model, stats in model_stats.items():
        avg_rtf = sum(stats["rtf"]) / len(stats["rtf"])
        avg_cer = sum(stats["cer"]) / len(stats["cer"])
        lines.append(f"| {model} | {avg_rtf:.2f} | {avg_cer:.1%} | +{stats['mem']:.0f}MB | {stats['device']} |")

    lines.extend([
        "",
        "*RTF (Real-Time Factor) = processing_time / audio_duration. Lower is faster.*",
        "*CER (Character Error Rate) = edit_distance / reference_length. Lower is better.*",
        "",
        "---",
        "",
    ])

    # Detailed results per segment
    for seg_result in results:
        start_sec = seg_result.start_ms / 1000
        end_sec = seg_result.end_ms / 1000

        lines.extend([
            f"## {seg_result.name}",
            "",
            "### Audio Info",
            f"- **Duration:** {seg_result.duration:.1f}s",
            f"- **Time range:** {format_time(start_sec)} - {format_time(end_sec)}",
            f"- **Sample rate:** {seg_result.audio_info['sample_rate']}Hz",
            f"- **Channels:** {seg_result.audio_info['channels']}",
            "",
            "### YouTube Original (Ground Truth)",
            f"> {seg_result.ground_truth_clean}",
            "",
        ])

        for trans in seg_result.transcriptions:
            result = trans["result"]
            lines.extend([
                f"### {trans['model']}",
                f"- **Time:** {result.inference_time:.2f}s | **RTF:** {trans['rtf']:.2f} | **CER:** {trans['cer']:.1%}",
                f"- **Device:** {trans['device']} | **Info:** {trans['model_info']}",
                "",
            ])

            if result.error:
                lines.append(f"> **Error:** {result.error}")
            else:
                # Show both original and Traditional Chinese converted
                lines.append(f"> **Raw:** {result.text_clean}")
                lines.append(f">")
                lines.append(f"> **Traditional:** {result.text_traditional}")

            lines.extend(["", ""])

        lines.extend(["---", ""])

    with open(config.output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description='YouTube ASR Benchmark v2 - Compare with original transcript'
    )
    parser.add_argument(
        '--srt', '-s',
        default=os.path.join(PROJECT_ROOT, "youtube_test", "zh-tw.srt"),
        help='Path to SRT file'
    )
    parser.add_argument(
        '--output', '-o',
        default=os.path.join(PROJECT_ROOT, "benchmark_results", "youtube_benchmark_results.md"),
        help='Output markdown file'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['elevenlabs', 'sensevoice', 'paraformer', 'fun-asr-nano'],
        help='Models to benchmark (default: all)'
    )
    parser.add_argument(
        '--url', '-u',
        default="https://www.youtube.com/watch?v=56-dpUWm-sA",
        help='YouTube video URL'
    )
    parser.add_argument(
        '--title', '-t',
        default="志祺七七 - AI影片討論",
        help='YouTube video title'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  YouTube ASR Benchmark v2")
    print("=" * 60)

    config = BenchmarkConfig(
        youtube_url=args.url,
        youtube_title=args.title,
        srt_path=args.srt,
        output_file=args.output,
    )

    # Define segments to test
    youtube_test_dir = os.path.join(PROJECT_ROOT, "youtube_test")
    segments = [
        SegmentConfig(
            wav_path=os.path.join(youtube_test_dir, "segment_01.wav"),
            start_ms=33000,
            end_ms=50000,
            name="Segment 1 (Intro)"
        ),
        SegmentConfig(
            wav_path=os.path.join(youtube_test_dir, "segment_02.wav"),
            start_ms=60000,
            end_ms=90000,
            name="Segment 2 (AI Discussion)"
        ),
        SegmentConfig(
            wav_path=os.path.join(youtube_test_dir, "segment_03.wav"),
            start_ms=166000,  # 00:02:46 - after ad ends
            end_ms=196000,    # 00:03:16
            name="Segment 3 (AI ASMR Discussion)"
        ),
    ]

    # Check files exist
    for seg in segments:
        if not os.path.exists(seg.wav_path):
            print(f"Error: Segment file not found: {seg.wav_path}")
            return

    if not os.path.exists(args.srt):
        print(f"Error: SRT file not found: {args.srt}")
        return

    run_youtube_benchmark(config, segments, args.models)


if __name__ == "__main__":
    main()
