#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "funasr",
#     "torch",
#     "numpy",
#     "transformers",
#     "pydub",
#     "opencc-python-reimplemented",
#     "psutil",
# ]
# ///
"""
Full YouTube ASR Benchmark - Compare ASR models against YouTube's original transcript.

This script transcribes an ENTIRE video and provides side-by-side comparison
of ground truth vs each ASR model, segment by segment.

Features:
- Processes full video (not just pre-extracted segments)
- Dynamic audio extraction from video.wav
- Groups SRT entries into ~60s segments for comparison
- Side-by-side output format per segment
- Sequential model loading to save memory
- No cloud APIs (local models only)
"""

import argparse
import gc
import os
import re
import tempfile
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime

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
    text = re.sub(r'[🎼😊🎵🎶👏😄😢😠]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def convert_to_traditional(text: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese."""
    return S2T_CONVERTER.convert(text)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER) using Levenshtein distance."""
    ref = reference.replace(' ', '')
    hyp = hypothesis.replace(' ', '')

    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0

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


def format_time_ms(ms: int) -> str:
    """Format milliseconds as HH:MM:SS."""
    total_seconds = ms // 1000
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_audio_duration_ms(wav_path: str) -> int:
    """Get duration of a WAV file in milliseconds."""
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return int(frames / rate * 1000)


# =============================================================================
# SRT Parsing and Grouping
# =============================================================================

@dataclass
class SRTEntry:
    index: int
    start_ms: int
    end_ms: int
    text: str


@dataclass
class SegmentGroup:
    """A group of SRT entries combined into one segment."""
    index: int
    start_ms: int
    end_ms: int
    text: str  # Combined text from all entries
    entry_count: int


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


def group_srt_entries(entries: list[SRTEntry], target_duration_ms: int = 60000) -> list[SegmentGroup]:
    """
    Group consecutive SRT entries into segments of approximately target_duration_ms.

    Strategy:
    - Accumulate entries until we reach or exceed target_duration_ms
    - Then start a new segment
    - Handle gaps (silence) gracefully
    """
    if not entries:
        return []

    groups = []
    current_entries = []
    current_start_ms = entries[0].start_ms

    for entry in entries:
        # Check if adding this entry would exceed target duration
        potential_duration = entry.end_ms - current_start_ms

        if potential_duration > target_duration_ms and current_entries:
            # Finalize current group
            combined_text = ' '.join(e.text for e in current_entries)
            groups.append(SegmentGroup(
                index=len(groups) + 1,
                start_ms=current_start_ms,
                end_ms=current_entries[-1].end_ms,
                text=combined_text,
                entry_count=len(current_entries),
            ))
            # Start new group
            current_entries = [entry]
            current_start_ms = entry.start_ms
        else:
            current_entries.append(entry)

    # Don't forget the last group
    if current_entries:
        combined_text = ' '.join(e.text for e in current_entries)
        groups.append(SegmentGroup(
            index=len(groups) + 1,
            start_ms=current_start_ms,
            end_ms=current_entries[-1].end_ms,
            text=combined_text,
            entry_count=len(current_entries),
        ))

    return groups


# =============================================================================
# Audio Extraction
# =============================================================================

def extract_audio_segment(video_wav: str, start_ms: int, end_ms: int, temp_dir: str) -> str:
    """
    Extract a segment from the video WAV file.

    Returns path to temporary WAV file (16kHz mono).
    """
    audio = AudioSegment.from_wav(video_wav)
    segment = audio[start_ms:end_ms]

    # Ensure 16kHz mono for ASR models
    segment = segment.set_frame_rate(16000).set_channels(1)

    # Save to temp file
    temp_path = os.path.join(temp_dir, f"segment_{start_ms}_{end_ms}.wav")
    segment.export(temp_path, format="wav")

    return temp_path


# =============================================================================
# Model Runners
# =============================================================================

@dataclass
class TranscriptionResult:
    text: str
    text_clean: str
    text_traditional: str
    inference_time: float
    error: str | None = None


class BaseRunner:
    name: str = "Base"
    device: str = "Unknown"

    def transcribe(self, wav_path: str) -> str:
        raise NotImplementedError

    def cleanup(self):
        """Clean up model to free memory."""
        pass


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

    def cleanup(self):
        del self.model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


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

    def cleanup(self):
        del self.model
        gc.collect()


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

    def cleanup(self):
        del self.model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class ModelResult:
    """Results for one model across all segments."""
    name: str
    device: str
    load_time: float
    load_memory_mb: float
    segments: dict = field(default_factory=dict)  # segment_idx -> TranscriptionResult
    total_inference_time: float = 0.0


@dataclass
class SegmentComparison:
    """Comparison data for one segment."""
    index: int
    start_ms: int
    end_ms: int
    duration_ms: int
    ground_truth: str
    ground_truth_clean: str
    model_results: dict = field(default_factory=dict)  # model_name -> (TranscriptionResult, cer)


def run_model_on_segments(
    runner_class,
    runner_kwargs: dict,
    segments: list[SegmentGroup],
    video_wav: str,
    temp_dir: str,
) -> ModelResult:
    """
    Load one model, run it on all segments, then unload.
    This saves memory by not having multiple models loaded simultaneously.
    """
    # Load model
    print(f"\n  Loading model...", end=" ", flush=True)
    mem_before = get_memory_mb()
    load_start = time.time()

    runner = runner_class(**runner_kwargs)

    load_time = time.time() - load_start
    load_memory = get_memory_mb() - mem_before
    print(f"done in {load_time:.1f}s (+{load_memory:.0f}MB)")

    result = ModelResult(
        name=runner.name,
        device=runner.device,
        load_time=load_time,
        load_memory_mb=load_memory,
    )

    # Process each segment
    for seg in segments:
        print(f"    Segment {seg.index} ({format_time_ms(seg.start_ms)} - {format_time_ms(seg.end_ms)})...", end=" ", flush=True)

        # Extract audio
        wav_path = extract_audio_segment(video_wav, seg.start_ms, seg.end_ms, temp_dir)

        # Transcribe
        start_time = time.time()
        error = None
        text = ""

        try:
            text = runner.transcribe(wav_path)
        except Exception as e:
            error = str(e)
            import traceback
            traceback.print_exc()

        inference_time = time.time() - start_time
        result.total_inference_time += inference_time

        # Clean and convert
        text_clean = clean_transcript(text)
        text_traditional = convert_to_traditional(text_clean)

        trans_result = TranscriptionResult(
            text=text,
            text_clean=text_clean,
            text_traditional=text_traditional,
            inference_time=inference_time,
            error=error,
        )

        result.segments[seg.index] = trans_result

        duration_s = (seg.end_ms - seg.start_ms) / 1000
        rtf = inference_time / duration_s if duration_s > 0 else 0
        print(f"{inference_time:.1f}s (RTF: {rtf:.2f})")

        # Clean up temp file
        os.remove(wav_path)

    # Unload model
    print(f"  Unloading model...", end=" ", flush=True)
    runner.cleanup()
    gc.collect()
    print("done")

    return result


def run_full_benchmark(
    video_wav: str,
    srt_path: str,
    output_path: str,
    segment_duration_s: int = 60,
    models: list[str] | None = None,
    youtube_url: str = "",
    youtube_title: str = "",
):
    """Run full video benchmark with side-by-side comparison."""

    print("=" * 60)
    print("  Full YouTube ASR Benchmark")
    print("=" * 60)

    # Parse and group SRT
    print("\nParsing SRT file...")
    srt_entries = parse_srt(srt_path)
    print(f"  Found {len(srt_entries)} subtitle entries")

    segments = group_srt_entries(srt_entries, target_duration_ms=segment_duration_s * 1000)
    print(f"  Grouped into {len(segments)} segments (~{segment_duration_s}s each)")

    # Video info
    video_duration_ms = get_audio_duration_ms(video_wav)
    print(f"\nVideo duration: {format_time_ms(video_duration_ms)}")

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

    # Model configurations
    if models is None:
        models = ["sensevoice", "paraformer", "fun-asr-nano"]

    model_configs = {
        "sensevoice": (SenseVoiceRunner, {"device": device}),
        "paraformer": (ParaformerRunner, {}),
        "fun-asr-nano": (FunASRNanoRunner, {"device": device}),
    }

    # Create temp directory for extracted audio
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nTemp directory: {temp_dir}")

        # Run each model sequentially
        model_results: dict[str, ModelResult] = {}

        for model_name in models:
            if model_name not in model_configs:
                print(f"\nUnknown model: {model_name}, skipping")
                continue

            print(f"\n{'='*40}")
            print(f"  Model: {model_name.upper()}")
            print(f"{'='*40}")

            runner_class, runner_kwargs = model_configs[model_name]

            try:
                result = run_model_on_segments(
                    runner_class,
                    runner_kwargs,
                    segments,
                    video_wav,
                    temp_dir,
                )
                model_results[model_name] = result
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Build comparison data
    comparisons: list[SegmentComparison] = []

    for seg in segments:
        ground_truth = seg.text
        ground_truth_clean = clean_transcript(ground_truth)

        comparison = SegmentComparison(
            index=seg.index,
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            duration_ms=seg.end_ms - seg.start_ms,
            ground_truth=ground_truth,
            ground_truth_clean=ground_truth_clean,
        )

        for model_name, model_result in model_results.items():
            if seg.index in model_result.segments:
                trans = model_result.segments[seg.index]
                cer = calculate_cer(ground_truth_clean, trans.text_traditional)
                comparison.model_results[model_name] = (trans, cer)

        comparisons.append(comparison)

    # Generate report
    print(f"\n{'='*40}")
    print(f"  Generating Report")
    print(f"{'='*40}")

    generate_side_by_side_report(
        output_path=output_path,
        comparisons=comparisons,
        model_results=model_results,
        device_desc=device_desc,
        video_duration_ms=video_duration_ms,
        youtube_url=youtube_url,
        youtube_title=youtube_title,
    )

    print(f"\nReport saved to: {output_path}")
    print("Done!")


def generate_side_by_side_report(
    output_path: str,
    comparisons: list[SegmentComparison],
    model_results: dict[str, ModelResult],
    device_desc: str,
    video_duration_ms: int,
    youtube_url: str,
    youtube_title: str,
):
    """Generate markdown report with side-by-side segment comparison."""

    lines = [
        "# Full YouTube ASR Benchmark Results",
        "",
        "## Benchmark Info",
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Device:** {device_desc}",
        f"- **Video Duration:** {format_time_ms(video_duration_ms)}",
        f"- **Segments:** {len(comparisons)}",
    ]

    if youtube_url:
        lines.append(f"- **YouTube URL:** {youtube_url}")
    if youtube_title:
        lines.append(f"- **Video Title:** {youtube_title}")

    lines.extend(["", "---", ""])

    # Summary table
    lines.extend([
        "## Summary",
        "",
        "| Model | Device | Load Time | Load RAM | Total Time | Avg RTF | Avg CER |",
        "|-------|--------|-----------|----------|------------|---------|---------|",
    ])

    for model_name, result in model_results.items():
        # Calculate averages
        total_duration_s = sum(c.duration_ms for c in comparisons) / 1000
        avg_rtf = result.total_inference_time / total_duration_s if total_duration_s > 0 else 0

        cers = []
        for comp in comparisons:
            if model_name in comp.model_results:
                _, cer = comp.model_results[model_name]
                cers.append(cer)
        avg_cer = sum(cers) / len(cers) if cers else 0

        lines.append(
            f"| {result.name} | {result.device} | {result.load_time:.1f}s | "
            f"+{result.load_memory_mb:.0f}MB | {result.total_inference_time:.1f}s | "
            f"{avg_rtf:.2f} | {avg_cer:.1%} |"
        )

    lines.extend([
        "",
        "*RTF (Real-Time Factor) = processing_time / audio_duration. Lower is faster.*",
        "*CER (Character Error Rate) = edit_distance / reference_length. Lower is better.*",
        "",
        "---",
        "",
    ])

    # Side-by-side segment comparisons
    lines.append("## Segment Comparisons")
    lines.append("")

    for comp in comparisons:
        time_range = f"{format_time_ms(comp.start_ms)} - {format_time_ms(comp.end_ms)}"
        duration_s = comp.duration_ms / 1000

        lines.extend([
            f"### Segment {comp.index} ({time_range})",
            f"*Duration: {duration_s:.1f}s*",
            "",
            "**Ground Truth (YouTube):**",
            f"> {comp.ground_truth_clean}",
            "",
        ])

        # Each model's result
        for model_name, (trans, cer) in comp.model_results.items():
            rtf = trans.inference_time / duration_s if duration_s > 0 else 0

            lines.append(f"**{model_name}:** (CER: {cer:.1%}, RTF: {rtf:.2f})")

            if trans.error:
                lines.append(f"> **Error:** {trans.error}")
            else:
                lines.append(f"> {trans.text_traditional}")

            lines.append("")

        lines.extend(["---", ""])

    # Full transcriptions appendix
    lines.extend([
        "## Appendix: Full Transcriptions",
        "",
    ])

    for model_name, result in model_results.items():
        lines.extend([
            f"### {result.name}",
            "",
        ])

        full_text_parts = []
        for comp in comparisons:
            if model_name in comp.model_results:
                trans, _ = comp.model_results[model_name]
                if trans.text_traditional:
                    full_text_parts.append(trans.text_traditional)

        full_text = ' '.join(full_text_parts)
        lines.append(f"> {full_text}")
        lines.extend(["", ""])

    # Write file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description='Full YouTube ASR Benchmark - transcribe entire video with side-by-side comparison'
    )
    parser.add_argument(
        '--video', '-v',
        default=os.path.join(PROJECT_ROOT, "youtube_test", "video.wav"),
        help='Path to full video WAV file'
    )
    parser.add_argument(
        '--srt', '-s',
        default=os.path.join(PROJECT_ROOT, "youtube_test", "zh-tw.srt"),
        help='Path to SRT file'
    )
    parser.add_argument(
        '--output', '-o',
        default=os.path.join(PROJECT_ROOT, "benchmark_results", "full_benchmark_results.md"),
        help='Output markdown file'
    )
    parser.add_argument(
        '--segment-duration', '-d',
        type=int,
        default=60,
        help='Target segment duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['sensevoice', 'paraformer', 'fun-asr-nano'],
        help='Models to benchmark (default: all)'
    )
    parser.add_argument(
        '--url', '-u',
        default="https://www.youtube.com/watch?v=56-dpUWm-sA",
        help='YouTube video URL (for report metadata)'
    )
    parser.add_argument(
        '--title', '-t',
        default="志祺七七 - AI影片討論",
        help='YouTube video title (for report metadata)'
    )

    args = parser.parse_args()

    # Validate files exist
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    if not os.path.exists(args.srt):
        print(f"Error: SRT file not found: {args.srt}")
        return 1

    run_full_benchmark(
        video_wav=args.video,
        srt_path=args.srt,
        output_path=args.output,
        segment_duration_s=args.segment_duration,
        models=args.models,
        youtube_url=args.url,
        youtube_title=args.title,
    )

    return 0


if __name__ == "__main__":
    exit(main())
