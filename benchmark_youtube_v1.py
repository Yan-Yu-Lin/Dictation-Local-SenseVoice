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
# ]
# ///
"""
YouTube ASR Benchmark - Compare ASR models against YouTube's original transcript.

Compares:
1. YouTube original transcript (ground truth)
2. ElevenLabs Scribe v2 (cloud API)
3. SenseVoice (local)
4. Paraformer (local, CPU)
5. Fun-ASR-Nano (local)
"""

import argparse
import io
import os
import re
import time
import wave
from datetime import datetime
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import torch
from pydub import AudioSegment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class SRTEntry:
    index: int
    start_ms: int
    end_ms: int
    text: str


def parse_timestamp(ts: str) -> int:
    """Convert SRT timestamp to milliseconds."""
    # Format: HH:MM:SS,mmm
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

    # Split by double newline (entries separated by blank lines)
    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                # Parse timestamp line
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
        # Entry overlaps with range
        if entry.end_ms > start_ms and entry.start_ms < end_ms:
            texts.append(entry.text)
    return ' '.join(texts)


def get_audio_duration(wav_path: str) -> float:
    """Get duration of a WAV file in seconds."""
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate


def convert_wav_to_mp3(wav_path: str) -> bytes:
    """Convert WAV file to MP3 bytes for API upload."""
    audio = AudioSegment.from_wav(wav_path)
    buffer = io.BytesIO()
    audio.export(buffer, format="mp3")
    return buffer.getvalue()


# =============================================================================
# Model Runners (same as benchmark.py)
# =============================================================================

class ElevenLabsRunner:
    def __init__(self):
        from elevenlabs import ElevenLabs as ElevenLabsClient
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")
        self.client = ElevenLabsClient(api_key=api_key)
        self.name = "ElevenLabs Scribe v2"
        self.device = "Cloud API"

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


class SenseVoiceRunner:
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


class ParaformerRunner:
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


class FunASRNanoRunner:
    def __init__(self, device: str):
        from funasr import AutoModel
        remote_code_path = os.path.join(SCRIPT_DIR, "fun_asr_nano_model.py")
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


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class SegmentConfig:
    wav_path: str
    start_ms: int
    end_ms: int
    name: str


def run_youtube_benchmark(
    srt_path: str,
    segments: list[SegmentConfig],
    output_file: str,
    models: list[str] | None = None
):
    """Run benchmark comparing ASR models with YouTube transcript."""

    # Parse SRT
    print("Parsing SRT file...")
    srt_entries = parse_srt(srt_path)
    print(f"Found {len(srt_entries)} subtitle entries")

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Device: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda:0"
        print("Device: NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print("Device: CPU")

    # Initialize models
    print("\nLoading models...")
    runners = []

    if models is None:
        models = ["elevenlabs", "sensevoice", "paraformer", "fun-asr-nano"]

    for model_name in models:
        try:
            if model_name == "elevenlabs":
                print("  Loading ElevenLabs...")
                runners.append(ElevenLabsRunner())
            elif model_name == "sensevoice":
                print("  Loading SenseVoice...")
                runners.append(SenseVoiceRunner(device))
            elif model_name == "paraformer":
                print("  Loading Paraformer...")
                runners.append(ParaformerRunner())
            elif model_name == "fun-asr-nano":
                print("  Loading Fun-ASR-Nano...")
                runners.append(FunASRNanoRunner(device))
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")

    print(f"\nLoaded {len(runners)} model(s)")

    # Run benchmark
    results = []
    for seg in segments:
        duration = get_audio_duration(seg.wav_path)
        ground_truth = get_transcript_for_range(srt_entries, seg.start_ms, seg.end_ms)

        print(f"\nProcessing: {seg.name} ({duration:.1f}s)")
        print(f"  Ground truth: {ground_truth[:50]}...")

        seg_results = {
            "name": seg.name,
            "wav_path": seg.wav_path,
            "duration": duration,
            "start_ms": seg.start_ms,
            "end_ms": seg.end_ms,
            "ground_truth": ground_truth,
            "transcriptions": []
        }

        for runner in runners:
            print(f"  {runner.name}...", end=" ", flush=True)
            start_time = time.time()

            try:
                text = runner.transcribe(seg.wav_path)
                elapsed = time.time() - start_time
                print(f"{elapsed:.2f}s")

                seg_results["transcriptions"].append({
                    "model": runner.name,
                    "device": runner.device,
                    "time": elapsed,
                    "text": text,
                    "error": None,
                })
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"ERROR: {e}")
                seg_results["transcriptions"].append({
                    "model": runner.name,
                    "device": runner.device,
                    "time": elapsed,
                    "text": "",
                    "error": str(e),
                })

        results.append(seg_results)

    # Generate report
    print(f"\nGenerating report: {output_file}")
    generate_report(results, output_file)
    print("Done!")


def generate_report(results: list, output_file: str):
    """Generate markdown benchmark report."""
    lines = [
        "# YouTube ASR Benchmark Results",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Comparing ASR models against YouTube's original transcript.",
        "",
    ]

    for seg_result in results:
        start_sec = seg_result['start_ms'] / 1000
        end_sec = seg_result['end_ms'] / 1000

        lines.extend([
            f"## {seg_result['name']}",
            f"- Duration: {seg_result['duration']:.1f}s",
            f"- Time range: {start_sec:.1f}s - {end_sec:.1f}s",
            "",
            "### YouTube Original (Ground Truth)",
            f"> {seg_result['ground_truth']}",
            "",
        ])

        for trans in seg_result["transcriptions"]:
            lines.append(f"### {trans['model']}")
            lines.append(f"- Time: {trans['time']:.2f}s | Device: {trans['device']}")

            if trans["error"]:
                lines.append(f"> **Error:** {trans['error']}")
            elif trans["text"]:
                lines.append(f"> {trans['text']}")
            else:
                lines.append("> *(no output)*")

            lines.append("")

        lines.append("---")
        lines.append("")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description='YouTube ASR Benchmark - Compare with original transcript'
    )
    parser.add_argument(
        '--srt', '-s',
        default=os.path.join(SCRIPT_DIR, "youtube_test", "zh-tw.srt"),
        help='Path to SRT file'
    )
    parser.add_argument(
        '--output', '-o',
        default=os.path.join(SCRIPT_DIR, "youtube_benchmark_results.md"),
        help='Output markdown file'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['elevenlabs', 'sensevoice', 'paraformer', 'fun-asr-nano'],
        help='Models to benchmark (default: all)'
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  YouTube ASR Benchmark")
    print("=" * 50)

    # Define segments to test
    youtube_test_dir = os.path.join(SCRIPT_DIR, "youtube_test")
    segments = [
        SegmentConfig(
            wav_path=os.path.join(youtube_test_dir, "segment_01.wav"),
            start_ms=33000,  # 00:00:33
            end_ms=50000,    # 00:00:50
            name="Segment 1 (Intro)"
        ),
        SegmentConfig(
            wav_path=os.path.join(youtube_test_dir, "segment_02.wav"),
            start_ms=60000,  # 00:01:00
            end_ms=90000,    # 00:01:30
            name="Segment 2 (AI Discussion)"
        ),
        SegmentConfig(
            wav_path=os.path.join(youtube_test_dir, "segment_03.wav"),
            start_ms=120000, # 00:02:00
            end_ms=150000,   # 00:02:30
            name="Segment 3 (Continued)"
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

    run_youtube_benchmark(args.srt, segments, args.output, args.models)


if __name__ == "__main__":
    main()
