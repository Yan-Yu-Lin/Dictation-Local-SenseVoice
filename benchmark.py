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
ASR Benchmark Runner - Compare multiple ASR models on recorded audio files.

Models compared:
1. ElevenLabs Scribe v2 (cloud API - reference)
2. SenseVoice (local)
3. Paraformer (local, CPU)
4. Fun-ASR-Nano (local)

Usage:
    uv run python benchmark.py
    uv run python benchmark.py --folder ./my_recordings
    uv run python benchmark.py --output results.md
"""

import argparse
import io
import os
import time
import wave
from datetime import datetime

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Third-party imports
import torch
from pydub import AudioSegment

# Script directory for model paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RECORDINGS_DIR = os.path.join(SCRIPT_DIR, "recordings")
DEFAULT_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "results", "benchmark_results.md")


def get_device_info():
    """Detect available compute devices."""
    if torch.backends.mps.is_available():
        return "mps", "Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        return "cuda:0", "NVIDIA GPU (CUDA)"
    else:
        return "cpu", "CPU"


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
# Model Runners
# =============================================================================

class ElevenLabsRunner:
    """ElevenLabs Scribe v2 cloud API."""

    def __init__(self):
        from elevenlabs import ElevenLabs as ElevenLabsClient
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")
        self.client = ElevenLabsClient(api_key=api_key)
        self.name = "ElevenLabs Scribe v2"
        self.device = "Cloud API"

    def transcribe(self, wav_path: str) -> str:
        """Transcribe audio file using ElevenLabs API."""
        # Convert to MP3 for faster upload
        mp3_bytes = convert_wav_to_mp3(wav_path)

        # Create file-like object for API
        audio_file = io.BytesIO(mp3_bytes)
        audio_file.name = "audio.mp3"  # API needs filename with extension

        result = self.client.speech_to_text.convert(
            file=audio_file,
            model_id="scribe_v2",
            language_code=None,  # Auto-detect language
        )
        return result.text


class SenseVoiceRunner:
    """SenseVoice local model."""

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
        """Transcribe audio file using SenseVoice."""
        # Fix thread count drift bug
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
    """Paraformer local model (CPU recommended)."""

    def __init__(self):
        from funasr import AutoModel
        # Force CPU for Paraformer due to MPS compatibility issues
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
        """Transcribe audio file using Paraformer."""
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
    """Fun-ASR-Nano local model."""

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
        """Transcribe audio file using Fun-ASR-Nano."""
        torch.set_num_threads(4)

        res = self.model.generate(
            input=[wav_path],  # List format for Fun-ASR-Nano
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

def run_benchmark(recordings_dir: str, output_file: str, models: list[str] | None = None):
    """Run benchmark on all WAV files in the recordings directory."""

    # Find all WAV files
    wav_files = sorted([
        f for f in os.listdir(recordings_dir)
        if f.endswith('.wav')
    ])

    if not wav_files:
        print(f"No WAV files found in {recordings_dir}")
        return

    print(f"Found {len(wav_files)} audio file(s)")

    # Detect device
    device, device_desc = get_device_info()
    print(f"Device: {device_desc}")

    # Initialize models
    print("\nLoading models...")
    runners = []

    # Determine which models to load
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
            else:
                print(f"  Unknown model: {model_name}, skipping")
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")

    if not runners:
        print("No models loaded. Exiting.")
        return

    print(f"\nLoaded {len(runners)} model(s)")

    # Run benchmark
    results = []
    for wav_file in wav_files:
        wav_path = os.path.join(recordings_dir, wav_file)
        duration = get_audio_duration(wav_path)

        print(f"\nProcessing: {wav_file} ({duration:.1f}s)")

        file_results = {
            "filename": wav_file,
            "duration": duration,
            "transcriptions": []
        }

        for runner in runners:
            print(f"  {runner.name}...", end=" ", flush=True)
            start_time = time.time()

            try:
                text = runner.transcribe(wav_path)
                elapsed = time.time() - start_time
                print(f"{elapsed:.2f}s")

                file_results["transcriptions"].append({
                    "model": runner.name,
                    "device": runner.device,
                    "time": elapsed,
                    "text": text,
                    "error": None,
                })
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"ERROR: {e}")

                file_results["transcriptions"].append({
                    "model": runner.name,
                    "device": runner.device,
                    "time": elapsed,
                    "text": "",
                    "error": str(e),
                })

        results.append(file_results)

    # Generate markdown report
    print(f"\nGenerating report: {output_file}")
    generate_report(results, output_file)
    print("Done!")


def generate_report(results: list, output_file: str):
    """Generate markdown benchmark report."""
    lines = [
        "# ASR Benchmark Results",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for file_result in results:
        lines.extend([
            f"## {file_result['filename']}",
            f"Duration: {file_result['duration']:.1f} seconds",
            "",
        ])

        for i, trans in enumerate(file_result["transcriptions"]):
            # First model is reference
            suffix = " (Reference)" if i == 0 else ""
            lines.append(f"### {trans['model']}{suffix}")
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
        description='ASR Benchmark Runner - Compare multiple ASR models'
    )
    parser.add_argument(
        '--folder', '-f',
        default=DEFAULT_RECORDINGS_DIR,
        help=f'Folder containing WAV files (default: {DEFAULT_RECORDINGS_DIR})'
    )
    parser.add_argument(
        '--output', '-o',
        default=DEFAULT_OUTPUT_FILE,
        help=f'Output markdown file (default: {DEFAULT_OUTPUT_FILE})'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['elevenlabs', 'sensevoice', 'paraformer', 'fun-asr-nano'],
        help='Models to benchmark (default: all)'
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  ASR Benchmark Runner")
    print("=" * 50)

    if not os.path.isdir(args.folder):
        print(f"Error: Recordings folder not found: {args.folder}")
        print("Run record_audio.py first to create recordings.")
        return

    run_benchmark(args.folder, args.output, args.models)


if __name__ == "__main__":
    main()
