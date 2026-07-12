#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "funasr",
#     "torch",
#     "numpy",
#     "pydub",
#     "opencc-python-reimplemented",
# ]
# ///
"""
Transcribe an audio/video file using SenseVoice (or other FunASR models).

Usage:
    uv run transcribe_audio.py /path/to/audio.mp3
    uv run transcribe_audio.py /path/to/audio.mp3 --model paraformer
    uv run transcribe_audio.py /path/to/audio.mp3 --output transcript.txt

Supported models:
    - sensevoice (default): Multilingual ASR from Alibaba DAMO
    - paraformer: Chinese-focused ASR
    - fun-asr-nano: LLM-based ASR with 31 languages support

Output formats:
    - txt (default): Plain text transcription
    - srt: Subtitle format with timestamps (requires VAD segmentation)
    - json: Full output with metadata
"""

import argparse
import gc
import json
import os
import re
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import torch
import opencc
from pydub import AudioSegment

# Get project root for remote_code paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR

# OpenCC converter for Simplified -> Traditional Chinese
S2T_CONVERTER = opencc.OpenCC("s2t")


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((td.microseconds / 1000) % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as WebVTT timestamp (HH:MM:SS.mmm)."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((td.microseconds / 1000) % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def clean_transcript(text: str) -> str:
    """Clean transcript text - remove emoji artifacts, normalize whitespace."""
    # Remove common emotion/event tags from SenseVoice
    text = re.sub(r"[🎼😊🎵🎶👏😄😢😠]", "", text)
    # Remove tag prefixes like <|zh|>, <|en|>, <|NEUTRAL|>, etc.
    text = re.sub(r"<\|[^|]+\|>", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def convert_to_traditional(text: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese."""
    return S2T_CONVERTER.convert(text)


@dataclass
class TranscriptionSegment:
    """A single transcription segment with timing info."""

    start_time: float  # seconds
    end_time: float  # seconds
    text: str

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class BaseTranscriber:
    """Base class for ASR transcribers."""

    name: str = "Base"
    device: str = "Unknown"

    def __init__(self, device: str | None = None):
        self.model = None
        self._device = device

    def load(self):
        """Load the model. Override in subclasses."""
        raise NotImplementedError

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text. Override in subclasses."""
        raise NotImplementedError

    def transcribe_with_timestamps(self, audio_path: str) -> list[TranscriptionSegment]:
        """
        Transcribe with timestamps.
        Default implementation returns single segment for whole file.
        Override for VAD-based segmentation.
        """
        text = self.transcribe(audio_path)
        # Get audio duration
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0
        return [TranscriptionSegment(0.0, duration, text)]

    def cleanup(self):
        """Clean up model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


class SenseVoiceTranscriber(BaseTranscriber):
    """SenseVoice transcriber - multilingual ASR from Alibaba DAMO."""

    name = "SenseVoice"

    def __init__(self, device: str | None = None):
        super().__init__(device)
        self.device = device or self._detect_device()

    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def load(self):
        """Load SenseVoice model."""
        from funasr import AutoModel

        print(f"Loading SenseVoice model on {self.device.upper()}...")
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            device=self.device,
            disable_update=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
        )
        print("Model loaded!")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio to text."""
        torch.set_num_threads(4)

        res = self.model.generate(
            input=audio_path,
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
                from funasr.utils.postprocess_utils import (
                    rich_transcription_postprocess,
                )

                return rich_transcription_postprocess(text)
            except Exception:
                return text
        return ""

    def transcribe_with_timestamps(self, audio_path: str) -> list[TranscriptionSegment]:
        """
        Transcribe with VAD-based timestamps.
        Uses the VAD model to segment audio and transcribe each segment.
        """
        torch.set_num_threads(4)

        # Generate with VAD to get timestamp information
        res = self.model.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )

        torch.set_num_threads(4)

        segments = []

        if res and len(res) > 0:
            result = res[0]

            # Check if we have timestamp information
            if "sentence_info" in result:
                # Process sentence-level timestamps
                for sent in result["sentence_info"]:
                    start = sent.get("start", 0) / 1000.0  # Convert ms to seconds
                    end = sent.get("end", 0) / 1000.0
                    text = sent.get("text", "")

                    # Clean and process text
                    text = clean_transcript(text)
                    if text:
                        segments.append(TranscriptionSegment(start, end, text))
            elif "text" in result:
                # Fall back to single segment with full text
                text = result["text"]
                try:
                    from funasr.utils.postprocess_utils import (
                        rich_transcription_postprocess,
                    )

                    text = rich_transcription_postprocess(text)
                except Exception:
                    pass

                # Get audio duration
                audio = AudioSegment.from_file(audio_path)
                duration = len(audio) / 1000.0

                text = clean_transcript(text)
                if text:
                    segments.append(TranscriptionSegment(0.0, duration, text))

        return segments if segments else [TranscriptionSegment(0.0, 0.0, "")]


class ParaformerTranscriber(BaseTranscriber):
    """Paraformer transcriber - Chinese-focused ASR."""

    name = "Paraformer"
    device = "CPU"

    def load(self):
        """Load Paraformer model."""
        from funasr import AutoModel

        print("Loading Paraformer model on CPU...")
        print("(Paraformer has known MPS compatibility issues)")

        self.model = AutoModel(
            model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            device="cpu",
            disable_update=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            punc_model="ct-punc",
        )
        print("Model loaded!")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio to text."""
        torch.set_num_threads(4)

        res = self.model.generate(
            input=audio_path,
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


class FunASRNanoTranscriber(BaseTranscriber):
    """Fun-ASR-Nano transcriber - LLM-based ASR."""

    name = "Fun-ASR-Nano"

    def __init__(self, device: str | None = None):
        super().__init__(device)
        self.device = device or self._detect_device()

    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def load(self):
        """Load Fun-ASR-Nano model."""
        from funasr import AutoModel

        remote_code_path = PROJECT_ROOT / "models" / "fun_asr_nano.py"

        print(f"Loading Fun-ASR-Nano model on {self.device.upper()}...")
        self.model = AutoModel(
            model="FunAudioLLM/Fun-ASR-Nano-2512",
            trust_remote_code=True,
            remote_code=str(remote_code_path),
            device=self.device,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
        )
        print("Model loaded!")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio to text."""
        torch.set_num_threads(4)

        res = self.model.generate(
            input=[audio_path],
            cache={},
            batch_size=1,
            language="auto",
            itn=True,
        )

        torch.set_num_threads(4)

        if res and res[0].get("text"):
            return res[0]["text"]
        return ""


# Model registry
TRANSCRIBERS = {
    "sensevoice": SenseVoiceTranscriber,
    "paraformer": ParaformerTranscriber,
    "fun-asr-nano": FunASRNanoTranscriber,
    "nano": FunASRNanoTranscriber,
}


def convert_to_wav(audio_path: str, temp_dir: str | None = None) -> str:
    """
    Convert any audio format to 16kHz mono WAV for ASR processing.
    Returns path to WAV file (may be same as input if already WAV).
    """
    audio_path_obj = Path(audio_path)

    # If already WAV, check if we need to resample
    if audio_path_obj.suffix.lower() == ".wav":
        try:
            audio = AudioSegment.from_wav(str(audio_path_obj))
            # Check if already 16kHz mono
            if audio.frame_rate == 16000 and audio.channels == 1:
                return str(audio_path_obj)
        except Exception:
            pass

    # Convert to 16kHz mono WAV
    print(f"Converting {audio_path_obj.suffix} to 16kHz mono WAV...")

    audio = AudioSegment.from_file(str(audio_path_obj))
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Create temp file if needed
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    wav_path = Path(temp_dir) / f"{audio_path_obj.stem}_16k.wav"
    audio.export(str(wav_path), format="wav")

    return str(wav_path)


def write_txt_output(
    segments: list[TranscriptionSegment],
    output_path: str,
    convert_traditional: bool = True,
):
    """Write transcription as plain text."""
    texts = []
    for seg in segments:
        text = seg.text
        if convert_traditional:
            text = convert_to_traditional(text)
        texts.append(text)

    full_text = " ".join(texts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Transcription saved to: {output_path}")


def write_srt_output(
    segments: list[TranscriptionSegment],
    output_path: str,
    convert_traditional: bool = True,
):
    """Write transcription as SRT subtitle file."""
    lines = []

    for i, seg in enumerate(segments, 1):
        text = seg.text
        if convert_traditional:
            text = convert_to_traditional(text)

        lines.append(str(i))
        lines.append(
            f"{format_timestamp_srt(seg.start_time)} --> {format_timestamp_srt(seg.end_time)}"
        )
        lines.append(text)
        lines.append("")  # Empty line between entries

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"SRT subtitles saved to: {output_path}")


def write_vtt_output(
    segments: list[TranscriptionSegment],
    output_path: str,
    convert_traditional: bool = True,
):
    """Write transcription as WebVTT subtitle file."""
    lines = ["WEBVTT", ""]

    for seg in segments:
        text = seg.text
        if convert_traditional:
            text = convert_to_traditional(text)

        lines.append(
            f"{format_timestamp_vtt(seg.start_time)} --> {format_timestamp_vtt(seg.end_time)}"
        )
        lines.append(text)
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"WebVTT subtitles saved to: {output_path}")


def write_json_output(
    segments: list[TranscriptionSegment],
    output_path: str,
    metadata: dict,
    convert_traditional: bool = True,
):
    """Write transcription as JSON with metadata."""
    output = {"metadata": metadata, "segments": []}

    for seg in segments:
        text = seg.text
        if convert_traditional:
            text = convert_to_traditional(text)

        output["segments"].append(
            {
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "text": text,
            }
        )

    # Add full text
    output["full_text"] = " ".join(s["text"] for s in output["segments"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"JSON output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using SenseVoice or other FunASR models"
    )
    parser.add_argument("input", help="Path to audio/video file")
    parser.add_argument(
        "--model",
        "-m",
        choices=list(TRANSCRIBERS.keys()),
        default="sensevoice",
        help="ASR model to use (default: sensevoice)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: input filename with appropriate extension)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["txt", "srt", "vtt", "json"],
        default="txt",
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--device", "-d", help="Device to use: cpu, mps, cuda:0 (default: auto-detect)"
    )
    parser.add_argument(
        "--chinese",
        "-c",
        choices=["tw", "cn"],
        default="tw",
        help="Chinese output: tw (Traditional, default) or cn (Simplified)",
    )
    parser.add_argument(
        "--timestamps",
        "-t",
        action="store_true",
        help="Generate timestamps (for srt/vtt/json output)",
    )
    parser.add_argument(
        "--keep-wav",
        "-k",
        action="store_true",
        help="Keep converted WAV file (default: delete temp files)",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(f".{args.format}")

    # Create transcriber
    transcriber_class = TRANSCRIBERS[args.model]
    transcriber = transcriber_class(device=args.device)

    print("=" * 60)
    print(f"  Audio Transcription - {transcriber.name}")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Format: {args.format.upper()}")
    print(
        f"Chinese: {'Traditional (TW)' if args.chinese == 'tw' else 'Simplified (CN)'}"
    )
    print("")

    # Convert to WAV if needed
    temp_wav = None
    try:
        if input_path.suffix.lower() != ".wav":
            temp_wav = convert_to_wav(str(input_path))
            audio_path = temp_wav
        else:
            audio_path = str(input_path)

        # Load model
        load_start = time.time()
        transcriber.load()
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.1f}s\n")

        # Transcribe
        print("Transcribing...")
        transcribe_start = time.time()

        if args.timestamps or args.format in ("srt", "vtt", "json"):
            segments = transcriber.transcribe_with_timestamps(audio_path)
        else:
            text = transcriber.transcribe(audio_path)
            # Get audio duration for metadata
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0
            segments = [TranscriptionSegment(0.0, duration, text)]

        transcribe_time = time.time() - transcribe_start

        # Calculate stats
        total_duration = sum(seg.duration for seg in segments)
        rtf = transcribe_time / total_duration if total_duration > 0 else 0

        print(f"\nTranscription complete!")
        print(f"  Duration: {total_duration:.1f}s")
        print(f"  Time: {transcribe_time:.1f}s")
        print(f"  RTF: {rtf:.2f}x")
        print(f"  Segments: {len(segments)}")
        print("")

        # Write output
        convert_traditional = args.chinese == "tw"

        metadata = {
            "model": transcriber.name,
            "input_file": str(input_path),
            "duration_seconds": total_duration,
            "processing_time_seconds": transcribe_time,
            "rtf": rtf,
            "segments_count": len(segments),
            "generated_at": datetime.now().isoformat(),
        }

        if args.format == "txt":
            write_txt_output(segments, str(output_path), convert_traditional)
        elif args.format == "srt":
            write_srt_output(segments, str(output_path), convert_traditional)
        elif args.format == "vtt":
            write_vtt_output(segments, str(output_path), convert_traditional)
        elif args.format == "json":
            write_json_output(segments, str(output_path), metadata, convert_traditional)

        # Print preview
        print("\n--- Preview ---")
        preview_text = " ".join(seg.text for seg in segments[:3])
        if convert_traditional:
            preview_text = convert_to_traditional(preview_text)
        print(preview_text[:200] + "..." if len(preview_text) > 200 else preview_text)
        print("---")

    finally:
        # Cleanup
        transcriber.cleanup()

        # Remove temp WAV if not keeping
        if temp_wav and not args.keep_wav:
            try:
                os.remove(temp_wav)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    exit(main())
