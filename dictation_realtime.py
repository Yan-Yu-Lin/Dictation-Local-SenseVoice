#!/usr/bin/env python3
"""
Realtime meeting transcriber using local FunASR models.

Behavior:
- Continuously listens from the microphone.
- Uses a lightweight frame-level VAD endpoint detector.
- When speech ends (enough silence), saves a WAV segment and transcribes it.
- Prints live transcript in terminal and writes transcript files to disk.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
import queue
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import opencc
import sounddevice as sd


SAMPLE_RATE = 16000
CHANNELS = 1


MODELS = {
    "small": "iic/SenseVoiceSmall",
    "sensevoice": "iic/SenseVoiceSmall",
    "paraformer": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-large": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-zh": "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1",
    "nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "fun-asr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
}


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@contextlib.contextmanager
def suppress_backend_output(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def format_hms(seconds: float) -> str:
    whole = int(seconds)
    ms = int((seconds - whole) * 1000)
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())


def load_asr_model(
    model_size: str = "small",
    force_device: str | None = None,
    verbose: bool = True,
    quiet_backend: bool = False,
):
    if verbose:
        print("Importing libraries...", end=" ", flush=True)
    with suppress_backend_output(quiet_backend):
        from funasr import AutoModel
        import torch

    if verbose:
        print("done")

    model_id = MODELS.get(model_size, model_size)
    model_name = model_id.split("/")[-1]
    download_size = "~2GB" if "Large" in model_id else "~800MB"

    is_sensevoice = "sensevoice" in model_id.lower()
    is_paraformer = "paraformer" in model_id.lower()
    is_fun_asr_nano = "fun-asr" in model_id.lower() or "funaudiollm" in model_id.lower()

    if force_device:
        device = force_device
        if verbose:
            print(f"Device: {device} (forced)")
    elif torch.backends.mps.is_available():
        if is_paraformer:
            if verbose:
                print(
                    "Warning: Paraformer can have MPS issues. Use --device cpu if needed."
                )
        if is_fun_asr_nano:
            if verbose:
                print(
                    "Warning: Fun-ASR-Nano on MPS is not fully validated. Use --device cpu if needed."
                )
        device = "mps"
        if verbose:
            print("Device: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda:0"
        if verbose:
            print("Device: NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        if verbose:
            print("Device: CPU")

    if verbose:
        print(f"Model: {model_name}")
        print(f"Loading model... (first run downloads {download_size})", flush=True)

    if is_fun_asr_nano:
        remote_code_path = os.path.join(SCRIPT_DIR, "models", "fun_asr_nano.py")
        model_kwargs = {
            "model": model_id,
            "trust_remote_code": True,
            "remote_code": remote_code_path,
            "device": device,
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
        }
    else:
        model_kwargs = {
            "model": model_id,
            "trust_remote_code": is_sensevoice,
            "device": device,
            "disable_update": True,
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
        }
        if is_paraformer:
            model_kwargs["punc_model"] = "ct-punc"

    with suppress_backend_output(quiet_backend):
        model = AutoModel(**model_kwargs)
    if verbose:
        print("Model loaded!")
    return model, is_fun_asr_nano


def transcribe_file(
    model,
    is_fun_asr_nano: bool,
    audio_path: Path,
    language: str = "auto",
    quiet_backend: bool = False,
) -> str:
    import torch

    torch.set_num_threads(4)
    with suppress_backend_output(quiet_backend):
        if is_fun_asr_nano:
            res = model.generate(
                input=[str(audio_path)],
                cache={},
                batch_size=1,
                language=language,
                itn=True,
            )
        else:
            res = model.generate(
                input=str(audio_path),
                cache={},
                language=language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
    torch.set_num_threads(4)

    if not (res and res[0].get("text")):
        return ""

    text = res[0]["text"]
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return rich_transcription_postprocess(text)
    except Exception:
        return text


class AdaptiveEnergyVAD:
    def __init__(
        self,
        speech_margin_db: float = 10.0,
        absolute_threshold_db: float = -42.0,
        init_noise_floor_db: float = -60.0,
        noise_alpha: float = 0.97,
    ):
        self.speech_margin_db = speech_margin_db
        self.absolute_threshold_db = absolute_threshold_db
        self.noise_floor_db = init_noise_floor_db
        self.noise_alpha = noise_alpha

    @staticmethod
    def frame_db(frame: np.ndarray) -> float:
        x = frame.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(x * x) + 1e-12)
        return float(20.0 * np.log10(rms + 1e-12))

    def is_speech(self, frame: np.ndarray) -> tuple[bool, float, float]:
        level_db = self.frame_db(frame)
        threshold = max(
            self.noise_floor_db + self.speech_margin_db, self.absolute_threshold_db
        )
        speech = level_db >= threshold

        if not speech:
            self.noise_floor_db = (
                self.noise_alpha * self.noise_floor_db
                + (1.0 - self.noise_alpha) * level_db
            )

        return speech, level_db, threshold


@dataclass
class Segment:
    index: int
    start_sec: float
    end_sec: float
    audio: np.ndarray


class EndpointDetector:
    def __init__(
        self,
        sample_rate: int,
        frame_ms: int,
        min_speech_ms: int,
        start_trigger_ms: int,
        silence_ms: int,
        pre_roll_ms: int,
        max_segment_s: float,
        vad: AdaptiveEnergyVAD,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)

        self.min_frames = max(1, int(min_speech_ms / frame_ms))
        self.start_trigger_frames = max(1, int(start_trigger_ms / frame_ms))
        self.silence_frames = max(1, int(silence_ms / frame_ms))
        self.pre_roll_frames = max(1, int(pre_roll_ms / frame_ms))
        self.max_frames = max(1, int(max_segment_s * 1000 / frame_ms))
        self.tail_keep_frames = max(1, int(200 / frame_ms))

        self.vad = vad

        self.total_frames = 0
        self.segment_counter = 0
        self.pre_roll: deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)

        self.in_speech = False
        self.trigger_count = 0
        self.current_frames: list[np.ndarray] = []
        self.current_start_frame = 0
        self.current_voice_frames = 0
        self.current_silence_frames = 0

    def _make_segment(self, keep_frames: int) -> Segment | None:
        if keep_frames <= 0:
            return None

        frames = self.current_frames[:keep_frames]
        if not frames:
            return None

        audio = np.concatenate(frames, axis=0).astype(np.int16)
        start_sec = self.current_start_frame * self.frame_ms / 1000.0
        end_sec = (self.current_start_frame + keep_frames) * self.frame_ms / 1000.0
        self.segment_counter += 1
        return Segment(
            index=self.segment_counter,
            start_sec=start_sec,
            end_sec=end_sec,
            audio=audio,
        )

    def _reset_current(self) -> None:
        self.in_speech = False
        self.trigger_count = 0
        self.current_frames = []
        self.current_start_frame = 0
        self.current_voice_frames = 0
        self.current_silence_frames = 0

    def process_frame(
        self, frame: np.ndarray
    ) -> tuple[list[Segment], bool, float, float]:
        segments: list[Segment] = []
        speech, level_db, threshold_db = self.vad.is_speech(frame)

        if not self.in_speech:
            self.pre_roll.append(frame.copy())

            if speech:
                self.trigger_count += 1
            else:
                self.trigger_count = max(0, self.trigger_count - 1)

            if self.trigger_count >= self.start_trigger_frames:
                self.in_speech = True
                self.current_frames = list(self.pre_roll)
                self.current_start_frame = max(
                    0, self.total_frames - len(self.current_frames) + 1
                )
                self.current_voice_frames = 1
                self.current_silence_frames = 0
        else:
            self.current_frames.append(frame.copy())

            if speech:
                self.current_voice_frames += 1
                self.current_silence_frames = 0
            else:
                self.current_silence_frames += 1

            should_end_by_silence = (
                self.current_voice_frames >= self.min_frames
                and self.current_silence_frames >= self.silence_frames
            )
            should_end_by_max_len = len(self.current_frames) >= self.max_frames

            if should_end_by_silence or should_end_by_max_len:
                if should_end_by_silence:
                    keep_frames = (
                        len(self.current_frames)
                        - self.current_silence_frames
                        + self.tail_keep_frames
                    )
                    keep_frames = min(max(1, keep_frames), len(self.current_frames))
                else:
                    keep_frames = len(self.current_frames)

                seg = self._make_segment(keep_frames)
                if seg is not None:
                    segments.append(seg)

                self.pre_roll.clear()
                self._reset_current()

        self.total_frames += 1
        return segments, speech, level_db, threshold_db

    def flush(self, force: bool = False) -> Segment | None:
        if not self.in_speech:
            return None

        if not force and self.current_voice_frames < self.min_frames:
            self._reset_current()
            return None

        seg = self._make_segment(len(self.current_frames))
        self._reset_current()
        return seg


@dataclass
class WorkerItem:
    segment: Segment


class RealtimeMeetingTranscriber:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.quiet_backend = args.display == "text"
        self.frame_samples = int(args.sample_rate * args.frame_ms / 1000)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4096)
        self.work_queue: queue.Queue[WorkerItem | None] = queue.Queue(maxsize=256)
        self.stop_event = threading.Event()

        self.model, self.is_fun_asr_nano = load_asr_model(
            args.model,
            args.device,
            verbose=(args.display != "text"),
            quiet_backend=self.quiet_backend,
        )
        self.converter = opencc.OpenCC("s2t" if args.chinese == "tw" else "t2s")

        session_name = (
            args.session_name
            or f"meeting_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.session_dir = Path(args.output_root).expanduser().resolve() / session_name
        self.audio_dir = self.session_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.transcript_path = self.session_dir / "transcript.txt"
        self.transcript_plain_path = self.session_dir / "transcript_plain.txt"
        self._transcript_lock = threading.Lock()
        self.session_started_at = dt.datetime.now()

        self.detector = EndpointDetector(
            sample_rate=args.sample_rate,
            frame_ms=args.frame_ms,
            min_speech_ms=args.min_speech_ms,
            start_trigger_ms=args.start_trigger_ms,
            silence_ms=args.silence_ms,
            pre_roll_ms=args.pre_roll_ms,
            max_segment_s=args.max_segment_s,
            vad=AdaptiveEnergyVAD(
                speech_margin_db=args.vad_margin_db,
                absolute_threshold_db=args.vad_abs_db,
                init_noise_floor_db=args.vad_init_noise_db,
            ),
        )

        with self.transcript_path.open("w", encoding="utf-8") as f:
            f.write("Realtime Meeting Transcript\n")
            f.write(
                f"Started: {self.session_started_at.isoformat(timespec='seconds')}\n"
            )
            f.write(f"Model: {args.model}\n")
            f.write(f"Sample rate: {args.sample_rate}\n")
            f.write("\n")

        with self.transcript_plain_path.open("w", encoding="utf-8") as f:
            f.write("")

    def _audio_callback(self, indata, frames, time_info, status):
        del time_info

        if status and self.args.display != "text":
            print(f"[audio] {status}")

        if frames != self.frame_samples:
            frame = indata[:, 0].copy().reshape(-1).astype(np.int16)
            if len(frame) < self.frame_samples:
                padded = np.zeros((self.frame_samples,), dtype=np.int16)
                padded[: len(frame)] = frame
                frame = padded
            else:
                frame = frame[: self.frame_samples]
        else:
            frame = indata[:, 0].copy().reshape(-1).astype(np.int16)

        try:
            self.audio_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _append_transcript(self, line: str) -> None:
        with self._transcript_lock:
            with self.transcript_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _append_plain_transcript(self, text: str) -> None:
        with self._transcript_lock:
            with self.transcript_plain_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n\n")

    def _worker(self) -> None:
        while True:
            item = self.work_queue.get()
            if item is None:
                self.work_queue.task_done()
                break

            seg = item.segment
            seg_name = f"seg_{seg.index:05d}.wav"
            audio_path = self.audio_dir / seg_name

            write_wav(audio_path, seg.audio, self.args.sample_rate)

            text = transcribe_file(
                model=self.model,
                is_fun_asr_nano=self.is_fun_asr_nano,
                audio_path=audio_path,
                language=self.args.language,
                quiet_backend=self.quiet_backend,
            ).strip()

            text = self.converter.convert(text) if text else ""
            time_range = f"[{format_hms(seg.start_sec)} -> {format_hms(seg.end_sec)}]"

            if text:
                line = f"{time_range} {text}"
                if self.args.display == "text":
                    print(text)
                    print("")
                else:
                    print(line)
                    print(f"  audio: {audio_path}")

                seg_txt_path = audio_path.with_suffix(".txt")
                seg_txt_path.write_text(text + "\n", encoding="utf-8")

                self._append_transcript(line)
                self._append_transcript(f"audio: {audio_path}")
                self._append_transcript("")
                self._append_plain_transcript(text)
            else:
                line = f"{time_range} (no speech recognized)"
                if self.args.display != "text":
                    print(line)
                    print(f"  audio: {audio_path}")
                self._append_transcript(line)
                self._append_transcript(f"audio: {audio_path}")
                self._append_transcript("")

            self.work_queue.task_done()

    def run(self) -> None:
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()

        if self.args.display != "text":
            print("=" * 70)
            print("Realtime meeting transcriber")
            print("=" * 70)
            print(f"Session folder: {self.session_dir}")
            print(f"Transcript file: {self.transcript_path}")
            print(f"Plain transcript: {self.transcript_plain_path}")
            print("Listening... press Ctrl+C to stop.")
            print("")

        try:
            with sd.InputStream(
                samplerate=self.args.sample_rate,
                channels=CHANNELS,
                dtype=np.int16,
                blocksize=self.frame_samples,
                callback=self._audio_callback,
            ):
                next_meter_at = time.monotonic() + 1.0
                while not self.stop_event.is_set():
                    try:
                        frame = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    segments, speech, level_db, threshold_db = (
                        self.detector.process_frame(frame)
                    )
                    for seg in segments:
                        self.work_queue.put(WorkerItem(segment=seg))

                    if self.args.show_vad and self.args.display != "text":
                        now = time.monotonic()
                        if now >= next_meter_at:
                            state = "speech" if speech else "silence"
                            print(
                                f"[vad] {state:<7} level={level_db:6.1f}dB "
                                f"th={threshold_db:6.1f}dB queue={self.work_queue.qsize()}"
                            )
                            next_meter_at = now + 1.0
        except KeyboardInterrupt:
            if self.args.display != "text":
                print("\nStopping... flushing remaining audio.")
        finally:
            self.stop_event.set()

            final_seg = self.detector.flush(force=True)
            if final_seg is not None:
                self.work_queue.put(WorkerItem(segment=final_seg))

            self.work_queue.put(None)
            worker.join()

            if self.args.display != "text":
                print("Done.")
                print(f"Session saved in: {self.session_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime meeting transcriber with VAD endpointing"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="small",
        help="Model alias or model ID (default: small)",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Force device: cpu, mps, cuda:0 (default: auto)",
    )
    parser.add_argument(
        "--chinese",
        choices=["tw", "cn"],
        default="tw",
        help="Chinese output variant: tw or cn (default: tw)",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="ASR language hint (default: auto)",
    )

    parser.add_argument(
        "--sample-rate", type=int, default=SAMPLE_RATE, help="Audio sample rate"
    )
    parser.add_argument("--frame-ms", type=int, default=30, help="Frame size in ms")
    parser.add_argument(
        "--start-trigger-ms",
        type=int,
        default=240,
        help="Speech start trigger duration",
    )
    parser.add_argument(
        "--silence-ms", type=int, default=900, help="Silence duration to end a segment"
    )
    parser.add_argument(
        "--min-speech-ms", type=int, default=350, help="Minimum speech duration"
    )
    parser.add_argument(
        "--pre-roll-ms", type=int, default=300, help="Audio pre-roll kept before speech"
    )
    parser.add_argument(
        "--max-segment-s", type=float, default=45.0, help="Hard max segment duration"
    )

    parser.add_argument(
        "--vad-margin-db",
        type=float,
        default=10.0,
        help="Speech margin over noise floor",
    )
    parser.add_argument(
        "--vad-abs-db", type=float, default=-42.0, help="Absolute VAD threshold in dB"
    )
    parser.add_argument(
        "--vad-init-noise-db",
        type=float,
        default=-60.0,
        help="Initial noise floor in dB",
    )
    parser.add_argument(
        "--show-vad", action="store_true", help="Print VAD meter every second"
    )
    parser.add_argument(
        "--display",
        choices=["full", "text"],
        default="full",
        help="Terminal output mode: full (default) or text (transcript only)",
    )

    parser.add_argument(
        "--output-root",
        default="recordings",
        help="Root folder for session outputs (default: recordings)",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Optional session folder name",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.sample_rate != 16000 and args.display != "text":
        print("Warning: 16kHz is recommended for these models.")

    app = RealtimeMeetingTranscriber(args)
    app.run()


if __name__ == "__main__":
    main()
