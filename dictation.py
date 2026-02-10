#!/usr/bin/env python3
"""
Dictation app using local SenseVoice model (offline STT).
Press hotkey to start/stop recording, text is pasted when you finish speaking.
"""

import argparse
import asyncio
import os
import subprocess
import threading
import time
import wave
import tempfile
from typing import Optional

import numpy as np
import opencc
import pyperclip
import sounddevice as sd
from pynput.keyboard import Controller, Key

# QuickMacHotKey for global hotkey interception (blocks keypress from reaching other apps)
from quickmachotkey import quickHotKey, mask
from quickmachotkey.constants import (
    kVK_ANSI_D,
    kVK_RightShift,
    cmdKey,
    controlKey,
    optionKey,
)

# PyObjC imports for NSApplication + overlay
from AppKit import (
    NSApplication,
    NSBackingStoreBuffered,
    NSColor,
    NSEvent,
    NSEventMaskFlagsChanged,
    NSEventModifierFlagShift,
    NSPanel,
    NSScreen,
    NSStatusWindowLevel,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowCollectionBehaviorTransient,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSMakeRect, NSObject
from PyObjCTools import AppHelper

# Configuration
TRIGGER_KEY = "d"  # The key to press with hyper key
SAMPLE_RATE = 16000  # 16kHz required by SenseVoice
CHANNELS = 1  # Mono
TRANSCRIPT_DEBUG_LOG = os.getenv("DICTATION_TRANSCRIPT_DEBUG", "1") != "0"

# Sound effects (macOS system sounds)
SOUND_START = "/System/Library/Sounds/Pop.aiff"
SOUND_STOP = "/System/Library/Sounds/Tink.aiff"
SOUND_START_VOLUME = 0.25
SOUND_STOP_VOLUME = 0.25

# Global state
event_loop = None  # Store reference to the event loop
async_loop_ready = threading.Event()  # Signals when async loop is initialized
status_overlay = None


def play_sound(sound_path, volume=0.25):
    """Play a system sound asynchronously (non-blocking)"""
    try:
        safe_volume = max(0.0, min(1.0, float(volume)))
        subprocess.Popen(
            ["afplay", "-v", str(safe_volume), sound_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # Silently fail if sound can't be played


def set_overlay_recording():
    global status_overlay
    if status_overlay:
        AppHelper.callAfter(status_overlay.show_recording)


def set_overlay_finalizing():
    global status_overlay
    if status_overlay:
        AppHelper.callAfter(status_overlay.show_finalizing)


def hide_overlay():
    global status_overlay
    if status_overlay:
        AppHelper.callAfter(status_overlay.hide)


def _log_preview_text(text: str, limit: int = 220) -> str:
    """Build a single-line preview for transcript debug logs."""
    safe = text.replace("\n", "\\n")
    if len(safe) <= limit:
        return safe
    return safe[: limit - 3] + "..."


def log_transcript_stage(session_id: int, stage: str, text: str):
    """Print transcript text after each processing stage for debugging."""
    if not TRANSCRIPT_DEBUG_LOG:
        return
    preview = _log_preview_text(text)
    print(f"📊 [session {session_id}] {stage} (len={len(text)}): {preview}")


class StatusOverlay(NSObject):
    """Small always-on-top circular indicator for dictation state."""

    WIDTH = 30
    HEIGHT = 30
    TOP_MARGIN = 18
    DOT_SIZE = 10

    def init(self):
        self = super().init()
        if self is None:
            return None

        self.panel = None
        self.dot_view = None
        self._create_panel()
        return self

    def _screen_rect(self):
        point = NSEvent.mouseLocation()
        screen = None
        for candidate in NSScreen.screens():
            frame = candidate.frame()
            min_x = frame.origin.x
            min_y = frame.origin.y
            max_x = frame.origin.x + frame.size.width
            max_y = frame.origin.y + frame.size.height
            if min_x <= point.x <= max_x and min_y <= point.y <= max_y:
                screen = candidate
                break

        if screen is None:
            screen = NSScreen.mainScreen()

        if screen is None:
            return NSMakeRect(40, 40, self.WIDTH, self.HEIGHT)
        frame = screen.visibleFrame()
        x = frame.origin.x + (frame.size.width - self.WIDTH) / 2
        y = frame.origin.y + frame.size.height - self.HEIGHT - self.TOP_MARGIN
        return NSMakeRect(x, y, self.WIDTH, self.HEIGHT)

    def _create_panel(self):
        frame = self._screen_rect()
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        self.panel.setFloatingPanel_(True)
        self.panel.setLevel_(NSStatusWindowLevel)
        self.panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorTransient
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorIgnoresCycle
        )
        self.panel.setOpaque_(False)
        self.panel.setHasShadow_(True)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.setHidesOnDeactivate_(False)

        content = self.panel.contentView()
        content.setWantsLayer_(True)
        layer = content.layer()
        layer.setCornerRadius_(self.WIDTH / 2)
        layer.setMasksToBounds_(True)
        layer.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.02, 0.02, 0.03, 0.96
            ).CGColor()
        )

        dot_x = (self.WIDTH - self.DOT_SIZE) / 2
        dot_y = (self.HEIGHT - self.DOT_SIZE) / 2
        self.dot_view = NSView.alloc().initWithFrame_(
            NSMakeRect(dot_x, dot_y, self.DOT_SIZE, self.DOT_SIZE)
        )
        self.dot_view.setWantsLayer_(True)
        dot_layer = self.dot_view.layer()
        dot_layer.setCornerRadius_(self.DOT_SIZE / 2)
        dot_layer.setBackgroundColor_(NSColor.systemRedColor().CGColor())
        content.addSubview_(self.dot_view)

        self.hide()

    def _set_dot_color(self, color):
        if not self.dot_view:
            return
        dot_layer = self.dot_view.layer()
        if dot_layer:
            dot_layer.setBackgroundColor_(color.CGColor())

    def show_recording(self):
        if not self.panel:
            return
        self._set_dot_color(NSColor.systemRedColor())
        self.panel.setFrame_display_(self._screen_rect(), True)
        self.panel.orderFrontRegardless()

    def show_finalizing(self):
        if not self.panel:
            return
        self._set_dot_color(NSColor.systemOrangeColor())
        self.panel.setFrame_display_(self._screen_rect(), True)
        self.panel.orderFrontRegardless()

    def hide(self):
        if self.panel:
            self.panel.orderOut_(None)


def paste_text(text):
    """Paste text using clipboard (much faster than typing)"""
    try:
        # Save current clipboard
        old_clipboard = pyperclip.paste()

        # Copy text to clipboard
        pyperclip.copy(text)

        # Simulate Cmd+V to paste
        keyboard_controller = Controller()
        keyboard_controller.press(Key.cmd)
        keyboard_controller.press("v")
        keyboard_controller.release("v")
        keyboard_controller.release(Key.cmd)

        # Delay before restoring clipboard (gives time for paste and clipboard managers)
        time.sleep(0.6)

        # Restore old clipboard
        pyperclip.copy(old_clipboard)
    except Exception:
        # Fallback to typing if paste fails
        keyboard_controller = Controller()
        keyboard_controller.type(text)


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


# Available models
MODELS = {
    "small": "iic/SenseVoiceSmall",
    "sensevoice": "iic/SenseVoiceSmall",
    "paraformer": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-large": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-zh": "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1",
    "nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "fun-asr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
}

# Get the directory where this script is located (for remote_code)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_sensevoice_model(model_size="small", force_device=None):
    """Load the ASR model (SenseVoice, Paraformer, or Fun-ASR-Nano)"""
    print("Importing libraries...", end=" ", flush=True)
    from funasr import AutoModel
    import torch
    print("done")

    model_id = MODELS.get(model_size, model_size)
    model_name = model_id.split("/")[-1]
    download_size = "~2GB" if "Large" in model_id else "~800MB"

    # Detect model type
    is_sensevoice = "sensevoice" in model_id.lower()
    is_paraformer = "paraformer" in model_id.lower()
    is_fun_asr_nano = "fun-asr" in model_id.lower() or "funaudiollm" in model_id.lower()

    # Detect best device
    if force_device:
        device = force_device
        print(f"Device: {device} (forced)")
    elif torch.backends.mps.is_available():
        if is_paraformer:
            print("Warning: Paraformer has known MPS compatibility issues.")
            print("         If you experience hangs, try: --device cpu")
        if is_fun_asr_nano:
            print("Warning: Fun-ASR-Nano MPS compatibility is untested.")
            print("         If you experience issues, try: --device cpu")
        device = "mps"
        print("Device: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda:0"
        print("Device: NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print("Device: CPU")

    print(f"Model: {model_name}")
    print(f"Loading model... (first run downloads {download_size})", flush=True)

    # Build model kwargs based on model type
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

    model = AutoModel(**model_kwargs)

    print("Model loaded!")
    return model, is_fun_asr_nano


class DictationApp:
    def __init__(self, chinese="tw", model=None, is_fun_asr_nano=False):
        self.is_recording = False
        self.audio_chunks = []
        self.recording_thread = None
        self.stop_event = threading.Event()
        self.is_fun_asr_nano = is_fun_asr_nano
        self.session_id = 0
        self.session_lock = asyncio.Lock()
        self.paste_lock = asyncio.Lock()
        self.finalize_task: Optional[asyncio.Task] = None

        # Initialize Chinese character converter
        self.chinese_variant = chinese
        if chinese == "tw":
            self.chinese_converter = opencc.OpenCC("s2t")
        else:
            self.chinese_converter = opencc.OpenCC("t2s")

        # Use provided model or load new one
        if model is None:
            model, self.is_fun_asr_nano = load_sensevoice_model()
        self.model = model

        print("\nDictation App Ready!")
        print(
            f"Chinese output: {'Traditional (TW)' if chinese == 'tw' else 'Simplified (CN)'}"
        )
        print(f"Press Cmd+Option+Control+{TRIGGER_KEY.upper()} to start/stop recording")
        print("(Or press your Hyper Key + D if you have it configured)\n")

    def record_audio_thread(self):
        """Record audio in background thread"""
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16
        ) as stream:
            while not self.stop_event.is_set():
                data, _ = stream.read(int(SAMPLE_RATE * 0.1))  # Read 0.1 seconds
                self.audio_chunks.append(data.copy())

    async def start_recording(self):
        """Start recording audio"""
        async with self.session_lock:
            if self.is_recording:
                return

            self.is_recording = True
            self.session_id += 1
            self.audio_chunks = []
            self.stop_event.clear()

        # Play start sound and show overlay
        play_sound(SOUND_START, SOUND_START_VOLUME)
        set_overlay_recording()

        print("\n🎙️  Recording started... Speak now!")

        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio_thread)
        self.recording_thread.start()

    async def stop_recording(self):
        """Stop recording — returns instantly, transcription runs in background"""
        async with self.session_lock:
            if not self.is_recording:
                return

            current_session = self.session_id
            self.is_recording = False

            # Signal recording thread to stop
            self.stop_event.set()

            # Play stop sound, switch overlay to finalizing
            play_sound(SOUND_STOP, SOUND_STOP_VOLUME)
            set_overlay_finalizing()
            print("\n🛑 Recording stopped. Transcribing...")

            # Capture references for background finalization
            old_thread = self.recording_thread
            old_chunks = self.audio_chunks

            # Clear state so a new session can start immediately
            self.recording_thread = None
            self.audio_chunks = []

        # Kick off transcription in background (does NOT block the event loop)
        self.finalize_task = asyncio.create_task(
            self._finalize_session(old_thread, old_chunks, current_session)
        )

    async def _finalize_session(
        self,
        recording_thread: Optional[threading.Thread],
        audio_chunks: list,
        session_id: int,
    ):
        """Join recording thread, transcribe, convert, and paste — runs as background task."""
        try:
            # Join the recording thread without blocking the event loop
            if recording_thread:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, recording_thread.join)

            if not audio_chunks:
                print("No audio recorded")
                return

            # Concatenate audio
            audio_data = np.concatenate(audio_chunks, axis=0)
            duration = len(audio_data) / SAMPLE_RATE
            print(f"Audio length: {duration:.1f} seconds")

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                with wave.open(temp_path, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_data.tobytes())

            # Transcribe
            try:
                text = await self.transcribe(temp_path)
                if text:
                    await self._process_final_transcript(text, session_id)
                else:
                    print("Could not recognize speech")
            finally:
                os.unlink(temp_path)

        except Exception as e:
            print(f"⚠️  Error during finalization (session {session_id}): {e}")
        finally:
            hide_overlay()
            self.finalize_task = None

    async def transcribe(self, audio_path):
        """Transcribe audio using the loaded model"""
        import torch

        # Fix for FunASR thread count drift bug (GitHub Issues #2652, #2770)
        torch.set_num_threads(4)

        loop = asyncio.get_running_loop()

        if self.is_fun_asr_nano:
            res = await loop.run_in_executor(
                None,
                lambda: self.model.generate(
                    input=[audio_path],
                    cache={},
                    batch_size=1,
                    language="auto",
                    itn=True,
                ),
            )
        else:
            res = await loop.run_in_executor(
                None,
                lambda: self.model.generate(
                    input=audio_path,
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                ),
            )

        # Reset thread count after inference to prevent drift
        torch.set_num_threads(4)

        if res and res[0].get("text"):
            text = res[0]["text"]
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                return rich_transcription_postprocess(text)
            except Exception:
                return text
        return ""

    async def _process_final_transcript(self, text: str, session_id: int):
        """Process final transcript: convert characters, paste."""
        async with self.paste_lock:
            log_transcript_stage(session_id, "input.raw", text)

            # OpenCC conversion (sync, fast)
            converted_text = self.chinese_converter.convert(text)
            log_transcript_stage(session_id, "after.opencc", converted_text)

            # Paste
            paste_text(converted_text)
            log_transcript_stage(session_id, "paste.output", converted_text)
            print(f"\n✅ Pasted: {converted_text}\n")

    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_event.set()
            if self.recording_thread:
                self.recording_thread.join()


# Global app instance
app = None
right_shift_ptt_monitor = None
right_shift_ptt_enabled = True


class RightShiftPTTMonitor:
    """Global right-shift push-to-talk monitor."""

    def __init__(self):
        self.monitor_token = None
        self.right_shift_down = False
        self.started_session = False

    def start(self):
        if self.monitor_token is not None:
            return
        self.monitor_token = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSEventMaskFlagsChanged,
            self._handle_flags_changed,
        )
        print("Right Shift push-to-talk enabled (hold to record)")

    def stop(self):
        if self.monitor_token is not None:
            NSEvent.removeMonitor_(self.monitor_token)
            self.monitor_token = None
        self.right_shift_down = False
        self.started_session = False

    async def _stop_when_possible(self):
        """Stop recording after a right-shift release, even if start is still in flight."""
        global app
        for _ in range(25):
            if self.right_shift_down:
                return
            if app and app.is_recording:
                await app.stop_recording()
                return
            await asyncio.sleep(0.02)

    def _handle_flags_changed(self, event):
        """Handle global modifier changes and detect right-shift hold/release."""
        global app, event_loop

        try:
            if event.keyCode() != kVK_RightShift:
                return

            is_down = bool(event.modifierFlags() & NSEventModifierFlagShift)
            if is_down == self.right_shift_down:
                return

            self.right_shift_down = is_down

            if not app or not event_loop:
                return

            if is_down:
                if not app.is_recording:
                    self.started_session = True
                    asyncio.run_coroutine_threadsafe(app.start_recording(), event_loop)
                else:
                    self.started_session = False
            else:
                if self.started_session:
                    self.started_session = False
                    asyncio.run_coroutine_threadsafe(
                        self._stop_when_possible(),
                        event_loop,
                    )
        except Exception as e:
            print(f"⚠️  Right Shift PTT monitor error: {e}")


# Global hotkey handler using QuickMacHotKey
@quickHotKey(virtualKey=kVK_ANSI_D, modifierMask=mask(cmdKey, controlKey, optionKey))
def handle_hotkey():
    """
    Handle the global hotkey Cmd+Option+Control+D.
    QuickMacHotKey automatically consumes the keypress, preventing it from reaching other apps.
    """
    global app, event_loop

    if app and event_loop:
        if not app.is_recording:
            asyncio.run_coroutine_threadsafe(app.start_recording(), event_loop)
        else:
            asyncio.run_coroutine_threadsafe(app.stop_recording(), event_loop)


class AppDelegate(NSObject):
    """Simple app delegate for NSApplication."""

    def applicationDidFinishLaunching_(self, notification):
        """Set up when app finishes launching."""
        global right_shift_ptt_monitor, right_shift_ptt_enabled, status_overlay

        status_overlay = StatusOverlay.alloc().init()

        if right_shift_ptt_enabled:
            right_shift_ptt_monitor = RightShiftPTTMonitor()
            right_shift_ptt_monitor.start()

        print("Hotkey monitor started. Press Cmd+Option+Control+D to toggle recording.")
        if right_shift_ptt_enabled:
            print("Hold Right Shift for push-to-talk.")
        print("(QuickMacHotKey will intercept the keypress - terminal won't see it)")
        print("Press Ctrl+C to exit.\n")


def setup_async_loop(chinese, model, is_fun_asr_nano):
    """Set up the async event loop in a separate thread."""
    global app, event_loop

    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    event_loop = loop

    # Create app instance (model already loaded)
    app = DictationApp(chinese=chinese, model=model, is_fun_asr_nano=is_fun_asr_nano)

    # Signal that initialization is complete
    async_loop_ready.set()

    # Run the event loop forever
    loop.run_forever()


def start_app(chinese="tw", model_size="small", device=None, enable_right_shift_ptt=True):
    """Start the application with NSApplication event loop."""
    global right_shift_ptt_enabled, right_shift_ptt_monitor

    right_shift_ptt_enabled = enable_right_shift_ptt
    right_shift_ptt_monitor = None

    # Load model in main thread so user sees progress
    print("=" * 50)
    print("  Local Dictation (SenseVoice / Paraformer / Fun-ASR-Nano)")
    print("=" * 50)
    model, is_fun_asr_nano = load_sensevoice_model(model_size, force_device=device)

    # Start asyncio event loop in a separate thread
    async_thread = threading.Thread(
        target=setup_async_loop, args=(chinese, model, is_fun_asr_nano), daemon=True
    )
    async_thread.start()

    # Wait for the async thread to initialize
    async_loop_ready.wait()

    # Create the NSApplication
    ns_app = NSApplication.sharedApplication()

    # Create and set the delegate
    delegate = AppDelegate.alloc().init()
    ns_app.setDelegate_(delegate)

    # Run the NSApplication event loop (blocks until app quits)
    try:
        AppHelper.runEventLoop()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if app and app.is_recording:
            if event_loop:
                asyncio.run_coroutine_threadsafe(app.stop_recording(), event_loop)
                time.sleep(1)  # Give time for cleanup
    finally:
        if right_shift_ptt_monitor:
            right_shift_ptt_monitor.stop()
        hide_overlay()
        if app:
            app.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dictation app using local ASR models (offline STT)"
    )
    parser.add_argument(
        "--chinese",
        choices=["tw", "cn"],
        default="tw",
        help="Chinese character variant: tw (Traditional, default) or cn (Simplified)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="small",
        help="Model: small/sensevoice (default), paraformer (Chinese), nano/fun-asr-nano (31 langs + dialects)",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Force device: cpu, mps, or cuda:0 (default: auto-detect)",
    )
    parser.add_argument(
        "--disable-right-shift-ptt",
        action="store_true",
        help="Disable hold-to-talk on Right Shift (Cmd+Option+Control+D toggle stays enabled)",
    )
    args = parser.parse_args()
    start_app(
        chinese=args.chinese,
        model_size=args.model,
        device=args.device,
        enable_right_shift_ptt=not args.disable_right_shift_ptt,
    )
