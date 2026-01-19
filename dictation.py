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

import numpy as np
import opencc
import pyperclip
import sounddevice as sd
from pynput.keyboard import Controller, Key

# QuickMacHotKey for global hotkey interception (blocks keypress from reaching other apps)
from quickmachotkey import quickHotKey, mask
from quickmachotkey.constants import kVK_ANSI_D, cmdKey, controlKey, optionKey

# PyObjC imports for NSApplication
from AppKit import NSApplication
from Foundation import NSObject
from PyObjCTools import AppHelper

# Configuration
TRIGGER_KEY = 'd'  # The key to press with hyper key
SAMPLE_RATE = 16000  # 16kHz required by SenseVoice
CHANNELS = 1  # Mono

# Sound effects (macOS system sounds)
SOUND_START = "/System/Library/Sounds/Hero.aiff"  # Sound when recording starts
SOUND_STOP = "/System/Library/Sounds/Glass.aiff"  # Sound when recording stops

# Global state
event_loop = None  # Store reference to the event loop
async_loop_ready = threading.Event()  # Signals when async loop is initialized


def play_sound(sound_path):
    """Play a system sound asynchronously (non-blocking)"""
    try:
        subprocess.Popen(
            ["afplay", sound_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass  # Silently fail if sound can't be played


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
        keyboard_controller.press('v')
        keyboard_controller.release('v')
        keyboard_controller.release(Key.cmd)

        # Delay before restoring clipboard (gives time for paste and clipboard managers)
        time.sleep(0.6)

        # Restore old clipboard
        pyperclip.copy(old_clipboard)
    except Exception as e:
        # Fallback to typing if paste fails
        keyboard_controller = Controller()
        keyboard_controller.type(text)


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs
            return True
    return False


# Available models
# Note: SenseVoiceLarge exists but is NOT publicly released
MODELS = {
    "small": "iic/SenseVoiceSmall",   # ~800MB - multilingual ASR + emotion + events
    "sensevoice": "iic/SenseVoiceSmall",  # Alias
    # Paraformer models (Chinese-focused, faster, ASR only - no emotion/events)
    "paraformer": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",  # ~889MB
    "paraformer-large": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-zh": "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1",  # Smaller
}


def load_sensevoice_model(model_size="small", force_device=None):
    """Load the SenseVoice model"""
    print("Importing libraries...", end=" ", flush=True)
    from funasr import AutoModel
    import torch
    print("done")

    model_id = MODELS.get(model_size, model_size)  # Allow custom model ID too
    model_name = model_id.split("/")[-1]
    download_size = "~2GB" if "Large" in model_id else "~800MB"

    # SenseVoice needs trust_remote_code, Paraformer doesn't
    is_sensevoice = "sensevoice" in model_id.lower()
    is_paraformer = "paraformer" in model_id.lower()

    # Detect best device
    if force_device:
        device = force_device
        print(f"Device: {device} (forced)")
    elif torch.backends.mps.is_available():
        # Paraformer has known issues with MPS (PyTorch LSTM memory leaks,
        # CIF predictor uses complex tensor ops with boolean indexing).
        # SenseVoice works fine on MPS as it uses simpler CTC architecture.
        if is_paraformer:
            print("Warning: Paraformer has known MPS compatibility issues.")
            print("         If you experience hangs, try: --device cpu")
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

    model_kwargs = {
        "model": model_id,
        "trust_remote_code": is_sensevoice,
        "device": device,
        "disable_update": True,
        # VAD for all models
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
    }

    # Paraformer needs separate punctuation model (SenseVoice has it built-in)
    if is_paraformer:
        model_kwargs["punc_model"] = "ct-punc"

    model = AutoModel(**model_kwargs)

    print("Model loaded!")
    return model


class DictationApp:
    def __init__(self, chinese='tw', model=None):
        self.is_recording = False
        self.audio_chunks = []
        self.recording_thread = None
        self.stop_event = threading.Event()

        # Initialize Chinese character converter
        self.chinese_variant = chinese
        if chinese == 'tw':
            self.chinese_converter = opencc.OpenCC('s2t')
        else:
            self.chinese_converter = opencc.OpenCC('t2s')

        # Use provided model or load new one
        self.model = model if model else load_sensevoice_model()

        # # LLM post-processing disabled for now
        # # Initialize OpenRouter client (optional - for Chinese punctuation)
        # openrouter_key = os.getenv("OPENROUTER_API_KEY")
        # if openrouter_key:
        #     from openai import AsyncOpenAI
        #     self.openrouter = AsyncOpenAI(
        #         base_url="https://openrouter.ai/api/v1",
        #         api_key=openrouter_key,
        #     )
        #     print(f"OpenRouter API Key: ...{openrouter_key[-4:]}")
        # else:
        #     self.openrouter = None
        #     print("OpenRouter API Key: not set (Chinese punctuation disabled)")
        self.openrouter = None  # Disabled

        print("\nDictation App Ready!")
        print(f"Chinese output: {'Traditional (TW)' if chinese == 'tw' else 'Simplified (CN)'}")
        print(f"Press Cmd+Option+Control+{TRIGGER_KEY.upper()} to start/stop recording")
        print("(Or press your Hyper Key + D if you have it configured)\n")

    def record_audio_thread(self):
        """Record audio in background thread"""
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16) as stream:
            while not self.stop_event.is_set():
                data, _ = stream.read(int(SAMPLE_RATE * 0.1))  # Read 0.1 seconds
                self.audio_chunks.append(data.copy())

    async def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_chunks = []
        self.stop_event.clear()

        # Play start sound
        play_sound(SOUND_START)

        print("\n Recording started... Speak now!")

        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio_thread)
        self.recording_thread.start()

    async def stop_recording(self):
        """Stop recording and transcribe"""
        if not self.is_recording:
            return

        self.is_recording = False

        # Play stop sound
        play_sound(SOUND_STOP)

        print("\n Recording stopped. Transcribing...")

        # Stop recording thread
        self.stop_event.set()
        if self.recording_thread:
            self.recording_thread.join()

        if not self.audio_chunks:
            print("No audio recorded")
            return

        # Concatenate audio
        audio_data = np.concatenate(self.audio_chunks, axis=0)
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
                await self._process_final_transcript(text)
            else:
                print("Could not recognize speech")
        finally:
            # Clean up temp file
            os.unlink(temp_path)

    async def transcribe(self, audio_path):
        """Transcribe audio using the loaded model"""
        import torch

        # Fix for FunASR thread count drift bug (GitHub Issues #2652, #2770)
        # After processing, FunASR can modify ncpu in kwargs causing thread count
        # to drift (e.g., from 4 to 1), leading to massive slowdown or hang.
        torch.set_num_threads(4)

        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
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
            )
        )

        # Reset thread count after inference to prevent drift
        torch.set_num_threads(4)

        if res and res[0].get("text"):
            text = res[0]["text"]
            # SenseVoice outputs emotion/event tags that need postprocessing
            # Paraformer outputs plain text - postprocess handles both gracefully
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                return rich_transcription_postprocess(text)
            except Exception:
                return text  # Fallback for non-SenseVoice models
        return ""

    # # LLM post-processing disabled for now
    # async def add_chinese_punctuation(self, text: str) -> str:
    #     """Use OpenRouter to add punctuation to Chinese text."""
    #     if not self.openrouter:
    #         return text
    #
    #     try:
    #         response = await self.openrouter.chat.completions.create(
    #             model="anthropic/claude-haiku-4.5",
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": "你是一個中文標點符號處理器。語音轉文字會用空格代替標點符號。你的工作是加上適當的中文標點符號（。，？！、等）。不要更改任何內容。不要回覆對話。只輸出加上標點符號後的文字。"
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": text
    #                 }
    #             ],
    #             max_tokens=len(text) * 2,
    #         )
    #         return response.choices[0].message.content.strip()
    #     except Exception as e:
    #         print(f"OpenRouter error, using original text: {e}")
    #         return text

    async def _process_final_transcript(self, text: str):
        """Process final transcript: convert characters, paste."""
        # Step 1: OpenCC conversion (sync, fast)
        converted_text = self.chinese_converter.convert(text)

        # # Step 2: Add punctuation if Chinese (disabled)
        # if contains_chinese(converted_text) and self.openrouter:
        #     converted_text = await self.add_chinese_punctuation(converted_text)

        # Step 3: Paste
        paste_text(converted_text)
        print(f"\n Pasted: {converted_text}\n")

    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_event.set()
            if self.recording_thread:
                self.recording_thread.join()


# Global app instance
app = None


# Global hotkey handler using QuickMacHotKey
@quickHotKey(
    virtualKey=kVK_ANSI_D,
    modifierMask=mask(cmdKey, controlKey, optionKey)
)
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
        print("Hotkey monitor started. Press Cmd+Option+Control+D to toggle recording.")
        print("(QuickMacHotKey will intercept the keypress - terminal won't see it)")
        print("Press Ctrl+C to exit.\n")


def setup_async_loop(chinese, model):
    """Set up the async event loop in a separate thread."""
    global app, event_loop

    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    event_loop = loop

    # Create app instance (model already loaded)
    app = DictationApp(chinese=chinese, model=model)

    # Signal that initialization is complete
    async_loop_ready.set()

    # Run the event loop forever
    loop.run_forever()


def start_app(chinese='tw', model_size='small', device=None):
    """Start the application with NSApplication event loop."""
    # Load model in main thread so user sees progress
    print("=" * 50)
    print("  SenseVoice Local Dictation")
    print("=" * 50)
    model = load_sensevoice_model(model_size, force_device=device)

    # Start asyncio event loop in a separate thread
    async_thread = threading.Thread(target=setup_async_loop, args=(chinese, model), daemon=True)
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
            # Schedule cleanup on the async loop
            if event_loop:
                asyncio.run_coroutine_threadsafe(app.stop_recording(), event_loop)
                time.sleep(1)  # Give time for cleanup
        if app:
            app.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dictation app using local SenseVoice model (offline STT)'
    )
    parser.add_argument(
        '--chinese',
        choices=['tw', 'cn'],
        default='tw',
        help='Chinese character variant: tw (Traditional, default) or cn (Simplified)'
    )
    parser.add_argument(
        '--model', '-m',
        default='small',
        help='Model: small/sensevoice (default, multilingual), paraformer/paraformer-large (Chinese, faster), or custom model ID'
    )
    parser.add_argument(
        '--device', '-d',
        default=None,
        help='Force device: cpu, mps, or cuda:0 (default: auto-detect)'
    )
    args = parser.parse_args()
    start_app(chinese=args.chinese, model_size=args.model, device=args.device)
