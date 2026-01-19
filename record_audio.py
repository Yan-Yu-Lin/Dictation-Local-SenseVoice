#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["sounddevice", "numpy", "quickmachotkey", "pyobjc-framework-Cocoa"]
# ///
"""
Audio recorder for ASR benchmarking.
Press hotkey to start/stop recording, audio is saved to ./recordings/ folder.
"""

import os
import subprocess
import threading
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd

# QuickMacHotKey for global hotkey interception
from quickmachotkey import quickHotKey, mask
from quickmachotkey.constants import kVK_ANSI_D, cmdKey, controlKey, optionKey

# PyObjC imports for NSApplication
from AppKit import NSApplication
from Foundation import NSObject
from PyObjCTools import AppHelper

# Configuration
SAMPLE_RATE = 16000  # 16kHz required by most ASR models
CHANNELS = 1  # Mono
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")

# Sound effects (macOS system sounds)
SOUND_START = "/System/Library/Sounds/Hero.aiff"
SOUND_STOP = "/System/Library/Sounds/Glass.aiff"


def play_sound(sound_path):
    """Play a system sound asynchronously (non-blocking)"""
    try:
        subprocess.Popen(
            ["afplay", sound_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.audio_chunks = []
        self.recording_thread = None
        self.stop_event = threading.Event()
        self.recording_start_time = None

        # Ensure recordings directory exists
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

        print("\nAudio Recorder Ready!")
        print(f"Recordings will be saved to: {RECORDINGS_DIR}")
        print("Press Cmd+Option+Control+D to start/stop recording")
        print("(Or press your Hyper Key + D if you have it configured)\n")

    def record_audio_thread(self):
        """Record audio in background thread"""
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16) as stream:
            while not self.stop_event.is_set():
                data, _ = stream.read(int(SAMPLE_RATE * 0.1))  # Read 0.1 seconds
                self.audio_chunks.append(data.copy())

    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_chunks = []
        self.stop_event.clear()
        self.recording_start_time = datetime.now()

        # Play start sound
        play_sound(SOUND_START)

        print("\n Recording started... Speak now!")

        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio_thread)
        self.recording_thread.start()

    def stop_recording(self):
        """Stop recording and save to file"""
        if not self.is_recording:
            return

        self.is_recording = False

        # Play stop sound
        play_sound(SOUND_STOP)

        print("\n Recording stopped. Saving...")

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

        # Generate filename with timestamp
        timestamp = self.recording_start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join(RECORDINGS_DIR, filename)

        # Save to WAV file
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        print(f"\n Saved: {filepath}")
        print(f"   Duration: {duration:.1f} seconds\n")

    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_event.set()
            if self.recording_thread:
                self.recording_thread.join()


# Global recorder instance
recorder = None


# Global hotkey handler using QuickMacHotKey
@quickHotKey(
    virtualKey=kVK_ANSI_D,
    modifierMask=mask(cmdKey, controlKey, optionKey)
)
def handle_hotkey():
    """Handle the global hotkey Cmd+Option+Control+D."""
    global recorder
    if recorder:
        recorder.toggle_recording()


class AppDelegate(NSObject):
    """Simple app delegate for NSApplication."""

    def applicationDidFinishLaunching_(self, notification):
        """Set up when app finishes launching."""
        print("Hotkey monitor started. Press Cmd+Option+Control+D to toggle recording.")
        print("Press Ctrl+C to exit.\n")


def main():
    global recorder

    print("=" * 50)
    print("  Audio Recorder for ASR Benchmarking")
    print("=" * 50)

    # Create recorder instance
    recorder = AudioRecorder()

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
        if recorder:
            recorder.cleanup()


if __name__ == "__main__":
    main()
