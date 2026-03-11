"""Comical and funky loading animations for Homie.

Shows entertaining progress messages while the LLM is thinking.
Works in both CLI (terminal) and overlay (tkinter) contexts.
"""
from __future__ import annotations

import itertools
import random
import sys
import threading
import time
from typing import Optional


# Funky thinking messages — rotated randomly
_THINKING_MESSAGES = [
    "Consulting my crystal ball",
    "Asking the rubber duck",
    "Googling... just kidding, I'm local",
    "Rummaging through my brain attic",
    "Summoning the matrix",
    "Crunching neurons at lightspeed",
    "Downloading more RAM... not really",
    "Bribing the hamsters on the wheel",
    "Untangling my neural spaghetti",
    "Doing big brain math",
    "Pretending to think really hard",
    "Warming up my quantum cores",
    "Flipping through my infinite notebook",
    "Channeling the spirit of Turing",
    "Shuffling my weights and biases",
    "Calculating the meaning of life",
    "Percolating some fresh thoughts",
    "Running on pure caffeine and tensors",
    "Teaching my neurons new tricks",
    "Assembling the answer like IKEA furniture",
    "Spinning up my thinking hamster",
    "Converting coffee to intelligence",
    "Traversing the knowledge graph",
    "Defragmenting my thought process",
    "Compiling wit and wisdom",
    "Loading sarcasm module",
    "Polishing my response crystal",
    "Tuning the frequency of genius",
    "Shaking the magic 8-ball",
    "Reading between the embeddings",
]

# Funky spinner frames
_SPINNER_FRAMES = [
    "( o_o)  ",
    "( o_O)  ",
    "( O_o)  ",
    "( O_O)  ",
    "( o_o) .",
    "( o_o) ..",
    "( o_o) ...",
    "( -_-) zzz",
    "\\( o_o)/",
    " ( o_o) ",
    "  (o_o )",
    " ( o_o) ",
]

_BRAINWAVE_FRAMES = [
    "~(o_o)~  ",
    "~~(o_o)~ ",
    "~~~(o_o) ",
    "~~(o_o)~ ",
    "~(o_o)~~ ",
    " (o_o)~~~",
    "~(o_o)~~ ",
    "~~(o_o)~ ",
]

_DANCE_FRAMES = [
    "♪ ┏(o_o)┛ ♪",
    "♫ ┗(o_o)┓ ♫",
    "♪ ┏(o_o)┛ ♪",
    "♬ ┗(o_o)┓ ♬",
]


class CLILoadingSpinner:
    """Terminal-based funky loading spinner with rotating messages."""

    def __init__(self, style: str = "random"):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._style = style
        self._message = ""

    def _pick_frames(self) -> list[str]:
        if self._style == "brainwave":
            return _BRAINWAVE_FRAMES
        elif self._style == "dance":
            return _DANCE_FRAMES
        elif self._style == "random":
            return random.choice([_SPINNER_FRAMES, _BRAINWAVE_FRAMES, _DANCE_FRAMES])
        return _SPINNER_FRAMES

    def _spin(self) -> None:
        frames = self._pick_frames()
        frame_cycle = itertools.cycle(frames)
        messages = list(_THINKING_MESSAGES)
        random.shuffle(messages)
        msg_cycle = itertools.cycle(messages)
        current_msg = next(msg_cycle)
        msg_timer = 0

        while self._running:
            frame = next(frame_cycle)
            # Rotate message every ~3 seconds
            if msg_timer >= 12:
                current_msg = next(msg_cycle)
                msg_timer = 0

            line = f"\r  {frame}  {current_msg}..."
            # Pad to clear previous line
            sys.stdout.write(f"{line:<75}")
            sys.stdout.flush()

            time.sleep(0.25)
            msg_timer += 1

        # Clear the spinner line
        sys.stdout.write("\r" + " " * 75 + "\r")
        sys.stdout.flush()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None


class OverlayLoadingAnimation:
    """Tkinter-compatible loading animation for the overlay popup.

    Updates a tkinter Label with funky animated text. Must be called
    from the tkinter main thread via root.after().
    """

    def __init__(self):
        self._frame_idx = 0
        self._msg_idx = 0
        self._messages = list(_THINKING_MESSAGES)
        random.shuffle(self._messages)
        self._frames = random.choice([_SPINNER_FRAMES, _BRAINWAVE_FRAMES, _DANCE_FRAMES])
        self._running = False
        self._tick_count = 0

    def start(self, root, label) -> None:
        """Start animating a tkinter Label."""
        self._running = True
        self._tick_count = 0
        self._animate(root, label)

    def stop(self) -> None:
        """Stop the animation."""
        self._running = False

    def _animate(self, root, label) -> None:
        if not self._running:
            return
        try:
            frame = self._frames[self._frame_idx % len(self._frames)]
            msg = self._messages[self._msg_idx % len(self._messages)]
            label.config(text=f"{frame}  {msg}...")

            self._frame_idx += 1
            self._tick_count += 1
            # Rotate message every ~3 seconds (12 ticks at 250ms)
            if self._tick_count % 12 == 0:
                self._msg_idx += 1

            root.after(250, lambda: self._animate(root, label))
        except Exception:
            pass  # widget destroyed


def get_random_thinking_message() -> str:
    """Get a single random thinking message."""
    return random.choice(_THINKING_MESSAGES)
