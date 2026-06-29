"""Synthesize spoken narration of the subtitle track with a local Piper voice
and lay it on a timeline that matches the on-screen captions.

This is plain offline text-to-speech of the same subtitles shown in the video
(a neutral narrator voice) -- not an impersonation of any real person.

Usage:
    python build_narration.py --voice voices/en_US-hfc_female-medium.onnx \
        --intro 20 --duration 166 --out narration.wav
"""

from __future__ import annotations

import argparse
import re
import wave

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig

from ogre.sim import INTRO_SCRIPT, SUBTITLE_SCRIPT

SR = 22050

# Spoken-form fixups so the narrator says things naturally.
REPLACEMENTS = [
    (r"\(.*?\)", ""),               # drop visual asides like "(top-left)"
    ("5D-connascence", "five-dimensional connascence"),
    ("5D", "five dimensional"),
    ("RK4", "R K four"),
    ("FIFO", "first-in first-out"),
    ("OGrE", "Ogre"),
    ("\u2192", ", "),               # arrow -> pause
    ("\u2014", ", "),               # em dash
    ("\u2013", ", "),               # en dash
    ("KEEP", "keep"), ("EVICT", "evict"), ("REVIEW", "review"),
]


def clean(text: str) -> str:
    for pat, rep in REPLACEMENTS:
        if pat.startswith("\\") or "(" in pat or ".*" in pat:
            text = re.sub(pat, rep, text)
        else:
            text = text.replace(pat, rep)
    text = re.sub(r"\s+", " ", text).replace(" ,", ",").replace(" .", ".")
    return text.strip()


def synth(voice: PiperVoice, text: str, length_scale: float = 1.0) -> np.ndarray:
    cfg = SynthesisConfig(length_scale=length_scale, normalize_audio=True)
    parts = [c.audio_int16_array for c in voice.synthesize(text, cfg)]
    if not parts:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(parts).astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="voices/en_US-hfc_female-medium.onnx")
    ap.add_argument("--intro", type=float, default=20.0)
    ap.add_argument("--duration", type=float, default=166.0)
    ap.add_argument("--out", default="narration.wav")
    args = ap.parse_args()

    # Merge cues onto one wall-clock timeline.
    cues = [(s, clean(t)) for s, _, t in INTRO_SCRIPT]
    cues += [(s + args.intro, clean(t)) for s, _, t in SUBTITLE_SCRIPT]
    cues.sort()
    starts = [c[0] for c in cues]

    print(f"loading voice {args.voice} ...")
    voice = PiperVoice.load(args.voice)

    master = np.zeros(int((args.duration + 1.5) * SR), dtype=np.int32)
    for i, (start, text) in enumerate(cues):
        next_start = starts[i + 1] if i + 1 < len(starts) else args.duration
        window = max(1.0, next_start - start - 0.15)
        audio = synth(voice, text)
        dur = len(audio) / SR
        if dur > window:                       # speed up to fit its caption window
            ls = max(0.80, window / dur)
            audio = synth(voice, text, length_scale=ls)
            dur = len(audio) / SR
        at = int(start * SR)
        end = min(len(master), at + len(audio))
        master[at:end] += audio[: end - at].astype(np.int32)
        print(f"  [{start:6.1f}s] {dur:4.1f}s  {text[:54]}")

    master = np.clip(master, -32768, 32767).astype(np.int16)
    with wave.open(args.out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(master.tobytes())
    print(f"wrote {args.out}  ({len(master)/SR:.1f}s)")


if __name__ == "__main__":
    main()
