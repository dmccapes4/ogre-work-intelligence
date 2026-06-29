"""Local ingestion agent.

A background worker thread pulls raw work-item text off an input queue, asks a
local Ollama model to enrich it into structured form (concepts, importance,
load-bearing, explicit references), and pushes a ready-to-insert ``Node`` onto
an output queue. All network I/O happens off the render thread, so the pygame
loop never stalls. If Ollama is slow or unavailable, a deterministic heuristic
keeps the simulation flowing.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from typing import List, Optional, Tuple

import requests

from .model import Node

OLLAMA_URL = os.environ.get("OGRE_OLLAMA_URL", "http://localhost:11434")
# Default to the strongest model that fits comfortably in 12 GB VRAM.
DEFAULT_MODEL = os.environ.get("OGRE_MODEL", "qwen2.5-coder:14b-instruct-q4_K_M")
REQUEST_TIMEOUT = float(os.environ.get("OGRE_TIMEOUT", "45"))
# Cold model loads (a 9 GB model into VRAM) can take a while — give the warmup
# its own generous budget so it doesn't get mistaken for an outage.
WARMUP_TIMEOUT = float(os.environ.get("OGRE_WARMUP_TIMEOUT", "240"))
MAX_CONSECUTIVE_FAILURES = 3
REPROBE_INTERVAL = 15.0

_STOPWORDS = {
    "the", "a", "an", "to", "in", "for", "of", "and", "or", "with", "on", "at",
    "from", "by", "is", "are", "be", "add", "fix", "the", "into", "out", "up",
    "this", "that", "before", "after", "when", "exceeds", "show", "build",
    "write", "set", "use", "using", "via",
}

_PROMPT = """You are a work-intelligence ingestion agent. Extract structured metadata \
from a single work item. Respond with ONLY a compact JSON object, no prose.

Work item: {text}

JSON schema:
{{
  "concepts": [3-6 short lowercase domain keywords],
  "importance": "high" | "medium" | "low",
  "load_bearing": true if other work clearly depends on this, else false,
  "references": [keywords of other systems/items this depends on]
}}"""


def heuristic_enrich(text: str, importance_hint: str, lb_hint: bool) -> dict:
    """Cheap, dependency-free enrichment used as a fallback."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", text.lower())
    concepts = []
    for w in words:
        if w in _STOPWORDS or len(w) < 3:
            continue
        if w not in concepts:
            concepts.append(w)
    return {
        "concepts": concepts[:6],
        "importance": importance_hint,
        "load_bearing": lb_hint,
        "references": [],
    }


def _coerce(payload: dict, text: str, importance_hint: str, lb_hint: bool) -> dict:
    """Validate / clean a model response into our enrichment shape."""
    out = heuristic_enrich(text, importance_hint, lb_hint)
    if not isinstance(payload, dict):
        return out
    concepts = payload.get("concepts")
    if isinstance(concepts, list):
        cleaned = [str(c).strip().lower() for c in concepts if str(c).strip()]
        if cleaned:
            out["concepts"] = cleaned[:6]
    imp = str(payload.get("importance", "")).strip().lower()
    if imp in ("high", "medium", "low"):
        out["importance"] = imp
    if isinstance(payload.get("load_bearing"), bool):
        out["load_bearing"] = payload["load_bearing"]
    refs = payload.get("references")
    if isinstance(refs, list):
        out["references"] = [str(r).strip().lower() for r in refs if str(r).strip()]
    return out


class IngestionAgent:
    """Threaded Ollama-backed work-item enrichment agent."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.inbox: "queue.Queue[Tuple[str, str, bool, int]]" = queue.Queue()
        self.outbox: "queue.Queue[Node]" = queue.Queue()
        # starting | loading | ollama | busy | heuristic
        self.status = "starting"
        self.online = False
        self.processed = 0
        self.last_latency = 0.0
        self.consecutive_failures = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ogre-agent", daemon=True)

    # ---- lifecycle ------------------------------------------------------
    def start(self) -> None:
        self.online = self._probe()
        # The worker warms the model first (so the cold VRAM load doesn't look
        # like a stalled request), then begins consuming the inbox.
        self.status = "loading" if self.online else "heuristic"
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _probe(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # ---- public API -----------------------------------------------------
    def submit(self, text: str, importance: str, load_bearing: bool, batch: int) -> None:
        self.inbox.put((text, importance, load_bearing, batch))

    def drain(self) -> List[Node]:
        """Collect all nodes the worker has finished enriching."""
        ready: List[Node] = []
        while True:
            try:
                ready.append(self.outbox.get_nowait())
            except queue.Empty:
                break
        return ready

    def pending(self) -> int:
        return self.inbox.qsize()

    # ---- worker ---------------------------------------------------------
    def _run(self) -> None:
        if self.online:
            self._warmup()
        last_reprobe = time.time()

        while not self._stop.is_set():
            try:
                text, imp, lb, batch = self.inbox.get(timeout=0.25)
            except queue.Empty:
                # Idle: if we've gone offline, periodically try to recover.
                if not self.online and (time.time() - last_reprobe) > REPROBE_INTERVAL:
                    last_reprobe = time.time()
                    if self._probe():
                        self.online = True
                        self.consecutive_failures = 0
                        self._warmup()
                elif self.online and self.status == "busy":
                    self.status = "ollama"
                continue

            self.status = "busy"
            enrich = self._enrich(text, imp, lb)
            node = Node(
                text=text,
                concepts=set(enrich["concepts"]) | set(enrich.get("references", [])),
                importance=enrich["importance"],
                load_bearing=enrich["load_bearing"],
                batch=batch,
            )
            self.outbox.put(node)
            self.processed += 1

    def _warmup(self) -> None:
        """Force the model into VRAM with a generous timeout before ingesting."""
        self.status = "loading"
        t0 = time.time()
        try:
            requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": self.model, "prompt": "ok", "stream": False,
                      "options": {"num_predict": 1}},
                timeout=WARMUP_TIMEOUT,
            ).raise_for_status()
            self.last_latency = time.time() - t0
            self.status = "ollama"
            self.consecutive_failures = 0
        except requests.RequestException:
            # Couldn't warm up; fall back but keep trying to recover later.
            self.online = False
            self.status = "heuristic"

    def _enrich(self, text: str, imp: str, lb: bool) -> dict:
        if not self.online:
            return heuristic_enrich(text, imp, lb)
        t0 = time.time()
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": _PROMPT.format(text=text),
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 200},
                },
                timeout=REQUEST_TIMEOUT,
            )
            self.last_latency = time.time() - t0
            resp.raise_for_status()
            raw = resp.json().get("response", "{}")
            payload = json.loads(raw)
            self.status = "ollama"
            self.consecutive_failures = 0
            return _coerce(payload, text, imp, lb)
        except (requests.RequestException, json.JSONDecodeError, ValueError):
            # Tolerate transient failures: use the heuristic for *this* item but
            # only drop offline after several consecutive failures.
            self.consecutive_failures += 1
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self.online = False
                self.status = "heuristic"
            return heuristic_enrich(text, imp, lb)
