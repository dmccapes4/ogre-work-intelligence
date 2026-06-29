"""Pygame visualization of the OGrE work-intelligence system.

Layout
------
+----------------------------------------+---------------------------+
| top bar: title / agent status / keys                               |
+----------------------------------------+---------------------------+
|                                        |   Lorenz attractor (GC)   |
|        stable work graph               +---------------------------+
|   (force-directed, 5D edges)           |   eviction log (FIFO/100) |
|                                        +---------------------------+
|                                        |   stats                   |
+----------------------------------------+---------------------------+
| activity console                                                   |
+--------------------------------------------------------------------+
"""

from __future__ import annotations

import math
import os
import random
import threading
import time
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional, Tuple

import pygame

from .agent import IngestionAgent
from .gc_engine import LorenzGC, GCResult
from .model import (
    CONNASCENCE_COLORS,
    CONNASCENCE_WEIGHTS,
    Node,
    WorkGraph,
)
from .workitems import WorkItemSource

# ---- palette ------------------------------------------------------------
BG = (14, 16, 22)
PANEL = (22, 26, 34)
PANEL_HI = (28, 33, 44)
BORDER = (46, 53, 68)
TEXT = (214, 220, 230)
MUTED = (122, 132, 148)
ACCENT = (124, 196, 255)

CLASS_COLOR = {
    "KEEP": (52, 211, 153),
    "REVIEW": (251, 191, 36),
    "EVICT": (255, 92, 92),
}

WIDTH, HEIGHT = 1480, 940
TOPBAR_H = 56
CONSOLE_H = 132
RIGHT_W = 452
PAD = 12


def blend(fg, bg, a: float):
    """Alpha-composite fg over bg with alpha a in [0,1] (no per-pixel alpha)."""
    a = max(0.0, min(1.0, a))
    return (
        int(fg[0] * a + bg[0] * (1 - a)),
        int(fg[1] * a + bg[1] * (1 - a)),
        int(fg[2] * a + bg[2] * (1 - a)),
    )


# Narration for the animated Lorenz intro scene, keyed to seconds into the intro.
INTRO_SCRIPT: List[Tuple[float, float, str]] = [
    (0.5, 6.5, "Meet the Lorenz attractor \u2014 just three short equations (top-left). That is the entire rule."),
    (6.5, 13.0, "Yet the solution is chaos: one point traces these two wings forever \u2014 never repeating, never escaping."),
    (13.0, 19.5, "OGrE's trick is to let chaos decide what to remember. Left wing \u2192 KEEP, right wing \u2192 EVICT. No hand-tuned thresholds \u2014 just the geometry."),
]

# Timed narration, keyed to wall-clock seconds since launch. Walks through the
# whole system in order so the recording is self-explanatory.
SUBTITLE_SCRIPT: List[Tuple[float, float, str]] = [
    (1.0, 11.0, "OGrE: a live simulation of graph-based work intelligence. A local agent ingests work items into a knowledge graph."),
    (11.0, 21.0, "Each item is enriched by a local Ollama model (qwen2.5-coder 14B) that extracts its concepts, importance, and dependencies."),
    (21.0, 32.0, "Opportunistic Graph enrichment (OGrE) then weaves 5D-connascence edges between related items."),
    (32.0, 43.0, "The five dimensions: STRUCTURAL, CONCEPTUAL, TEMPORAL, CO-OCCURRENCE, CO-VARIANCE. Edge color = type, thickness = strength."),
    (43.0, 54.0, "Every node has a vitality that decays over time \u2014 but vitality propagates along edges, so connected work reinforces itself."),
    (54.0, 65.0, "Densely re-referenced clusters hold each other up and become stable attractors: the graph's long-term memory (glowing green)."),
    (65.0, 77.0, "Now the payoff \u2014 that butterfly is the garbage collector. Each node's connectivity, connascence and vitality become a point dropped into the attractor (top-right)."),
    (77.0, 89.0, "RK4 integrates its trajectory and we watch which wing it settles into: left wing KEEPs it, right wing EVICTs it, the chaotic boundary flags REVIEW. Chaos casts the deciding vote."),
    (89.0, 100.0, "Evicted items leave the graph and drop into a 100-slot FIFO eviction log. Load-bearing items are spared and downgraded to REVIEW."),
    (100.0, 111.0, "When the log fills, the oldest evictions fall off the tail \u2014 forgotten forever."),
    (111.0, 123.0, "An agent searching the past checks the live graph first, then the eviction log..."),
    (123.0, 135.0, "...and resurrects any evicted item that gets referenced again, pulling it back into the graph (blue ring)."),
    (135.0, 146.0, "Ingest \u2192 enrich \u2192 decay \u2192 classify \u2192 evict \u2192 forget \u2192 resurrect. A working memory governed by the geometry of chaos."),
]


class Recorder:
    """Pipes rendered pygame frames into a .mov via imageio's bundled ffmpeg.

    Frames are sampled at a fixed wall-clock ``fps`` (independent of the render
    rate) so playback is real-time regardless of how fast the loop runs.
    """

    def __init__(self, path: str, fps: int = 30) -> None:
        import numpy as np            # local import: only needed when recording
        import imageio.v2 as imageio
        self._np = np
        self.path = path
        self.fps = fps
        self.frames = 0
        self._interval = 1.0 / fps
        self._accum = 0.0
        self._writer = imageio.get_writer(
            path, fps=fps, codec="libx264", quality=8,
            macro_block_size=2,                  # 1480x940 are both even
            ffmpeg_log_level="error",
            output_params=["-pix_fmt", "yuv420p"],
        )

    def maybe_capture(self, screen: "pygame.Surface", dt_real: float) -> None:
        # Wall-clock catch-up: emit the right number of frames for elapsed time
        # (duplicating when the loop is slow) so playback stays real-time. Cap
        # duplicates so a slow encode can't spiral.
        self._accum += dt_real
        emitted = 0
        while self._accum >= self._interval and emitted < 4:
            self.capture(screen)
            self._accum -= self._interval
            emitted += 1
        if self._accum > self._interval:
            self._accum = self._interval

    def capture(self, screen: "pygame.Surface") -> None:
        w, h = screen.get_size()
        buf = pygame.image.tostring(screen, "RGB")
        arr = self._np.frombuffer(buf, dtype=self._np.uint8).reshape((h, w, 3))
        self._writer.append_data(arr)
        self.frames += 1

    def close(self) -> None:
        try:
            self._writer.close()
        except Exception:
            pass


class SimApp:
    def __init__(self, model: Optional[str] = None, seed: int = 7,
                 record_path: Optional[str] = None, record_seconds: float = 0.0,
                 record_fps: int = 30, subtitles: bool = False,
                 intro: float = 0.0) -> None:
        pygame.init()
        pygame.display.set_caption("OGrE — Opportunistic Graph-enrichment Work Intelligence")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.rng = random.Random(seed)

        # fonts
        self.f_title = pygame.font.SysFont("dejavusans", 20, bold=True)
        self.f_h = pygame.font.SysFont("dejavusans", 15, bold=True)
        self.f = pygame.font.SysFont("dejavusans", 13)
        self.f_sm = pygame.font.SysFont("dejavusans", 11)
        self.f_mono = pygame.font.SysFont("dejavusansmono", 12)
        self.f_sub = pygame.font.SysFont("dejavusans", 17, bold=True)

        # domain objects
        self.graph = WorkGraph()
        self.agent = IngestionAgent(model=model) if model else IngestionAgent()
        self.agent.start()
        self.gc = LorenzGC(rho=28.0, tau=2.0)
        self.source = WorkItemSource(seed=seed)

        # regions
        self.graph_rect = pygame.Rect(
            PAD, TOPBAR_H + PAD,
            WIDTH - RIGHT_W - 3 * PAD, HEIGHT - TOPBAR_H - CONSOLE_H - 3 * PAD,
        )
        rx = self.graph_rect.right + PAD
        avail_h = HEIGHT - TOPBAR_H - CONSOLE_H - 3 * PAD
        self.lorenz_rect = pygame.Rect(rx, TOPBAR_H + PAD, RIGHT_W - PAD, int(avail_h * 0.42))
        self.evict_rect = pygame.Rect(rx, self.lorenz_rect.bottom + PAD, RIGHT_W - PAD, int(avail_h * 0.36))
        self.stats_rect = pygame.Rect(rx, self.evict_rect.bottom + PAD, RIGHT_W - PAD,
                                      avail_h - self.lorenz_rect.height - self.evict_rect.height - 2 * PAD)
        self.console_rect = pygame.Rect(PAD, HEIGHT - CONSOLE_H, WIDTH - 2 * PAD, CONSOLE_H - PAD)

        # simulation control
        self.running = True
        self.paused = False
        self.speed = 1.0
        self.auto_ingest = True
        self.show_help = not subtitles      # help and subtitles share the lower area
        self.subtitles_on = subtitles
        self._wall_elapsed = 0.0            # real seconds since launch (subtitle clock)
        self.intro_seconds = intro          # full-screen animated Lorenz attractor intro
        self.batch_counter = 0
        self.max_nodes = 72       # working-set budget for the stable graph
        self._cap_spill_accum = 0
        self.max_pending = 8      # agent-queue backpressure cap for auto-ingest

        # recording
        self.record_fps = record_fps
        self.record_seconds = record_seconds
        self.recorder: Optional[Recorder] = None
        self._record_started_at = 0.0
        self._pending_record_path = record_path

        self._ingest_timer = 0.0
        self._gc_timer = 0.0
        self._search_timer = 0.0
        self.ingest_interval = 2.2
        self.gc_interval = 6.0
        self.search_interval = 3.5

        # async GC state
        self._gc_thread: Optional[threading.Thread] = None
        self._gc_snapshot_result: Optional[dict] = None
        self.gc_running = False
        self.last_gc_duration = 0.0
        self.gc_passes = 0

        # precompute a faint Lorenz butterfly backdrop
        self._butterfly = self._precompute_butterfly()
        # high-resolution 3D trajectory for the animated intro scene
        self._lorenz_hi = self._precompute_lorenz_hi()

        # console
        self.console: Deque[Tuple[str, Tuple[int, int, int]]] = deque(maxlen=8)
        self.log("OGrE simulation started", ACCENT)
        self.log(f"agent: {self.agent.model} ({'ollama' if self.agent.online else 'heuristic fallback'})",
                 CLASS_COLOR["KEEP"] if self.agent.online else CLASS_COLOR["REVIEW"])

        # seed a few items so the graph isn't empty
        self._ingest_batch(size=3)

        if self._pending_record_path:
            self.start_recording(self._pending_record_path)

    # ---- logging --------------------------------------------------------
    def log(self, msg: str, color=TEXT) -> None:
        ts = f"{self.graph.clock:6.1f}s"
        self.console.append((f"[{ts}] {msg}", color))

    # ---- recording ------------------------------------------------------
    def start_recording(self, path: Optional[str] = None) -> None:
        if self.recorder is not None:
            return
        if not path:
            path = f"ogre-capture-{datetime.now():%Y%m%d-%H%M%S}.mov"
        try:
            self.recorder = Recorder(path, fps=self.record_fps)
        except Exception as exc:
            self.log(f"recording failed to start: {exc}", CLASS_COLOR["EVICT"])
            return
        self._record_started_at = time.time()
        self.log(f"recording -> {os.path.basename(path)} @ {self.record_fps}fps", CLASS_COLOR["EVICT"])

    def stop_recording(self) -> None:
        if self.recorder is None:
            return
        path, frames = self.recorder.path, self.recorder.frames
        self.recorder.close()
        self.recorder = None
        self.log(f"saved {os.path.basename(path)} ({frames} frames)", CLASS_COLOR["KEEP"])

    def toggle_recording(self) -> None:
        if self.recorder is None:
            self.start_recording()
        else:
            self.stop_recording()

    # ---- precompute -----------------------------------------------------
    def _precompute_butterfly(self) -> List[Tuple[float, float]]:
        from provenance_engine import integrate_portal
        traj = integrate_portal(0.1, 0.0, 0.0, rho=28.0, t_max=40.0, dt=0.01)
        return [(p[0], p[2]) for p in traj[400:]]   # (x, z), skip transient

    def _precompute_lorenz_hi(self) -> List[Tuple[float, float, float]]:
        """A long, dense 3D Lorenz trajectory for the rotating intro butterfly."""
        from provenance_engine import integrate_portal
        traj = integrate_portal(0.1, 0.0, 0.0, rho=28.0, t_max=75.0, dt=0.01)
        return [tuple(p) for p in traj[300:]]       # skip the transient swing-in

    # ---- ingestion ------------------------------------------------------
    def _ingest_one(self) -> None:
        text, imp, lb = self.source.next_item()
        self.batch_counter += 1
        self.agent.submit(text, imp, lb, self.batch_counter)

    def _ingest_batch(self, size: int = 3) -> None:
        self.batch_counter += 1
        for text, imp, lb in self.source.next_batch(size):
            self.agent.submit(text, imp, lb, self.batch_counter)

    def _collect_enriched(self) -> None:
        for node in self.agent.drain():
            # spawn near graph centre with a little scatter
            cx, cy = self.graph_rect.center
            node.x = cx + self.rng.uniform(-60, 60)
            node.y = cy + self.rng.uniform(-60, 60)
            self.graph.add_node(node)
            connected = self.graph.ogre_enrich(node)
            # opportunistically reference what we just linked to (keeps clusters warm)
            for cid in connected:
                n = self.graph.get(cid)
                if n:
                    n.flash = max(n.flash, 0.6)
            tag = "resurrect" if node.resurrected else "ingest"
            self.log(f"{tag}: '{node.label}' [{node.importance}] +{len(connected)} edges",
                     CLASS_COLOR["KEEP"] if connected else MUTED)

    # ---- GC (async) -----------------------------------------------------
    def _start_gc(self) -> None:
        if self.gc_running or len(self.graph) == 0:
            return
        snapshot = self.graph.to_pe_nodes()
        self.gc_running = True
        self._gc_snapshot_result = None

        def work():
            t0 = time.time()
            res = self.gc.classify(snapshot)
            self._gc_snapshot_result = {"results": res, "duration": time.time() - t0}

        self._gc_thread = threading.Thread(target=work, daemon=True)
        self._gc_thread.start()

    def _poll_gc(self) -> None:
        if not self.gc_running or self._gc_snapshot_result is None:
            return
        payload = self._gc_snapshot_result
        self._gc_snapshot_result = None
        self.gc_running = False
        self.last_gc_duration = payload["duration"]
        self.gc_passes += 1

        results = payload["results"]
        evicted = [nid for nid, r in results.items() if r["classification"] == "EVICT"]
        res = GCResult(classifications=results, evicted_ids=evicted,
                       rho=self.gc.rho, tau=self.gc.tau, duration=payload["duration"])
        recs = self.gc.apply(self.graph, res)
        c = self.graph.counts()
        if recs:
            names = ", ".join(r.node.label for r in recs[:3])
            extra = "" if len(recs) <= 3 else f" +{len(recs) - 3}"
            self.log(f"GC pass {self.gc_passes}: evicted {len(recs)} ({names}{extra})", CLASS_COLOR["EVICT"])
        else:
            self.log(f"GC pass {self.gc_passes}: KEEP={c['KEEP']} REVIEW={c['REVIEW']} no evictions", MUTED)
        if self._cap_spill_accum:
            self.log(f"capacity governance: spilled {self._cap_spill_accum} weakest items to log",
                     CLASS_COLOR["REVIEW"])
            self._cap_spill_accum = 0

    # ---- search ---------------------------------------------------------
    def _do_search(self) -> None:
        term = self.gc.sample_query_term(self.graph, self.rng)
        if not term:
            return
        outcome, node = self.gc.search(self.graph, term)
        if outcome == "graph-hit":
            self.log(f"search '{term}' -> graph hit: '{node.label}' (referenced)", ACCENT)
        elif outcome == "resurrected":
            node.x = self.graph_rect.centerx + self.rng.uniform(-40, 40)
            node.y = self.graph_rect.centery + self.rng.uniform(-40, 40)
            self.log(f"search '{term}' -> RESURRECTED from eviction log: '{node.label}'",
                     CLASS_COLOR["REVIEW"])
        else:
            self.log(f"search '{term}' -> miss (gone from graph and log)", MUTED)

    # ---- update ---------------------------------------------------------
    def update(self, dt_real: float) -> None:
        self._wall_elapsed += dt_real
        self._collect_enriched()
        # Hard working-set bound every frame (the logged narrative fires on GC).
        spilled = self.gc.enforce_capacity(self.graph, self.max_nodes)
        if spilled:
            self._cap_spill_accum += len(spilled)
        self._poll_gc()
        if self.paused:
            self._layout_step(dt_real)
            return

        dt = dt_real * self.speed
        self.graph.decay_step(dt)

        self._ingest_timer += dt
        self._gc_timer += dt
        self._search_timer += dt

        if self.auto_ingest and self._ingest_timer >= self.ingest_interval:
            self._ingest_timer = 0.0
            # Backpressure: don't outrun the agent. A bigger model is slower, so
            # hold off submitting more when its queue is already deep.
            if self.agent.pending() < self.max_pending:
                if self.rng.random() < 0.5:
                    self._ingest_batch(self.rng.randint(2, 4))
                else:
                    self._ingest_one()

        if self._gc_timer >= self.gc_interval:
            self._gc_timer = 0.0
            self._start_gc()

        if self._search_timer >= self.search_interval:
            self._search_timer = 0.0
            self._do_search()

        self._layout_step(dt_real)

    # ---- force-directed layout -----------------------------------------
    def _layout_step(self, dt_real: float) -> None:
        nodes = list(self.graph.values())
        n = len(nodes)
        if n == 0:
            return
        r = self.graph_rect
        cx, cy = r.center
        k_rep = 5200.0
        k_spring = 0.018
        rest = 70.0
        damping = 0.82

        for i in range(n):
            a = nodes[i]
            fx = fy = 0.0
            # repulsion
            for j in range(n):
                if i == j:
                    continue
                b = nodes[j]
                dx = a.x - b.x
                dy = a.y - b.y
                d2 = dx * dx + dy * dy + 0.01
                if d2 < 90000:
                    f = k_rep / d2
                    d = math.sqrt(d2)
                    fx += f * dx / d
                    fy += f * dy / d
            # springs along edges
            for e in a.edges.values():
                b = self.graph.get(e.target)
                if b is None:
                    continue
                dx = b.x - a.x
                dy = b.y - a.y
                d = math.sqrt(dx * dx + dy * dy) + 0.01
                f = k_spring * (d - rest) * (0.5 + e.strength)
                fx += f * dx / d
                fy += f * dy / d
            # gentle gravity to centre
            fx += (cx - a.x) * 0.0016
            fy += (cy - a.y) * 0.0016

            a.vx = (a.vx + fx) * damping
            a.vy = (a.vy + fy) * damping
            sp = math.hypot(a.vx, a.vy)
            if sp > 18:
                a.vx *= 18 / sp
                a.vy *= 18 / sp
            a.x += a.vx
            a.y += a.vy
            # clamp inside panel
            m = 26
            a.x = max(r.left + m, min(r.right - m, a.x))
            a.y = max(r.top + m, min(r.bottom - m, a.y))

    # ---- rendering ------------------------------------------------------
    def _panel(self, rect: pygame.Rect, title: str) -> None:
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, rect, width=1, border_radius=8)
        if title:
            self.screen.blit(self.f_h.render(title, True, TEXT), (rect.x + 12, rect.y + 8))

    def draw(self) -> None:
        if self._wall_elapsed < self.intro_seconds:
            self._draw_lorenz_scene()
            pygame.display.flip()
            return
        self.screen.fill(BG)
        self._draw_topbar()
        self._draw_graph()
        self._draw_lorenz()
        self._draw_eviction_log()
        self._draw_stats()
        self._draw_console()
        if self.show_help:
            self._draw_help()
        if self.subtitles_on:
            self._draw_subtitles()
        pygame.display.flip()

    def _draw_topbar(self) -> None:
        bar = pygame.Rect(0, 0, WIDTH, TOPBAR_H)
        pygame.draw.rect(self.screen, PANEL_HI, bar)
        pygame.draw.line(self.screen, BORDER, (0, TOPBAR_H), (WIDTH, TOPBAR_H))
        self.screen.blit(self.f_title.render("OGrE  ·  Work Intelligence", True, TEXT), (PAD, 8))
        sub = "Opportunistic Graph enrichment + 5D-connascence + Lorenz GC"
        self.screen.blit(self.f_sm.render(sub, True, MUTED), (PAD, 34))

        # REC indicator (blinks)
        if self.recorder is not None:
            blink = (int(time.time() * 2) % 2) == 0
            rc = CLASS_COLOR["EVICT"]
            cx = WIDTH // 2 - 60
            if blink:
                pygame.draw.circle(self.screen, rc, (cx, 18), 6)
            txt = self.f.render(f"REC  {self.recorder.frames // self.record_fps}s  "
                                f"{self.recorder.frames}f", True, rc)
            self.screen.blit(txt, (cx + 14, 10))

        # agent status pill
        status_colors = {
            "ollama": CLASS_COLOR["KEEP"],
            "busy": CLASS_COLOR["KEEP"],
            "loading": ACCENT,
            "heuristic": CLASS_COLOR["REVIEW"],
            "starting": MUTED,
        }
        scol = status_colors.get(self.agent.status, MUTED)
        atxt = f"agent: {self.agent.model}  [{self.agent.status}]  q={self.agent.pending()}"
        if self.agent.last_latency:
            atxt += f"  {self.agent.last_latency*1000:.0f}ms"
        surf = self.f.render(atxt, True, scol)
        self.screen.blit(surf, (WIDTH - surf.get_width() - PAD, 8))

        state = []
        state.append("PAUSED" if self.paused else f"x{self.speed:.1f}")
        state.append("auto-ingest ON" if self.auto_ingest else "auto-ingest OFF")
        st = self.f_sm.render("   ".join(state), True, MUTED)
        self.screen.blit(st, (WIDTH - st.get_width() - PAD, 34))

    def _draw_graph(self) -> None:
        self._panel(self.graph_rect, "")
        title = self.f_h.render("STABLE WORK GRAPH", True, TEXT)
        self.screen.blit(title, (self.graph_rect.x + 12, self.graph_rect.y + 8))
        sub = self.f_sm.render(
            f"{len(self.graph)} items · {self.graph.edge_count()} connascence edges", True, MUTED)
        self.screen.blit(sub, (self.graph_rect.x + 12, self.graph_rect.y + 28))
        self._draw_edge_legend()

        clip = self.graph_rect.inflate(-4, -4)
        self.screen.set_clip(clip)

        # edges first
        drawn = set()
        for node in self.graph.values():
            for e in node.edges.values():
                other = self.graph.get(e.target)
                if other is None:
                    continue
                key = (min(node.id, e.target), max(node.id, e.target))
                if key in drawn:
                    continue
                drawn.add(key)
                col = CONNASCENCE_COLORS.get(e.etype, MUTED)
                a = 0.18 + 0.5 * e.strength
                c = blend(col, BG, a)
                w = 1 + int(e.strength * 2.5)
                pygame.draw.line(self.screen, c, (node.x, node.y), (other.x, other.y), w)

        # nodes
        for node in self.graph.values():
            self._draw_node(node)

        self.screen.set_clip(None)

    def _draw_node(self, node: Node) -> None:
        base = CLASS_COLOR.get(node.classification, MUTED)
        vit = node.vitality
        radius = int(6 + node.importance_weight * 6 + vit * 6)
        # fade dim nodes toward background
        col = blend(base, BG, 0.35 + 0.65 * vit)

        # attractor glow for vital KEEP nodes
        if node.classification == "KEEP" and vit > 0.6:
            glow = blend(base, BG, 0.12)
            pygame.draw.circle(self.screen, glow, (int(node.x), int(node.y)), radius + 7)

        pygame.draw.circle(self.screen, col, (int(node.x), int(node.y)), radius)

        # load-bearing ring
        if node.load_bearing:
            pygame.draw.circle(self.screen, blend((255, 255, 255), col, 0.6),
                               (int(node.x), int(node.y)), radius + 2, 2)
        # resurrected marker
        if node.resurrected:
            pygame.draw.circle(self.screen, ACCENT, (int(node.x), int(node.y)), radius + 4, 1)
        # reference flash
        if node.flash > 0.01:
            fc = blend((255, 255, 255), col, node.flash)
            pygame.draw.circle(self.screen, fc, (int(node.x), int(node.y)), radius, 2)

        if radius >= 9:
            lab = self.f_sm.render(node.label[:14], True, blend(TEXT, BG, 0.4 + 0.6 * vit))
            self.screen.blit(lab, (node.x - lab.get_width() / 2, node.y + radius + 1))

    def _draw_edge_legend(self) -> None:
        x = self.graph_rect.right - 150
        y = self.graph_rect.y + 8
        for etype, col in CONNASCENCE_COLORS.items():
            pygame.draw.line(self.screen, col, (x, y + 6), (x + 16, y + 6), 3)
            self.screen.blit(self.f_sm.render(etype.replace("_", " ").title(), True, MUTED), (x + 22, y))
            y += 15

    def _draw_lorenz(self) -> None:
        r = self.lorenz_rect
        self._panel(r, "LORENZ GC — ATTRACTOR WINGS")
        sub = self.f_sm.render(f"rho={self.gc.rho:.0f}  tau={self.gc.tau:.1f}  "
                               f"pass={self.gc_passes}  {self.last_gc_duration*1000:.0f}ms"
                               + ("  [running]" if self.gc_running else ""), True, MUTED)
        self.screen.blit(sub, (r.x + 12, r.y + 28))

        plot = pygame.Rect(r.x + 12, r.y + 48, r.width - 24, r.height - 60)

        def to_screen(x, z):
            sx = plot.x + (x + 22) / 44.0 * plot.width
            sz = plot.bottom - (z + 5) / 55.0 * plot.height
            return int(sx), int(sz)

        # wing shading + tau lines
        midx = plot.x + (0 + 22) / 44.0 * plot.width
        tl = plot.x + (-self.gc.tau + 22) / 44.0 * plot.width
        tr = plot.x + (self.gc.tau + 22) / 44.0 * plot.width
        pygame.draw.rect(self.screen, blend(CLASS_COLOR["KEEP"], PANEL, 0.10),
                         pygame.Rect(plot.x, plot.y, tl - plot.x, plot.height))
        pygame.draw.rect(self.screen, blend(CLASS_COLOR["EVICT"], PANEL, 0.10),
                         pygame.Rect(tr, plot.y, plot.right - tr, plot.height))
        for xline, lab, c in ((tl, "-tau", CLASS_COLOR["KEEP"]), (tr, "+tau", CLASS_COLOR["EVICT"])):
            pygame.draw.line(self.screen, blend(c, PANEL, 0.6),
                             (xline, plot.y), (xline, plot.bottom), 1)

        # faint butterfly backdrop
        pts = [to_screen(x, z) for x, z in self._butterfly]
        prev = None
        col_bf = blend(ACCENT, PANEL, 0.16)
        for p in pts[::2]:
            if prev is not None:
                pygame.draw.line(self.screen, col_bf, prev, p, 1)
            prev = p

        # node dots by mean_x (vertical = vitality spread)
        for node in self.graph.values():
            mx = max(-22, min(22, node.mean_x))
            zz = 5 + node.vitality * 40
            sx, sz = to_screen(mx, zz)
            col = CLASS_COLOR.get(node.classification, MUTED)
            pygame.draw.circle(self.screen, col, (sx, sz), 3)

        # labels
        self.screen.blit(self.f_sm.render("KEEP", True, CLASS_COLOR["KEEP"]), (plot.x + 4, plot.bottom - 14))
        rv = self.f_sm.render("EVICT", True, CLASS_COLOR["EVICT"])
        self.screen.blit(rv, (plot.right - rv.get_width() - 4, plot.bottom - 14))
        rev = self.f_sm.render("REVIEW", True, CLASS_COLOR["REVIEW"])
        self.screen.blit(rev, (midx - rev.get_width() / 2, plot.y + 2))

    def _draw_eviction_log(self) -> None:
        r = self.evict_rect
        n = len(self.gc.eviction_log)
        cap = self.gc.eviction_log.maxlen
        warn = n >= cap * 0.9
        self._panel(r, "")
        tcol = CLASS_COLOR["EVICT"] if warn else TEXT
        self.screen.blit(self.f_h.render(f"EVICTION LOG  ·  FIFO {n}/{cap}", True, tcol), (r.x + 12, r.y + 8))
        self.screen.blit(self.f_sm.render(
            f"evicted={self.gc.total_evicted}  resurrected={self.gc.total_resurrected}  "
            f"forgotten={self.gc.total_forgotten}", True, MUTED), (r.x + 12, r.y + 28))

        # capacity bar
        bar = pygame.Rect(r.x + 12, r.y + 46, r.width - 24, 6)
        pygame.draw.rect(self.screen, blend(BORDER, PANEL, 1.0), bar, border_radius=3)
        fillw = int(bar.width * n / cap)
        fc = CLASS_COLOR["EVICT"] if warn else ACCENT
        pygame.draw.rect(self.screen, fc, pygame.Rect(bar.x, bar.y, fillw, bar.height), border_radius=3)

        # most-recent-first list
        y = r.y + 58
        line_h = 16
        max_rows = (r.bottom - y - 8) // line_h
        recs = list(self.gc.eviction_log)[::-1][:max_rows]
        for rec in recs:
            age = self.graph.clock - rec.evicted_at
            fade = max(0.3, 1.0 - age / 60.0)
            dot = blend(CLASS_COLOR["EVICT"], PANEL, fade)
            pygame.draw.circle(self.screen, dot, (r.x + 18, y + 7), 3)
            label = rec.node.label[:30]
            self.screen.blit(self.f_sm.render(label, True, blend(TEXT, PANEL, fade)), (r.x + 28, y))
            agetxt = self.f_sm.render(f"{age:4.0f}s", True, MUTED)
            self.screen.blit(agetxt, (r.right - agetxt.get_width() - 12, y))
            y += line_h

    def _draw_stats(self) -> None:
        r = self.stats_rect
        self._panel(r, "SYSTEM")
        c = self.graph.counts()
        total = max(1, len(self.graph))
        x = r.x + 12
        y = r.y + 30
        chips = [("KEEP", c["KEEP"]), ("REVIEW", c["REVIEW"]), ("EVICT(log)", len(self.gc.eviction_log))]
        cx = x
        for name, val in chips:
            col = CLASS_COLOR.get(name.split("(")[0], MUTED)
            txt = self.f.render(f"{name}: {val}", True, col)
            pygame.draw.circle(self.screen, col, (cx + 4, y + 8), 4)
            self.screen.blit(txt, (cx + 14, y))
            cx += txt.get_width() + 34
        y += 24

        evict_rate = self.gc.total_evicted / max(1, self.gc.total_evicted + len(self.graph))
        lines = [
            f"items: {len(self.graph)}    edges: {self.graph.edge_count()}",
            f"sim clock: {self.graph.clock:6.1f}s    speed: x{self.speed:.1f}",
            f"eviction rate: {evict_rate*100:4.1f}%    agent processed: {self.agent.processed}",
            f"GC interval: {self.gc_interval:.0f}s    ingest: {self.ingest_interval:.1f}s    search: {self.search_interval:.1f}s",
        ]
        for ln in lines:
            self.screen.blit(self.f.render(ln, True, MUTED), (x, y))
            y += 18

    def _draw_console(self) -> None:
        r = self.console_rect
        self._panel(r, "")
        self.screen.blit(self.f_h.render("ACTIVITY", True, TEXT), (r.x + 12, r.y + 6))
        y = r.y + 26
        for msg, col in list(self.console):
            self.screen.blit(self.f_mono.render(msg, True, col), (r.x + 12, y))
            y += 15

    def _draw_help(self) -> None:
        lines = [
            "SPACE ingest   B batch   A auto   G run GC   S search past",
            "R record .mov   T subtitles   P pause   +/- speed   H hide help   ESC quit",
        ]
        w = 560
        box = pygame.Rect(self.graph_rect.x + 12, self.graph_rect.bottom - 48, w, 38)
        s = pygame.Surface(box.size, pygame.SRCALPHA)
        s.fill((10, 12, 18, 220))
        self.screen.blit(s, box.topleft)
        pygame.draw.rect(self.screen, BORDER, box, 1, border_radius=6)
        for i, ln in enumerate(lines):
            self.screen.blit(self.f_sm.render(ln, True, TEXT), (box.x + 10, box.y + 5 + i * 16))

    @staticmethod
    def _wrap(text: str, font: "pygame.font.Font", max_w: int) -> List[str]:
        lines: List[str] = []
        cur = ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if font.size(trial)[0] <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def _active_subtitle(self) -> Optional[str]:
        t = self._wall_elapsed - self.intro_seconds   # dashboard clock starts after intro
        for start, end, text in SUBTITLE_SCRIPT:
            if start <= t < end:
                return text
        return None

    def _blit_caption(self, text: str, centerx: int, bottom_y: int, max_w: int) -> None:
        lines = self._wrap(text, self.f_sub, max_w)
        line_h = self.f_sub.get_height() + 3
        pad_x, pad_y = 18, 12
        box_w = max(self.f_sub.size(ln)[0] for ln in lines) + pad_x * 2
        box_h = line_h * len(lines) + pad_y * 2
        bx = centerx - box_w // 2
        by = bottom_y - box_h

        s = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        s.fill((8, 10, 16, 232))
        self.screen.blit(s, (bx, by))
        pygame.draw.rect(self.screen, BORDER, pygame.Rect(bx, by, box_w, box_h), 1, border_radius=8)
        pygame.draw.rect(self.screen, ACCENT, pygame.Rect(bx, by, 4, box_h),
                         border_top_left_radius=8, border_bottom_left_radius=8)
        for i, ln in enumerate(lines):
            surf = self.f_sub.render(ln, True, (240, 244, 252))
            self.screen.blit(surf, (centerx - surf.get_width() // 2, by + pad_y + i * line_h))

    def _draw_subtitles(self) -> None:
        text = self._active_subtitle()
        if text:
            self._blit_caption(text, self.graph_rect.centerx,
                               self.graph_rect.bottom - 16,
                               min(self.graph_rect.width - 80, 960))

    # ---- animated Lorenz intro scene ------------------------------------
    def _draw_lorenz_scene(self) -> None:
        self.screen.fill(BG)
        cx, cy = WIDTH // 2, HEIGHT // 2 - 10
        scale = 11.5
        theta = self._wall_elapsed * 0.45            # slow rotation about the z-axis
        ct, st = math.cos(theta), math.sin(theta)
        pts = self._lorenz_hi
        n = len(pts)

        def project(p):
            x, y, z = p
            xr = x * ct - y * st                      # rotate (x,y) -> 3D parallax
            return (cx + xr * scale, cy - (z - 25.0) * scale)

        # faint full attractor path, colored by wing
        screen_pts = [project(p) for p in pts]
        step = 2
        for i in range(0, n - step, step):
            wx = pts[i][0]
            base = CLASS_COLOR["KEEP"] if wx < 0 else CLASS_COLOR["EVICT"]
            pygame.draw.line(self.screen, blend(base, BG, 0.22), screen_pts[i], screen_pts[i + step], 1)

        # bright comet tracers sweeping along the trajectory
        speed_pts = 900                               # trajectory points advanced per second
        for offset in (0, n // 3, 2 * n // 3):
            head = (int(self._wall_elapsed * speed_pts) + offset) % n
            trail = 90
            for j in range(trail):
                idx = (head - j) % n
                if idx + 1 >= n:
                    continue
                a = (1.0 - j / trail)
                wx = pts[idx][0]
                base = CLASS_COLOR["KEEP"] if wx < 0 else CLASS_COLOR["EVICT"]
                col = blend((255, 255, 255), base, 0.55 * a)
                pygame.draw.line(self.screen, blend(col, BG, 0.25 + 0.75 * a),
                                 screen_pts[idx], screen_pts[(idx + 1) % n], 2 if a > 0.4 else 1)
            hx, hy = screen_pts[head]
            pygame.draw.circle(self.screen, (255, 255, 255), (int(hx), int(hy)), 4)

        # title + equations
        title = self.f_title.render("THE LORENZ ATTRACTOR", True, TEXT)
        self.screen.blit(title, (cx - title.get_width() // 2, 40))
        sub = self.f.render("the chaotic core of OGrE's garbage collector", True, MUTED)
        self.screen.blit(sub, (cx - sub.get_width() // 2, 70))
        eqs = ["dx/dt = \u03c3 (y \u2212 x)", "dy/dt = x (\u03c1 \u2212 z) \u2212 y",
               "dz/dt = x y \u2212 \u03b2 z", "\u03c3=10   \u03c1=28   \u03b2=8/3"]
        ey = 100
        for i, e in enumerate(eqs):
            col = ACCENT if i == len(eqs) - 1 else TEXT
            surf = self.f_mono.render(e, True, col)
            self.screen.blit(surf, (40, ey + i * 18))
        # wing labels
        self.screen.blit(self.f_h.render("LEFT WING \u2192 KEEP", True, CLASS_COLOR["KEEP"]),
                         (60, HEIGHT - 90))
        rt = self.f_h.render("RIGHT WING \u2192 EVICT", True, CLASS_COLOR["EVICT"])
        self.screen.blit(rt, (WIDTH - rt.get_width() - 60, HEIGHT - 90))

        if self.subtitles_on:
            for start, end, text in INTRO_SCRIPT:
                if start <= self._wall_elapsed < end:
                    self._blit_caption(text, cx, HEIGHT - 30, 1000)
                    break

    # ---- events ---------------------------------------------------------
    def handle_events(self) -> None:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif k == pygame.K_SPACE:
                    self._ingest_one()
                elif k == pygame.K_b:
                    self._ingest_batch(self.rng.randint(2, 4))
                elif k == pygame.K_a:
                    self.auto_ingest = not self.auto_ingest
                    self.log(f"auto-ingest {'ON' if self.auto_ingest else 'OFF'}", ACCENT)
                elif k == pygame.K_g:
                    self._start_gc()
                    self.log("manual GC pass triggered", ACCENT)
                elif k == pygame.K_s:
                    self._do_search()
                elif k == pygame.K_p:
                    self.paused = not self.paused
                elif k == pygame.K_h:
                    self.show_help = not self.show_help
                elif k == pygame.K_t:
                    self.subtitles_on = not self.subtitles_on
                elif k == pygame.K_r:
                    self.toggle_recording()
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.speed = min(8.0, self.speed + 0.5)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed = max(0.5, self.speed - 0.5)

    # ---- main loop ------------------------------------------------------
    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()
            if self.recorder is not None:
                self.recorder.maybe_capture(self.screen, dt)
                if self.record_seconds and (time.time() - self._record_started_at) >= self.record_seconds:
                    self.stop_recording()
                    self.running = False
        self.stop_recording()
        self.agent.stop()
        pygame.quit()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="OGrE work-intelligence pygame simulation")
    parser.add_argument("--model", default=None, help="Ollama model (default: env OGRE_MODEL or llama3.2:3b)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--record", metavar="FILE.mov", default=None,
                        help="record the run to a .mov file")
    parser.add_argument("--record-seconds", type=float, default=0.0,
                        help="auto-stop recording and quit after N seconds")
    parser.add_argument("--record-fps", type=int, default=30, help="output video fps")
    parser.add_argument("--headless", action="store_true",
                        help="run without a visible window (still records video)")
    parser.add_argument("--subtitles", action="store_true",
                        help="show timed narration captions explaining each phase")
    parser.add_argument("--intro", type=float, default=0.0,
                        help="seconds of animated full-screen Lorenz attractor intro")
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    SimApp(
        model=args.model, seed=args.seed,
        record_path=args.record, record_seconds=args.record_seconds,
        record_fps=args.record_fps, subtitles=args.subtitles, intro=args.intro,
    ).run()


if __name__ == "__main__":
    main()
