# OGrE — Opportunistic Graph-enrichment Work Intelligence

A `pygame` simulation of **graph-based work intelligence**. Work items are
ingested by a **local Ollama agent**, woven into a graph by **5D-connascence
edges**, kept alive by **temporal decay + propagation** (forming stable
attractors), and garbage-collected by a **Lorenz attractor GC**
([`provenance-engine`](https://pypi.org/project/provenance-engine/)). Evictions
fall into a **100-item FIFO eviction log**, and a searching agent can
**resurrect** evicted items back into the graph.

```
work item ──▶ Ollama agent ──▶ OGrE enrichment ──▶ stable graph
                (enrich)         (5D edges)            │
                                                 decay + propagation
                                                       │ (attractors)
                                                 Lorenz GC (RK4)
                                              KEEP / REVIEW / EVICT
                                                       │
                                              EVICT ──▶ FIFO log (100)
                                                       │      │ overflow → forgotten
                                       search ◀────────┘      
                                  graph hit  OR  resurrect from log
```

## Quick start

```bash
./run.sh                      # creates venv, installs deps, launches
# or
./run.sh --model qwen3:8b     # pick any local Ollama model
```

Requires a local [Ollama](https://ollama.com) server. The default model is
`qwen2.5-coder:14b-instruct-q4_K_M` — the strongest model that fits in 12 GB of
VRAM (~9.3 GB resident, 100% on GPU), chosen for high-quality concept/importance
extraction. If Ollama is unreachable the agent transparently falls back to a
deterministic heuristic so the simulation always runs.

```bash
OGRE_MODEL=qwen3:8b ./run.sh                  # lighter / faster
OGRE_MODEL=llama3.2:3b ./run.sh               # smallest, snappiest
```

### Model selection (12 GB VRAM)

| Model | VRAM | Notes |
| --- | --- | --- |
| `qwen2.5-coder:14b-instruct-q4_K_M` | ~9.3 GB | **default** — richest extraction, ~2.2s/item warm |
| `qwen3:8b` | ~5.2 GB | strong, faster |
| `qwen2.5-coder:7b-instruct-q4_K_M` | ~4.7 GB | fast |
| `llama3.2:3b` | ~2.0 GB | snappiest |

On startup the agent **pre-warms** the model (loads it into VRAM) before
ingesting, so the cold load isn't mistaken for a stalled request. The top bar
shows the agent state: `loading` → `ollama` (live) / `busy`, or `heuristic` if
it falls back. Auto-ingest applies backpressure so it never outruns a slower
model. A single slow/failed call is tolerated; it only drops to the heuristic
after several consecutive failures, and re-probes to recover.

## Controls

| Key | Action |
| --- | --- |
| `SPACE` | ingest one work item |
| `B` | ingest a correlated batch (a "sprint") |
| `A` | toggle auto-ingest |
| `G` | run a Lorenz GC pass now |
| `S` | search the past (graph → eviction log, resurrects on hit) |
| `R` | start/stop recording a `.mov` |
| `T` | toggle narration subtitles |
| `P` | pause | 
| `+` / `-` | simulation speed |
| `H` | toggle help overlay |
| `ESC` / `Q` | quit |

## Recording a video (.mov)

Recording is built in — no system `ffmpeg` needed (it uses the `ffmpeg` binary
bundled by `imageio-ffmpeg`).

```bash
# interactive: launch, then press R to start/stop. A blinking REC + timer shows
./run.sh                                   # press R to toggle; file = ogre-capture-<timestamp>.mov

# record a fixed clip to a named file, then keep running
./run.sh --record demo.mov

# record exactly 30s and auto-quit (great for a deterministic clip)
./run.sh --record demo.mov --record-seconds 30 --record-fps 30

# no window at all — render + record straight to file (works over SSH)
./run.sh --headless --record demo.mov --record-seconds 30

# narrated explainer with an animated Lorenz-attractor intro (~2.7 min)
./run.sh --headless --subtitles --intro 15 --record demo.mov --record-seconds 161
```

`--intro N` opens with an N-second full-screen animated Lorenz attractor — the
real butterfly with comet tracers sweeping both wings (green left = KEEP, red
right = EVICT) — before cutting to the live dashboard.

### Voice narration (optional)

`build_narration.py` speaks the subtitle track with a local [Piper](https://github.com/OHF-Voice/piper1-gpl)
neural voice (offline) and lays it on a timeline matching the captions. It's a
neutral narrator reading the on-screen text — not an impersonation of anyone.

```bash
pip install piper-tts
# download a voice (e.g. en_US-ryan-high) into voices/
python build_narration.py --intro 20 --duration 166 --out narration.wav
# mux into the recording
ffmpeg -i ogre-demo.mov -i narration.wav -map 0:v -map 1:a \
       -c:v libx264 -crf 26 -pix_fmt yuv420p -c:a aac -movflags +faststart out.mp4
```

`--subtitles` (or the `T` key) overlays a timed narration track that explains
each phase — ingestion, OGrE enrichment, decay/attractors, Lorenz classification,
eviction, forgetting, and resurrection — keyed to wall-clock time.

Frames are sampled at a fixed wall-clock `--record-fps` (default 30), so the
output plays back in real time even if encoding slows the loop. Output is H.264
in a QuickTime `.mov` container.

## What you're looking at

- **Stable work graph** (left): force-directed layout. Node colour = GC verdict
  (green KEEP, amber REVIEW, red EVICT-in-flight). Node size = importance +
  vitality; faded nodes are decaying. A white ring marks load-bearing items, a
  blue ring marks resurrected ones. Edge colour encodes the connascence
  dimension; thickness encodes strength.
- **Lorenz GC panel** (top right): the attractor's two wings. Each node is a dot
  placed by the settled mean-x of its Lorenz trajectory — left wing = KEEP,
  right wing = EVICT, the `±tau` band in the middle = REVIEW.
- **Eviction log** (middle right): the 100-slot FIFO. Newest on top; entries
  fade with age. When full, the oldest is *forgotten* forever.
- **System / Activity** panels: live counts, eviction rate, agent status, and a
  scrolling event console.

## The model

### 5D connascence edges
Every edge carries one of five connascence dimensions (weights mirror
`provenance-engine`):

| Dimension | Weight | Meaning | OGrE trigger |
| --- | --- | --- | --- |
| `STRUCTURAL` | 1.2 | hard dependency | one item names another |
| `CONCEPTUAL` | 1.0 | shared meaning | concept-set Jaccard overlap |
| `CO_VARIANCE` | 0.8 | move together | same importance + both referenced |
| `CO_OCCURRENCE` | 0.6 | seen together | ingested in the same batch |
| `TEMPORAL` | 0.4 | close in time | ingested near each other |

### OGrE — opportunistic graph enrichment
Whenever an item is ingested, referenced, or resurrected, the graph is cheaply
re-scanned for high-signal edges across all five dimensions (`WorkGraph.ogre_enrich`).
Fan-out is capped to keep clusters legible.

### Temporal decay → stable attractors
Each tick every node's `vitality` decays exponentially, then receives vitality
*propagated* from its neighbours (weighted by connascence). Densely
inter-referenced clusters reinforce each other and saturate near 1 — they become
**stable attractors**. Isolated items bleed toward 0. References
(`referenced_at`, `references`) reset vitality to 1.

### Lorenz GC
The live graph is mapped to Lorenz initial conditions and integrated with RK4
(via `provenance-engine`):

- `x0` ← structural connectivity (normalised degree)
- `y0` ← connascence strength (weighted mean edge strength)
- `z0` ← temporal vitality

The settled trajectory's wing decides KEEP / EVICT / REVIEW. Load-bearing EVICTs
are escalated to REVIEW.

### Eviction, forgetting, resurrection
EVICT nodes leave the graph for a 100-entry FIFO log. A capacity governor spills
the weakest nodes when the working set exceeds budget (default 72). Searching the
past checks the live graph first, then the log; a match in the log is
**resurrected** — re-added, re-enriched, vitality reset. Items that fall off the
FIFO tail are gone for good.

## Layout

```
ogre/
  model.py       # Node, Edge, WorkGraph: 5D edges, decay + propagation, OGrE
  agent.py       # threaded Ollama ingestion agent (+ heuristic fallback)
  gc_engine.py   # Lorenz GC, FIFO eviction log, search + resurrection
  workitems.py   # themed synthetic work-item corpus
  sim.py         # pygame app: rendering, layout, controls, main loop
main.py          # entry point
smoke_test.py    # headless stress test (no window)
```

## Config (env vars)

| Var | Default | Purpose |
| --- | --- | --- |
| `OGRE_MODEL` | `qwen2.5-coder:14b-instruct-q4_K_M` | Ollama model name |
| `OGRE_OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `OGRE_TIMEOUT` | `45` | per-request timeout (seconds) |
| `OGRE_WARMUP_TIMEOUT` | `240` | cold model-load timeout (seconds) |

## Headless test

```bash
OGRE_OLLAMA_URL=http://localhost:9 .venv/bin/python smoke_test.py
```

Drives ~150 sim-seconds at 6×, forcing the heuristic agent, and asserts the GC,
eviction log capacity, forgetting, and resurrection all behave.
