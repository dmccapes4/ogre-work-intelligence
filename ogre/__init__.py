"""OGrE — Opportunistic Graph-enrichment Work Intelligence simulation.

A pygame simulation of graph-based work intelligence:

  * Work items are ingested by a local Ollama agent.
  * Items connect via 5D-connascence edges (STRUCTURAL, CONCEPTUAL,
    TEMPORAL, CO_OCCURRENCE, CO_VARIANCE).
  * Vitality decays over time but propagates along edges, so densely
    connected clusters form *stable attractors*.
  * A Lorenz-attractor garbage collector (provenance-engine) classifies
    every node KEEP / REVIEW / EVICT.
  * Evictions drop into a 100-item FIFO eviction log. Searching the past
    walks the stable graph first, then the log; referenced evictions are
    resurrected back into the graph.
"""

from .model import (
    Node,
    Edge,
    WorkGraph,
    CONNASCENCE_WEIGHTS,
    CONNASCENCE_COLORS,
)

__all__ = [
    "Node",
    "Edge",
    "WorkGraph",
    "CONNASCENCE_WEIGHTS",
    "CONNASCENCE_COLORS",
]
