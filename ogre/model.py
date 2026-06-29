"""Core graph model: nodes, 5D-connascence edges, temporal decay + propagation.

The "physics" here is a coupled decay/propagation system. Each node has a
``vitality`` in [0, 1] that decays exponentially every tick but is replenished
by the vitality of its neighbours (weighted by edge type and strength). Densely
inter-referenced clusters therefore reinforce one another and saturate near 1 —
they become *stable attractors*. Weakly connected nodes bleed out toward 0 and
become eviction candidates.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

# 5D connascence — edge types and their relative coupling weight.
# Mirrors provenance-engine's CONNASCENCE_WEIGHTS so the GC sees a familiar shape.
CONNASCENCE_WEIGHTS: Dict[str, float] = {
    "STRUCTURAL": 1.2,
    "CONCEPTUAL": 1.0,
    "CO_VARIANCE": 0.8,
    "CO_OCCURRENCE": 0.6,
    "TEMPORAL": 0.4,
}

# RGB colors for each connascence dimension (used by the renderer).
CONNASCENCE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "STRUCTURAL": (255, 92, 92),    # red    — hard dependency
    "CONCEPTUAL": (96, 165, 250),   # blue   — shared meaning
    "CO_VARIANCE": (167, 139, 250),  # violet — move together
    "CO_OCCURRENCE": (52, 211, 153),  # green  — seen together
    "TEMPORAL": (251, 191, 36),     # amber  — close in time
}

IMPORTANCE_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.2}

# Decay / propagation tuning.
DECAY_RATE = 0.18          # per simulation-second exponential bleed
PROP_GAIN = 0.55           # how strongly neighbour vitality replenishes a node
REFERENCE_BOOST = 1.0      # vitality a node is set to when referenced


def _id_factory():
    counter = itertools.count(1)
    return lambda: f"wi-{next(counter):04d}"


_next_id = _id_factory()


@dataclass
class Edge:
    """A single connascence edge from one node to another."""

    target: str
    etype: str
    strength: float
    created_at: float = field(default_factory=time.time)
    reinforced_at: float = field(default_factory=time.time)

    @property
    def weight(self) -> float:
        return CONNASCENCE_WEIGHTS.get(self.etype, 0.5) * self.strength


@dataclass
class Node:
    """A work item living in the graph."""

    text: str
    concepts: Set[str] = field(default_factory=set)
    importance: str = "medium"
    load_bearing: bool = False
    id: str = field(default_factory=_next_id)

    sim_created_at: float = 0.0       # simulation clock when ingested
    referenced_at: float = 0.0        # simulation clock of last reference
    references: int = 0               # total reference count
    batch: int = 0                    # ingestion batch (drives CO_OCCURRENCE)

    vitality: float = 1.0
    classification: str = "REVIEW"
    mean_x: float = 0.0
    confidence: float = 0.0

    edges: Dict[str, Edge] = field(default_factory=dict)

    # Layout / render state.
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    flash: float = 0.0                # transient highlight (0..1)
    resurrected: bool = False

    @property
    def importance_weight(self) -> float:
        return IMPORTANCE_WEIGHT.get(self.importance, 0.6)

    @property
    def degree(self) -> int:
        return len(self.edges)

    @property
    def label(self) -> str:
        if self.concepts:
            return sorted(self.concepts)[0]
        return self.text[:18]

    def connascence_strength(self) -> float:
        if not self.edges:
            return 0.0
        return sum(e.weight for e in self.edges.values()) / len(self.edges)


class WorkGraph:
    """The live, stable work-intelligence graph."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.clock: float = 0.0       # simulation seconds elapsed

    # ---- membership -----------------------------------------------------
    def __contains__(self, nid: str) -> bool:
        return nid in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def values(self) -> Iterable[Node]:
        return self.nodes.values()

    def get(self, nid: str) -> Optional[Node]:
        return self.nodes.get(nid)

    def add_node(self, node: Node) -> None:
        if node.sim_created_at == 0.0:
            node.sim_created_at = self.clock
        node.referenced_at = self.clock
        self.nodes[node.id] = node

    def remove_node(self, nid: str) -> Optional[Node]:
        node = self.nodes.pop(nid, None)
        if node is None:
            return None
        # Drop dangling edges that pointed at the removed node.
        for other in self.nodes.values():
            other.edges.pop(nid, None)
        return node

    # ---- edges ----------------------------------------------------------
    def _link(self, a: str, b: str, etype: str, strength: float) -> None:
        """Create or reinforce a symmetric edge between two nodes."""
        if a == b or a not in self.nodes or b not in self.nodes:
            return
        strength = max(0.05, min(1.0, strength))
        for src, dst in ((a, b), (b, a)):
            node = self.nodes[src]
            existing = node.edges.get(dst)
            if existing is None or CONNASCENCE_WEIGHTS.get(etype, 0) >= CONNASCENCE_WEIGHTS.get(existing.etype, 0):
                if existing is not None:
                    strength = max(strength, existing.strength)
                node.edges[dst] = Edge(target=dst, etype=etype, strength=strength)
            else:
                # Keep the stronger-typed edge but reinforce its strength.
                existing.strength = min(1.0, max(existing.strength, strength * 0.6))
                existing.reinforced_at = self.clock

    # ---- OGrE: opportunistic graph enrichment ---------------------------
    def ogre_enrich(self, node: Node, conceptual_thresh: float = 0.18) -> List[str]:
        """Opportunistically wire a (new or resurrected) node into the graph.

        Returns the list of node-ids it connected to. This is the heart of
        OGrE: every time we *touch* a node we look for cheap, high-signal
        edges to weave, across all five connascence dimensions.
        """
        connected: List[str] = []
        if node.id not in self.nodes:
            return connected

        text_lower = node.text.lower()
        for other in list(self.nodes.values()):
            if other.id == node.id:
                continue

            # CONCEPTUAL — shared concept vocabulary (Jaccard).
            if node.concepts and other.concepts:
                inter = node.concepts & other.concepts
                union = node.concepts | other.concepts
                jac = len(inter) / len(union) if union else 0.0
                if jac >= conceptual_thresh:
                    self._link(node.id, other.id, "CONCEPTUAL", 0.4 + jac)
                    connected.append(other.id)

            # STRUCTURAL — explicit textual reference to the other item.
            mentioned = any(c in text_lower for c in other.concepts if len(c) > 3)
            if mentioned or other.id in text_lower:
                self._link(node.id, other.id, "STRUCTURAL", 0.8)
                connected.append(other.id)

            # CO_OCCURRENCE — ingested in the same batch / session.
            if other.batch == node.batch and node.batch > 0:
                self._link(node.id, other.id, "CO_OCCURRENCE", 0.6)
                connected.append(other.id)

            # TEMPORAL — ingested close together in sim-time.
            dt = abs(other.sim_created_at - node.sim_created_at)
            if dt < 4.0:
                self._link(node.id, other.id, "TEMPORAL", 0.7 * math.exp(-dt / 3.0))
                connected.append(other.id)

            # CO_VARIANCE — similar importance *and* both actively referenced.
            if (other.importance == node.importance
                    and node.references > 0 and other.references > 0):
                self._link(node.id, other.id, "CO_VARIANCE", 0.5)
                connected.append(other.id)

        # Keep the graph legible: cap fan-out to the strongest edges.
        self._prune_edges(node.id, max_edges=6)
        for cid in set(connected):
            self._prune_edges(cid, max_edges=6)
        return list(set(connected))

    def _prune_edges(self, nid: str, max_edges: int) -> None:
        node = self.nodes.get(nid)
        if node is None or len(node.edges) <= max_edges:
            return
        ranked = sorted(node.edges.values(), key=lambda e: e.weight, reverse=True)
        keep = {e.target for e in ranked[:max_edges]}
        for tgt in list(node.edges):
            if tgt not in keep:
                node.edges.pop(tgt, None)
                other = self.nodes.get(tgt)
                if other:
                    other.edges.pop(nid, None)

    # ---- references -----------------------------------------------------
    def reference(self, nid: str) -> bool:
        """Record a reference: reset vitality, bump counters, reinforce edges."""
        node = self.nodes.get(nid)
        if node is None:
            return False
        node.vitality = REFERENCE_BOOST
        node.references += 1
        node.referenced_at = self.clock
        node.flash = 1.0
        for e in node.edges.values():
            e.strength = min(1.0, e.strength + 0.05)
            e.reinforced_at = self.clock
        return True

    # ---- temporal decay + propagation (the attractor dynamics) ----------
    def decay_step(self, dt: float) -> None:
        """Advance vitality one tick: exponential decay then edge propagation."""
        self.clock += dt
        if not self.nodes:
            return

        decay = math.exp(-DECAY_RATE * dt)
        snapshot = {nid: n.vitality for nid, n in self.nodes.items()}

        for node in self.nodes.values():
            decayed = snapshot[node.id] * decay
            # Propagation: neighbours lend vitality, weighted by connascence.
            support = 0.0
            for e in node.edges.values():
                support += e.weight * snapshot.get(e.target, 0.0)
            decayed += PROP_GAIN * support * dt
            node.vitality = max(0.0, min(1.0, decayed))
            if node.flash > 0.0:
                node.flash = max(0.0, node.flash - dt * 1.5)

    # ---- export for the Lorenz GC --------------------------------------
    def to_pe_nodes(self) -> List[dict]:
        """Map live graph state into provenance-engine's direct-value format.

        x0 (structural_connectivity) ← normalized degree
        y0 (connascence_strength)    ← weighted mean edge strength
        z0 (temporal_vitality)       ← live propagated vitality
        """
        if not self.nodes:
            return []
        max_deg = max((n.degree for n in self.nodes.values()), default=1) or 1
        out = []
        for n in self.nodes.values():
            out.append({
                "id": n.id,
                "structural_connectivity": n.degree / max_deg,
                "connascence_strength": n.connascence_strength(),
                "temporal_vitality": n.vitality,
                "importance": n.importance,
                "load_bearing": n.load_bearing,
                "edges": [
                    {"target": e.target, "type": e.etype, "strength": e.strength}
                    for e in n.edges.values()
                ],
            })
        return out

    # ---- stats ----------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        c = {"KEEP": 0, "REVIEW": 0, "EVICT": 0}
        for n in self.nodes.values():
            c[n.classification] = c.get(n.classification, 0) + 1
        return c

    def edge_count(self) -> int:
        return sum(n.degree for n in self.nodes.values()) // 2
