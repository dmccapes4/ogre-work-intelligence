"""Lorenz garbage collector + FIFO eviction log + resurrection-on-search.

The GC borrows provenance-engine's Portal (Lorenz attractor) classifier: each
node's (structural_connectivity, connascence_strength, temporal_vitality) is
mapped to a Lorenz initial condition, integrated with RK4, and classified by
which attractor wing the trajectory settles into:

    left wing  (mean_x < -tau)  -> KEEP    (consolidated)
    right wing (mean_x >  tau)  -> EVICT    (decayed)
    boundary                    -> REVIEW   (uncertain)

EVICT nodes are removed from the stable graph and pushed onto a 100-entry FIFO
eviction log. Searching the past walks the stable graph first, then the log;
a referenced eviction is resurrected back into the graph.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from provenance_engine import (
    build_graph,
    classify_node,
    integrate_portal,
    normalize_and_scale,
)

from .model import Node, WorkGraph

EVICTION_LOG_CAPACITY = 100

# Trim Lorenz integration so a full-graph GC pass stays snappy in a worker
# thread (600 steps; classifier still settles over its last 200).
GC_T_MAX = 6.0
GC_DT = 0.01


@dataclass
class EvictionRecord:
    node: Node
    evicted_at: float          # simulation clock
    wall_time: float = field(default_factory=time.time)
    reason: str = "lorenz-evict"


@dataclass
class GCResult:
    classifications: Dict[str, dict]      # node_id -> classify_node() output
    evicted_ids: List[str]
    rho: float
    tau: float
    duration: float
    dropped_from_log: int = 0             # records pushed off the FIFO tail


class LorenzGC:
    """Runs Portal classification and manages the eviction log."""

    def __init__(self, rho: float = 28.0, tau: float = 2.0) -> None:
        self.rho = rho
        self.tau = tau
        self.eviction_log: Deque[EvictionRecord] = deque(maxlen=EVICTION_LOG_CAPACITY)
        self.total_evicted = 0
        self.total_resurrected = 0
        self.total_forgotten = 0           # fell off the FIFO tail forever
        self.last_result: Optional[GCResult] = None
        self._busy = threading.Lock()

    # ---- classification (pure; safe to run in a worker thread) ----------
    def classify(self, pe_nodes: List[dict]) -> Dict[str, dict]:
        if not pe_nodes:
            return {}
        graph = build_graph(pe_nodes)
        scaled = normalize_and_scale(graph)
        results: Dict[str, dict] = {}
        for rec in scaled:
            traj = integrate_portal(
                rec["x0"], rec["y0"], rec["z0"],
                rho=self.rho, t_max=GC_T_MAX, dt=GC_DT,
            )
            results[rec["id"]] = classify_node(
                traj, tau=self.tau, load_bearing=rec.get("load_bearing", False)
            )
        return results

    def run(self, graph: WorkGraph) -> GCResult:
        """Full GC pass over a snapshot of the graph. Returns classifications.

        Note: this only *computes* classifications + the evict list. The caller
        applies eviction on the main thread (see ``apply``) to keep all graph
        mutation single-threaded.
        """
        t0 = time.time()
        pe_nodes = graph.to_pe_nodes()
        results = self.classify(pe_nodes)
        evicted = [nid for nid, r in results.items() if r["classification"] == "EVICT"]
        res = GCResult(
            classifications=results,
            evicted_ids=evicted,
            rho=self.rho,
            tau=self.tau,
            duration=time.time() - t0,
        )
        self.last_result = res
        return res

    # ---- mutation (main thread only) -----------------------------------
    def apply(self, graph: WorkGraph, result: GCResult) -> List[EvictionRecord]:
        """Stamp classifications onto nodes and evict EVICT nodes to the log."""
        for nid, r in result.classifications.items():
            node = graph.get(nid)
            if node is not None:
                node.classification = r["classification"]
                node.mean_x = r["mean_x"]
                node.confidence = r["confidence"]

        evicted_records: List[EvictionRecord] = []
        for nid in result.evicted_ids:
            node = graph.remove_node(nid)
            if node is None:
                continue
            node.classification = "EVICT"
            # If the log is full, the oldest record is forgotten forever.
            if len(self.eviction_log) == self.eviction_log.maxlen:
                self.total_forgotten += 1
                result.dropped_from_log += 1
            rec = EvictionRecord(node=node, evicted_at=graph.clock)
            self.eviction_log.append(rec)
            evicted_records.append(rec)
            self.total_evicted += 1
        return evicted_records

    def enforce_capacity(self, graph: WorkGraph, max_nodes: int) -> List[EvictionRecord]:
        """Soft governance cap: if the stable graph is over budget, evict the
        weakest (lowest-vitality, non-load-bearing) nodes into the FIFO log.

        This keeps the working set bounded — the same pressure a real memory
        system feels — and routes overflow through the normal eviction path so
        those items remain resurrectable.
        """
        over = len(graph) - max_nodes
        if over <= 0:
            return []
        # Prefer spilling weak, non-load-bearing items; only touch load-bearing
        # ones as a last resort to honour the hard resource bound.
        candidates = sorted(graph.values(), key=lambda n: (n.load_bearing, n.vitality))
        records: List[EvictionRecord] = []
        for node in candidates[:over]:
            removed = graph.remove_node(node.id)
            if removed is None:
                continue
            removed.classification = "EVICT"
            if len(self.eviction_log) == self.eviction_log.maxlen:
                self.total_forgotten += 1
            rec = EvictionRecord(node=removed, evicted_at=graph.clock, reason="capacity")
            self.eviction_log.append(rec)
            records.append(rec)
            self.total_evicted += 1
        return records

    # ---- search the past -----------------------------------------------
    def search(self, graph: WorkGraph, query: str) -> Tuple[str, Optional[Node]]:
        """Search stable graph first, then eviction log.

        Returns (outcome, node) where outcome is one of:
            'graph-hit'    — found live; referenced (vitality boosted)
            'resurrected'  — found in eviction log; brought back into graph
            'miss'         — not found anywhere
        """
        q = query.lower().strip()
        if not q:
            return ("miss", None)

        # 1) stable graph
        hit = self._match_in_graph(graph, q)
        if hit is not None:
            graph.reference(hit.id)
            return ("graph-hit", hit)

        # 2) eviction log (FIFO) — referenced evictions come back.
        for rec in list(self.eviction_log):
            if self._matches(rec.node, q):
                self._resurrect(graph, rec)
                return ("resurrected", rec.node)

        return ("miss", None)

    def _match_in_graph(self, graph: WorkGraph, q: str) -> Optional[Node]:
        best: Optional[Node] = None
        best_score = 0.0
        for node in graph.values():
            score = self._score(node, q)
            if score > best_score:
                best, best_score = node, score
        return best if best_score > 0 else None

    @staticmethod
    def _score(node: Node, q: str) -> float:
        score = 0.0
        if any(q == c or q in c or c in q for c in node.concepts):
            score += 2.0
        if q in node.text.lower():
            score += 1.0
        return score

    def _matches(self, node: Node, q: str) -> bool:
        return self._score(node, q) > 0

    def _resurrect(self, graph: WorkGraph, rec: EvictionRecord) -> None:
        try:
            self.eviction_log.remove(rec)
        except ValueError:
            pass
        node = rec.node
        node.vitality = 1.0
        node.classification = "REVIEW"
        node.references += 1
        node.resurrected = True
        node.flash = 1.0
        node.sim_created_at = graph.clock
        node.edges.clear()           # rebuilt by OGrE
        graph.add_node(node)
        graph.reference(node.id)
        graph.ogre_enrich(node)
        self.total_resurrected += 1

    # ---- query terms (for the automatic "agent searches the past") -----
    def sample_query_term(self, graph: WorkGraph, rng) -> Optional[str]:
        """Pick a concept to search for, biased toward recently-evicted items so
        resurrection is visibly demonstrated."""
        if self.eviction_log and rng.random() < 0.55:
            rec = rng.choice(list(self.eviction_log))
            if rec.node.concepts:
                return rng.choice(sorted(rec.node.concepts))
        live_concepts: List[str] = []
        for node in graph.values():
            live_concepts.extend(node.concepts)
        if live_concepts:
            return rng.choice(live_concepts)
        return None
