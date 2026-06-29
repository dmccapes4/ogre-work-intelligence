"""Headless smoke test: drive the sim hard for many frames, force GC churn,
fill+overflow the eviction log, and exercise search/resurrection. No window."""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from ogre.sim import SimApp


def run():
    app = SimApp(seed=3)
    app.speed = 6.0
    app.gc_interval = 1.0
    app.ingest_interval = 0.4
    app.search_interval = 0.8

    max_live = 0
    for frame in range(1500):
        app.update(1 / 60)
        max_live = max(max_live, len(app.graph))
        if app._gc_snapshot_result is not None or app.gc_running:
            # let any in-flight GC thread finish before polling-heavy frames
            pass
        app.draw()
        if frame % 300 == 0:
            # block for pending GC to apply deterministically
            import time
            for _ in range(50):
                app._collect_enriched()
                app._poll_gc()
                if not app.gc_running:
                    break
                time.sleep(0.01)
            c = app.graph.counts()
            print(f"frame {frame:4d}: clock={app.graph.clock:6.1f}s nodes={len(app.graph):3d} "
                  f"edges={app.graph.edge_count():3d} KEEP={c['KEEP']} REVIEW={c['REVIEW']} "
                  f"evicted={app.gc.total_evicted} log={len(app.gc.eviction_log)} "
                  f"resurrected={app.gc.total_resurrected} forgotten={app.gc.total_forgotten}")

    print("\nFINAL")
    print("  max live nodes (post-update):", max_live, "(budget", app.max_nodes, "+ slack)")
    print("  passes:", app.gc_passes)
    print("  processed:", app.agent.processed, "agent status:", app.agent.status)
    print("  eviction log size:", len(app.gc.eviction_log))
    print("  total evicted:", app.gc.total_evicted,
          "resurrected:", app.gc.total_resurrected,
          "forgotten:", app.gc.total_forgotten)
    assert app.gc_passes > 0, "GC never ran"
    assert app.agent.processed > 0, "agent never processed an item"
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    run()
