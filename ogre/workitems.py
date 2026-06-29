"""A themed corpus of synthetic work items.

Items are grouped into projects/themes so that conceptual + co-occurrence
edges naturally form clusters — which then become stable attractors under the
decay/propagation dynamics. A few deliberately isolated "noise" items are
included so the GC has clear eviction candidates.
"""

from __future__ import annotations

import random
from typing import List, Tuple

# (text, baseline importance, load_bearing). The agent may override importance.
CORPUS: List[Tuple[str, str, bool]] = [
    # --- Auth service cluster -------------------------------------------
    ("Implement OAuth2 token refresh in the auth service", "high", True),
    ("Auth service: rotate JWT signing keys nightly", "high", True),
    ("Fix session fixation vulnerability in login flow", "high", False),
    ("Add rate limiting to the auth login endpoint", "medium", False),
    ("Write integration tests for OAuth2 refresh tokens", "medium", False),
    ("Document the auth service token lifecycle", "low", False),
    ("Migrate auth user table to argon2 password hashes", "high", True),

    # --- Payments cluster -----------------------------------------------
    ("Payments: reconcile Stripe webhooks with ledger", "high", True),
    ("Add idempotency keys to the payment capture endpoint", "high", True),
    ("Investigate duplicate charge bug in payment retry path", "high", False),
    ("Build refund workflow for the payments service", "medium", False),
    ("Payments dashboard: show daily settlement totals", "low", False),
    ("Write load test for payment capture at 2k rps", "medium", False),

    # --- Data pipeline cluster ------------------------------------------
    ("Pipeline: backfill events table from S3 cold storage", "medium", False),
    ("Add schema validation to the ingestion pipeline", "high", True),
    ("Pipeline: dedupe events by idempotency key", "medium", False),
    ("Optimize the nightly aggregation job runtime", "medium", False),
    ("Pipeline: alert when ingestion lag exceeds 5 minutes", "high", False),

    # --- Frontend cluster ------------------------------------------------
    ("Frontend: dark mode toggle in settings", "low", False),
    ("Fix flaky checkout button race condition", "medium", False),
    ("Frontend: lazy-load the analytics dashboard", "low", False),
    ("Add accessibility labels to the nav menu", "low", False),
    ("Frontend: cache user profile in local storage", "low", False),

    # --- Infra / platform cluster ---------------------------------------
    ("Infra: migrate postgres to read replicas", "high", True),
    ("Set up blue-green deploy for the api gateway", "high", True),
    ("Infra: rotate TLS certificates before expiry", "high", True),
    ("Add prometheus metrics to the worker pool", "medium", False),
    ("Infra: tune autoscaling thresholds for the web tier", "medium", False),

    # --- Search / ML cluster --------------------------------------------
    ("Search: rebuild embedding index for product catalog", "medium", False),
    ("Tune BM25 weights for the search ranker", "low", False),
    ("Search: add typo tolerance to the query parser", "low", False),
    ("Evaluate vector recall on the eval set", "medium", False),

    # --- Isolated / one-off noise (likely eviction candidates) ----------
    ("Rename the temp variable in the legacy cron script", "low", False),
    ("Delete unused feature flag from 2019", "low", False),
    ("Update the office wifi password in the wiki", "low", False),
    ("Fix typo in the onboarding email subject line", "low", False),
    ("Archive the deprecated v1 changelog file", "low", False),
    ("Bump copyright year in the footer", "low", False),
    ("Remove commented-out code in utils.py", "low", False),
]


class WorkItemSource:
    """Yields work items, occasionally as a correlated batch (a 'sprint')."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.pool = list(CORPUS)
        self.rng.shuffle(self.pool)
        self.cursor = 0

    def _next_raw(self) -> Tuple[str, str, bool]:
        if self.cursor >= len(self.pool):
            self.rng.shuffle(self.pool)
            self.cursor = 0
        item = self.pool[self.cursor]
        self.cursor += 1
        return item

    def next_item(self) -> Tuple[str, str, bool]:
        return self._next_raw()

    def next_batch(self, size: int = 3) -> List[Tuple[str, str, bool]]:
        """A small cluster of related-by-time items (same ingestion batch)."""
        return [self._next_raw() for _ in range(size)]
