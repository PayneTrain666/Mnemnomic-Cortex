"""
Plain-language summary
----------------------
What this file is for: Schedules when consolidation should run.
How it fits in the system: Timing layer above brokers.
Status: OPT-IN / ACTIVE when scheduled consolidation used
Important notes for non-coders: Does not store memories itself.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class SchedCfg:
    interval_sec: int = 60
    max_merges_per_tick: int = 256
    max_index_updates_per_tick: int = 512


class ConsolidationScheduler:
    """
    Budgeted scheduler for consolidation merges, nudges, and index refreshes.
    """

    def __init__(self, cfg: SchedCfg = SchedCfg()):
        self.cfg = cfg
        self._last_tick = 0.0

    def tick(
        self,
        pending_merges: Callable[[], List[Tuple]],
        nudge_keys: Callable[[], List[str]],
        reindex_keys: Callable[[], List[str]],
        broker,
        cms_index=None,
        now: float = None,
    ):
        ts = time.time() if now is None else float(now)
        if ts - self._last_tick < self.cfg.interval_sec:
            return

        for key, cand_view, importance, src_info in pending_merges()[: self.cfg.max_merges_per_tick]:
            broker.ingest_from_ltm(key, cand_view, importance=importance, src_info=src_info)

        for key in nudge_keys()[: self.cfg.max_merges_per_tick]:
            broker.cms_pull_to_cps(key)
            broker.cps_push_to_cms(key)

        if cms_index is not None:
            cms_index.add_or_update(reindex_keys()[: self.cfg.max_index_updates_per_tick])
        self._last_tick = ts

