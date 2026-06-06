"""Stratified window sampling and seeded group-partition drawing."""
from __future__ import annotations

import numpy as np


def split_strata(window_ids: list[int], M: int) -> list[list[int]]:
    """M contiguous blocks over the sorted window ids, equal counts (+-1), oldest first.

    Shared by the training sampler and the stratified consistency metric, so
    "spread like training" means exactly the same block boundaries."""
    ordered = sorted(window_ids)
    sizes = [len(ordered) // M + (1 if i < len(ordered) % M else 0) for i in range(M)]
    out: list[list[int]] = []
    pos = 0
    for s in sizes:
        out.append(ordered[pos : pos + s])
        pos += s
    return out


class StratumSampler:
    """M contiguous strata over the usable-window sequence; W draws per stratum per step.

    Draws are without replacement via per-stratum shuffled queues. Queues are
    refilled and reshuffled **synchronously** — when any queue cannot serve W
    draws, all queues restart together (at most one leftover window per stratum
    is discarded) — defining an epoch. One tiling offset in [0, offset_range)
    is drawn per epoch, so a whole pass over the timeline uses a single tiling.
    Visit counters seed the partition draws; counters, offset, and RNG state are
    checkpoint state, so a resumed run replays the same draws.
    """

    def __init__(self, window_ids: list[int], M: int, W: int, rng: np.random.Generator, offset_range: int = 0):
        self.M, self.W, self.rng = M, W, rng
        self.strata = split_strata(window_ids, M)
        if any(len(s) < W for s in self.strata):
            raise ValueError(f"stratum smaller than W={W}: sizes={[len(s) for s in self.strata]}")
        self.queues: list[list[int]] = [[] for _ in range(M)]
        self.visits: dict[int, int] = {w: 0 for w in sorted(window_ids)}
        self.offset_range = offset_range
        self.offset = 0

    def draw(self) -> list[tuple[int, int]]:
        """One step's windows: [(window_id, visit_counter)], W per stratum, oldest stratum first."""
        if any(len(q) < self.W for q in self.queues):
            # epoch boundary: synchronized refill; one offset for the whole epoch
            self.queues = [list(self.rng.permutation(s)) for s in self.strata]
            if self.offset_range > 0:
                self.offset = int(self.rng.integers(0, self.offset_range))
        out = []
        for k in range(self.M):
            for _ in range(self.W):
                w = int(self.queues[k].pop())
                self.visits[w] += 1
                out.append((w, self.visits[w]))
        return out

    def state_dict(self) -> dict:
        return {
            "queues": [list(q) for q in self.queues],
            "visits": dict(self.visits),
            "rng_state": self.rng.bit_generator.state,
            "offset": self.offset,
        }

    def load_state_dict(self, st: dict) -> None:
        self.queues = [list(q) for q in st["queues"]]
        self.visits = {int(k): int(v) for k, v in st["visits"].items()}
        self.rng.bit_generator.state = st["rng_state"]
        self.offset = int(st.get("offset", 0))  # legacy checkpoints: fixed tiling


def draw_partitions(universe_size: int, group_counts: list[int], seed: tuple[int, int]) -> dict[int, list[np.ndarray]]:
    """Fresh uniformly random equal-size (+-1) partitions, one per scale.

    Seeded by (window id, visit counter); one RNG serves the whole ladder.
    Returns {group_count: [index arrays into the universe]}.
    """
    rng = np.random.default_rng(seed)
    out: dict[int, list[np.ndarray]] = {}
    for n in group_counts:
        perm = rng.permutation(universe_size)
        base, rem = divmod(universe_size, n)
        groups, pos = [], 0
        for gi in range(n):
            size = base + (1 if gi < rem else 0)
            groups.append(np.sort(perm[pos : pos + size]))
            pos += size
        out[n] = groups
    return out
