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

    Draws are without replacement via per-stratum shuffled queues; a queue is
    refilled and reshuffled only when empty. Visit counters seed the partition
    draws and are checkpoint state, so a resumed run replays the same partitions.
    """

    def __init__(self, window_ids: list[int], M: int, W: int, rng: np.random.Generator):
        self.M, self.W, self.rng = M, W, rng
        self.strata = split_strata(window_ids, M)
        if any(len(s) < 1 for s in self.strata):
            raise ValueError(f"stratum with no windows: sizes={[len(s) for s in self.strata]}")
        ordered = sorted(window_ids)
        self.queues: list[list[int]] = [[] for _ in range(M)]
        self.visits: dict[int, int] = {w: 0 for w in ordered}

    def draw(self) -> list[tuple[int, int]]:
        """One step's windows: [(window_id, visit_counter)], W per stratum, oldest stratum first."""
        out = []
        for k in range(self.M):
            for _ in range(self.W):
                if not self.queues[k]:
                    self.queues[k] = list(self.rng.permutation(self.strata[k]))
                w = int(self.queues[k].pop())
                self.visits[w] += 1
                out.append((w, self.visits[w]))
        return out

    def state_dict(self) -> dict:
        return {
            "queues": [list(q) for q in self.queues],
            "visits": dict(self.visits),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, st: dict) -> None:
        self.queues = [list(q) for q in st["queues"]]
        self.visits = {int(k): int(v) for k, v in st["visits"].items()}
        self.rng.bit_generator.state = st["rng_state"]


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
