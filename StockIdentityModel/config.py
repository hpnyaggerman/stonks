"""Configuration for the Stock Identity Encoder.

Tuning doctrine — paid for in runs r1-r10; read before changing any knob:

1. Change a WORKING config only on (a) measured harm or (b) a mechanism-backed
   hypothesis with a predicted observable. "Looks inefficient" is neither.
   Both r8 changes optimized away equilibrium features of the best run on
   record: util never reaching v0 looked like waste but was the inflation
   mechanism holding the space open (r8: per-dim variance crashed 100x under
   the floor, 28/32 dims dead); clip_norm=1.0 firing on 100% of steps looked
   mislabeled but is always-on gradient normalization, load-bearing for the
   late-run grind (r9: clip=25 ran ahead early, then stalled 15k steps at
   retrieval 0.5 vs r7's 0.8). A fence that is always leaned on is not
   redundant — it is the thing holding the shape.
2. One variable per run, same train_seed. Paired runs then share the window
   schedule, partitions, and dropout draws, so eval differences are
   attributable. r8 bundled two changes and needed r9 to un-confound them.
3. Geometry knobs (v0, m_sep, D) shift the balance of power toward the
   scale-blind loss terms: anc is l1 (constant grip at any distance), syn is
   a ratio (blind to size) — neither weakens when the space shrinks, while
   the hinge fences and util have bounded force ceilings. Shrink the target
   geometry and the scale-blind pulls win the opening race.
4. Train-log term values certify nothing (normalization holds them near 1 by
   construction). Decide on gradient forces (force/denom records), the
   per-dim variance spectrum, zbar distances, and the held-out eval curves.
   r1's collapse was invisible in values and obvious in forces; r9's stall
   was invisible in the train log entirely and obvious only in eval.
5. Capacity follows demand. r10 added +528k params of per-ticker ResidualMLP
   blocks at the two stage seams (temporal->context entry, pre-projection;
   one variable, same train_seed): identical climb to ~10.5k steps, then
   holdout margins bled (strat 1.44 -> 1.1) while every train-side indicator
   kept improving and the blocks' functional engagement kept growing; the
   no-blocks run held 0.8/~1.5 with no slide. Ablating the trained blocks at
   inference cost <=9% margin — epiphenomenal to the deployed geometry. With
   the hinges slack and synergy saturated, the loss had no unmet demand for
   extra function class, so the capacity was spent on what the loss prices
   but the goal cannot see: grouping-noise polish and train-ticker
   memorization (invisible in train-log values — point 4, third
   confirmation). Raise demand first (separation weights, ticker count);
   function class last.

Run ledger: r1 total collapse (per-term EMA normalization made collapse a
stable attractor -> normalization split + kappa_floor); r2 anchor l1 ratchet
(-> fixed denom sqrt(v0)); r3 fixed-tiling memorization + dimensional
concentration (-> window_offset, lambda_util 1->3); r6->r7 kappa_floor
0.01->0.05 (contraction/expansion rebalance; best run: holdout retrieval 0.8,
newest margin ~1.5); r8 v0 1.0->0.4 dimensional collapse (reverted); r9
clip_norm 1->25 premature stall (reverted); r10 seam-capacity ResidualMLPs
(+528k params) memorization slide, epiphenomenal at inference (reverted;
code in commit d34dcc4 -> doctrine point 5). This file's defaults = r7's
recipe.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# exchange -> market for cross-market mode. Every ticker maps to exactly one
# exchange in the shipped parquet (verified: 0 of 13,026 span two), so this
# also assigns each ticker to exactly one market calendar.
DEFAULT_EXCHANGE_MARKET_MAP = {
    "NASDAQ": "US", "NYSE": "US", "AMEX": "US", "BATS": "US",
    "NYSE MKT": "US", "NYSE NAT": "US", "NYSE ARCA": "US",
    "SHE": "CN", "SHG": "CN", "SHGB": "CN", "SHEB": "CN",
}

# Config fields an operator may change when resuming (--resume): host placement,
# cadences, output location, the run horizon, and data *paths* (data content
# identity is enforced separately against the checkpoint's data_meta). Every
# other field is run-permanent — data layout, sampler structure, architecture,
# loss geometry, schedule shape, eval protocol/selector — and train.py refuses
# to resume if any of them differs: splicing two configs into one curve is the
# confounding doctrine point 2 forbids, and the restored best-score is only
# comparable while best_metric and the eval protocol stay fixed. max_steps is a
# knob so finished runs can be extended; note that extending stretches the
# cosine LR horizon going forward. context_checkpoint/loss_chunk are exact
# recomputation (values unchanged); devices is placement — toggling it forks the
# per-device dropout realization, the same documented class as moving the
# existing `device` knob between hosts (paired runs hold it fixed).
KNOB_FIELDS = frozenset({
    "run_dir", "device", "num_threads", "grad_checkpoint", "grad_diag_every",
    "eval_every", "max_steps", "K_inf", "calibrate_init", "raw_dir", "parquet_dir",
    "context_checkpoint", "loss_chunk", "devices", "eval_parallel",
})


@dataclass
class Config:
    # --- paths ---
    raw_dir: str = "TrainingData/indicators_data/raw"   # raw OHLCV (the processed CSVs drop these columns)
    run_dir: str = "StockIdentityModel/runs/default"

    # --- data layer ---
    N: int = 64                  # window length, trading days
    M: int = 4                   # number of strata
    W: int = 2                   # windows drawn per stratum per step
    Y: int = 8                   # minimum group size
    calendar: str = "benchmark"  # "benchmark" = SPY trading days; "union" = union of ticker dates
    q_clip: float = 0.999        # log-return clipping quantile (global, symmetric, frozen)
    window_offset: bool = True   # training only: shift the whole tiling by one random offset
                                 # delta in [0, N) per sampler epoch — same 104 fixed inputs
                                 # repeated ~1500x fed the r3 memorization; eval/export keep
                                 # the fixed (delta=0) tiling
    data_format: str = "csv"     # "csv" = per-ticker {TICKER}_daily.csv under raw_dir/stocksData;
                                 # "parquet" = long-format shards (ticker,exchange,date,OHLCV)
    parquet_dir: str = "TrainingData/ohlcv_parts"   # dir of *.parquet shards (data_format=parquet)
    exchanges: tuple[str, ...] | None = (           # parquet, cross_market=False only: keep these
        "NASDAQ", "NYSE", "AMEX", "BATS", "NYSE MKT", "NYSE NAT", "NYSE ARCA",
    )                            # exchanges on the single (US) grid; None = all. With
                                 # cross_market=True this is ignored — the exchange_market_map
                                 # decides membership and each market gets its own grid.
    min_history_days: int | None = None  # drop tickers with fewer than this many usable grid-days
                                 # (counted on the ticker's own market grid). None -> backend
                                 # default: 252 (~1 trading year) for parquet, 0 (filter off) for
                                 # csv — a default csv run keeps the historical unfiltered
                                 # universe. An explicit value applies to either backend; 0 disables
    cross_market: bool = False   # include non-US markets on their own calendars (parquet only).
                                 # Per-market grids + window tiling; Chinese windows are derived
                                 # from each drawn US window by the per-day dominance rule (CN day
                                 # k never after US day k) and co-reside in the same training
                                 # step; attention groups stay single-market, so only the
                                 # population terms (xsep/util/syn pooling) span markets
    exchange_market_map: dict[str, str] = field(   # exchange -> market (cross_market only)
        default_factory=lambda: dict(DEFAULT_EXCHANGE_MARKET_MAP)
    )
    min_calendar_quorum: int = 5 # quorum-union grids (cross_market secondaries): absolute floor —
                                 # a date is a grid candidate only if >= this many tickers have a
                                 # raw bar (calendar_validity="rows") or a valid TRADED bar
                                 # ("traded"). With calendar_frac > 0 the fractional leg is the
                                 # operative guard; any floor in [5, ~400] is equivalent there
    calendar_validity: str = "rows"  # quorum-union grids only. "rows" = historical: candidate
                                 # dates counted by RAW rows — this admits vendor placeholder
                                 # days (e.g. CN holiday bars carrying V=0 forward-fills), each
                                 # of which zeroes window completeness for every ticker.
                                 # "traded" = count only valid traded bars (finite, prices > 0,
                                 # V > 0): the feed's own encoding of "market open"
    calendar_frac: float = 0.0   # second quorum leg: keep a candidate date only if its count
                                 # >= this fraction of the centered rolling max over
                                 # calendar_frac_window candidates. Era-robust closure/partial
                                 # detector (drops holiday placeholder days, vendor partial
                                 # deliveries, market-wide V=0 glitch days). 0.0 = off
                                 # (historical). Recommended 0.5 with calendar_validity="traded"
    calendar_frac_window: int = 121  # rolling-max window for calendar_frac, in candidate dates
    halt_markets: tuple[str, ...] = ()  # markets ingested halt-tolerantly. The feed encodes
                                 # closures/halts as carried bars (V=0, prices = last close;
                                 # measured 99.65-99.98% bit-exact forward fills) before 2024
                                 # and as MISSING ROWS after: with a market listed here, carried
                                 # bars are admitted as halted days and missing rows strictly
                                 # inside a ticker's listing span are synthesized to the same
                                 # convention within max_ffill_days of its last traded bar.
                                 # A window is complete when every day is admitted (traded or
                                 # halted-carried) and >= halt_minfrac of days traded. () =
                                 # strict trading everywhere (historical)
    halt_minfrac: float = 0.9    # minimum traded fraction per complete window in halt markets
                                 # (ceil(0.9*64) = 58 traded days): admits the dominant 1-6 day
                                 # holiday/compliance halts, keeps genuinely suspended cells out
    max_ffill_days: int = 14     # gap-synthesis cap in CALENDAR days from the ticker's last
                                 # TRADED bar — the anchor is synthesis-invariant, making the
                                 # fill idempotent across cache reloads; long holes (B-share
                                 # coverage gap, delisting reviews) stay excluded rather than
                                 # blessed with stale carried prices
    holdout_pool_windows: int | None = None  # None = holdout pool requires completeness in EVERY
                                 # window (historical rule, all markets). int K (>= 2) =
                                 # SECONDARY markets draw their holdout from tickers complete in
                                 # the newest min(K, F) windows — the block the acceptance
                                 # metrics actually read; the PRIMARY market always keeps the
                                 # all-windows rule, so the US draw at rng(holdout_seed) stays
                                 # seed-identical in every mode. Recommended K = eval_windows

    # --- model ---
    D: int = 32                  # embedding dimension
    d_model: int = 128
    n_heads: int = 4
    temporal_layers: int = 2
    L_ctx: int = 2               # context module depth
    d_ff: int = 512              # feed-forward width (4*d_model)
    p_attn: float = 0.15         # observer dropout rate
    p_ff: float = 0.10           # feed-forward dropout rate
    g_exact: int = 64            # exact peer view up to this group size; beyond it, the shared-context approximation

    # --- loss ---
    alpha_prox: float = 1.0      # proximity boost
    tau_prox: float | None = None  # None -> (#usable windows)/10
    c_g: float = 16.0            # context half-trust group size
    m_sep: float | None = None   # None -> sqrt(D*v0)/2 (geometry unit tracks the variance floor)
    v0: float = 1.0              # per-dimension variance floor — NOT a free unit: it sets the
                                 # fences' absolute holding line against the scale-blind pulls
                                 # (anc is l1 — constant grip at any distance; syn is a ratio —
                                 # blind to size; neither weakens when the space shrinks). r8
                                 # lowered it to 0.4 (the "honest" realized level) and the
                                 # opening race went to contraction: per-dim variance crashed
                                 # 100x under the floor by ~step 3k, 28/32 dims dead, dz < m_sep,
                                 # margins eroding all late run. The r7 standoff (vmed ~0.2-0.5
                                 # vs v0=1 — util pushing forever, never winning) IS the
                                 # inflation mechanism: the backstop's job is to push, not to win
    lambda_cov: float = 1.0      # decorrelation weight inside L_util
    eta0: float = 0.05           # anchor base gain
    beta: float = 0.99           # EMA decay (loss normalizers and tau_gain)
    eps: float = 1e-6
    kappa_floor: float = 0.05    # EMA-normalizer denominator floor, as a fraction of the term's
                                 # first-step value: caps a shrinking term's gradient
                                 # self-amplification at 1/kappa_floor (collapse guard, run r1).
                                 # Raised 0.01 -> 0.05 after r6: contraction forces (sc/tc) ran
                                 # 1.5-2x the expansion forces — a higher floor engages 5x sooner
                                 # and caps the amplification at 20x instead of 100x. In practice
                                 # the raws fall to <1% of first-step within ~500 steps, so this
                                 # floor IS the sc/tc denominator for the whole run (r7 logs)
    lambda_full: float = 1.0
    lambda_self: float = 0.5
    lambda_peer: float = 0.5
    lambda_sc: float = 1.0
    lambda_tc: float = 1.0
    lambda_xsep: float = 1.0
    lambda_psep: float = 1.0
    lambda_anc: float = 1.0
    lambda_util: float = 3.0     # raised 1 -> 3 after r3: dimensional concentration (median dim
                                 # variance ~1e-4 vs v0=1, identity packed into a handful of dims)
                                 # while util's force was the one fighting and losing at 1.0
    lambda_syn: float = 0.3      # principal tuning knob: sets the masked-vs-full equilibrium

    # --- optimizer & schedule ---
    lr_peak: float = 3e-4
    lr_min: float = 1e-5
    warmup_steps: int = 1000     # deliberately overlaps the loss-normalizer burn-in
    max_steps: int = 20000       # cosine horizon; the operator usually picks an earlier checkpoint off the eval curves
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    clip_norm: float = 1.0       # NOT a transient-spike guard: typical grad norms run 5-20, so
                                 # this clips 100% of steps — i.e. always-on gradient
                                 # normalization, and it is load-bearing: every step enters Adam
                                 # at the same magnitude, the effective step follows the lr
                                 # schedule alone, and spiky window draws can't outvote quiet
                                 # ones. One-variable test (r7 clip=1 vs r9 clip=25, same seed):
                                 # r9 ahead early, then stalled 15k steps at retrieval 0.5 /
                                 # margin ~1.1 (native gn decay 20->7.5 = double annealing);
                                 # r7 ground steadily to 0.8 / ~1.5. Do not "fix" this again.

    # --- holdout ---
    holdout_frac: float = 0.05
    holdout_floor: int = 10      # used when pool < 100
    holdout_seed: int = 7

    # --- training / eval ---
    train_seed: int = 0
    eval_every: int = 250        # eval cadence, in steps; checkpoints written at the same cadence
    eval_windows: int = 32       # windows used by the held-out consistency/retrieval protocol
    eval_redraws: int = 5        # partition-redraw agreement samples
    best_metric: str = "margin"  # best.pt selection: "retrieval" (max holdout retrieval_acc),
                                 # "consistency" (min stratified holdout consistency median;
                                 # scale-dependent — cross-check the retrieval column), or
                                 # "margin" (max stratified median impostor/own distance ratio —
                                 # continuous, scale-free form of retrieval; > 1 iff top-1 hit)
    strat_eval_windows: int = 48 # size of the stratified window set feeding BOTH stratified
                                 # selection modes ("consistency" and "margin"): spread over the
                                 # M strata the way training sampling spreads its draws.
                                 # `eval_windows` (newest-16 block) is a separate, always-on set.
    K_inf: int = 4               # windows averaged at inference
    calibrate_init: bool = True  # fresh runs: rescale the final projection so per-dim var(z-bar)
                                 # starts at v0 — hinges begin satisfied (fences, not springs)
    grad_diag_every: int = 250   # cadence (steps) of per-term gradient-force diagnostics
                                 # in train_log; 0 disables
    grad_checkpoint: bool = True # recompute temporal encoder in backward (CPU RAM)
    device: str = "auto"         # "auto" -> cuda if available, else cpu; or "cpu" / "cuda" / "cuda:N"
    num_threads: int = 4

    # --- memory / parallelism (independent, default-off knobs; defaults = historical path) ---
    context_checkpoint: bool = False  # recompute each (window, scale) three-view context pass during
                                 # backward instead of holding its activations across the step: only
                                 # the [rows, D] embeddings and the explicit dropout masks persist.
                                 # The masks enter the checkpoint as arguments, so segments contain
                                 # no RNG — recompute is deterministic and gradients are identical
                                 # to the unflagged step (a step's saved-for-backward set drops
                                 # ~170 GB -> ~10 GB at parquet scale for ~+30% step time)
    loss_chunk: int = 0          # > 0: compute the O(n^2) separation terms in checkpointed pieces —
                                 # xsep in row-chunks of this size (1024 keeps recompute transients
                                 # ~2 GB at parquet scale; the monolithic [n_seg, n_seg] matrix is
                                 # 26-58 GB there), psep per (slot, scale) slice. Same sums; only
                                 # float reduction order differs. 0 = monolithic (historical)
    devices: str | None = None   # comma-separated devices for single-RUN job parallelism, e.g.
                                 # "cuda:0,cuda:1": weights are mirrored to each device per step,
                                 # (window, market) jobs are placed by a fixed slot rule (a US
                                 # window and its derived secondary partner land on different
                                 # devices), every job's embeddings are gathered to the first
                                 # device mid-graph and the ONE global loss is computed there —
                                 # never sharded — then mirror grads are summed into the master
                                 # BEFORE the global clip. None = single device (cfg.device).
                                 # Per-device dropout RNG makes a multi-device run a different
                                 # draw realization than a single-device run of the same seed
                                 # (numpy streams — windows/partitions/offsets/holdout — are
                                 # unchanged); paired runs must hold this fixed
    eval_parallel: bool = False  # split eval windows across `devices` replicas (eval is no_grad +
                                 # deterministic and per-window independent -> byte-identical
                                 # metrics when the devices are the same type/model; mixed types,
                                 # e.g. cuda+cpu, differ at float level — TF32 on the CUDA side).
                                 # No-op with a single device

    def resolved_m_sep(self) -> float:
        return self.m_sep if self.m_sep is not None else math.sqrt(self.D * self.v0) / 2.0

    def resolved_min_history_days(self) -> int:
        if self.min_history_days is not None:
            return self.min_history_days
        return 252 if self.data_format == "parquet" else 0

    def resolved_devices(self) -> list[str]:
        if self.devices:
            return [d.strip() for d in self.devices.split(",") if d.strip()]
        return [self.resolved_device()]

    def resolved_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        d = json.loads(Path(path).read_text())
        if "cons_eval_windows" in d and "strat_eval_windows" not in d:  # pre-rename configs
            d["strat_eval_windows"] = d.pop("cons_eval_windows")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
