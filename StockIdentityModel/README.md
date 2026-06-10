# Stock Identity Encoder

Learns, from raw daily OHLCV candles alone, a 32-dimensional **identity embedding** per
stock — a vector capturing a quality of the stock that is stable in time and distinguishes
it from other stocks:

- **Persistent** — the same ticker gets (nearly) the same vector regardless of which time
  window, peer group, group size, or dropout draw produced it.
- **Distinctive** — different tickers get vectors far apart.
- **Two-route** — the vector is derivable from the ticker's own price behavior alone
  (self view) and from its relations to other tickers alone (peer view), and the combined
  computation (full view) is trained to beat both routes.
- **Inductive** — no per-ticker parameters: any ticker with enough candle history plus a
  set of context tickers can be embedded at inference, including tickers never seen in
  training.

One pass, end to end: candles are normalized per ticker per window (price level and volume
scale removed, shape of moves kept); a 2-layer transformer summarizes each ticker's window
into one vector; attention blocks over randomly drawn peer groups — at several group sizes,
under the three views — produce the embedding; the loss demands consistency across
windows/scales/groupings, margin separation between tickers, full-view synergy over the
masked views, anchor stability across training, and per-dimension utilization
(anti-collapse). Ticker identity and calendar position never enter the model, which is
what makes it inductive by construction.

Standalone PyTorch system; no touchpoint with the v4 forecasting pipeline. Input = raw
OHLCV candles from `TrainingData/indicators_data/raw/` (the processed CSVs drop the raw
columns this model needs).

## Environment

The v4 pipeline pins TF 2.10; this model needs its own environment:

```bash
python3 -m venv ~/venvs/stockid
~/venvs/stockid/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
~/venvs/stockid/bin/pip install pandas
```

## Commands (from the repo root)

```bash
# Train (writes runs/<name>/{config.json, data_meta.json, calibration.json, train_log.jsonl, eval_log.jsonl, latest.pt, best.pt})
python -m StockIdentityModel.train --run-dir StockIdentityModel/runs/r1

# Export the artifact from a chosen checkpoint (picked off the eval curves)
python -m StockIdentityModel.export --checkpoint StockIdentityModel/runs/r1/best.pt \
    --out StockIdentityModel/artifacts/v1

# Embed any ticker (including ones never seen in training)
python -m StockIdentityModel.inference --artifact StockIdentityModel/artifacts/v1 --ticker AMD
```

## GPU

Device is `auto` (CUDA when available); override with `--device cpu|cuda|cuda:N` on every CLI.
On a GPU box install the CUDA build (`pip install torch` — the default Linux wheel ships CUDA)
instead of the `+cpu` wheel above. Recommended on a 24 GB card:

```bash
python -m StockIdentityModel.train --run-dir StockIdentityModel/runs/r1 --no-grad-checkpoint
```

(`grad_checkpoint` exists to fit the temporal-encoder activations in memory; disabling it is
faster only at csv scale — at parquet scale keep it on.) TF32 matmuls are enabled automatically
on CUDA.

The model is ~1M parameters; what doesn't fit at parquet scale is a *step's activations* —
~170 GB saved-for-backward across the 16 coexisting (window, market) job graphs, plus O(n²)
loss-stage matrices (xsep alone is a 26–58 GB allocation). Two exact-recompute knobs make a
step fit one 24 GB card (~10–13 GB peak, ~+30% step time):

```bash
python -m StockIdentityModel.train --run-dir ... --data-format parquet --cross-market \
    --context-checkpoint --loss-chunk 1024
```

`--context-checkpoint` recomputes each (window, scale) three-view context pass during backward;
only the [rows, D] embeddings and the explicit dropout masks persist across the step. The masks
enter the checkpoint as arguments, so segments contain no RNG — gradients *and* the draw stream
are identical to the unflagged step. `--loss-chunk N` computes the O(n²) separation terms
(xsep/psep) in checkpointed chunks — same sums, float reduction order aside. Both are
recomputation, not approximation. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
against allocator fragmentation at this scale.

Multi-GPU of a single run is supported as *job placement under the one global loss* —
independent of the checkpointing knobs and composable with them:

```bash
python -m StockIdentityModel.train ... --devices cuda:0,cuda:1 --eval-parallel
```

`--devices` mirrors the weights to each listed device per step, forwards each (window, market)
job on a device chosen by a fixed slot rule (a US window and its derived secondary partner land
on different devices), gathers every job's embeddings to the first device mid-graph, computes
the single population-coupled loss there exactly as in single-device mode, and runs one
backward across all devices; mirror gradients are summed into the master *before* the global
`clip_norm` clip. The loss is never sharded — gradients equal the single-device step up to
float reassociation (~1.7–1.9× throughput). Caveat: per-device dropout RNG makes a multi-GPU
run a different draw realization than a single-GPU run of the same seed (numpy draws —
windows, partitions, offsets, holdout — stay identical); paired runs must hold `--devices`
fixed. `--eval-parallel` window-splits the deterministic no-grad eval across the same devices —
byte-identical metrics on identical device types (mixed types differ at float level), ~2× faster
evals. A second GPU can still be spent on a parallel run instead (`CUDA_VISIBLE_DEVICES=1`) when
comparing recipes.

Checkpoints and artifacts are machine-portable (loaded via `map_location`, artifact weights
saved on CPU): train on GPU, export/infer anywhere. Caveat: bit-exact `--resume` replay is a
CPU property; on CUDA, scatter/`index_add` use non-deterministic atomics, so resumed curves can
diverge at floating-point noise level (distributional state — anchors, EMA normalizers, queues —
is restored exactly).

`--resume` validates run identity before touching anything: every run-permanent config field
(data layout, sampler structure, architecture, loss geometry, schedule shape, eval
protocol/selector) must match the checkpoint's, and the rebuilt dataset must match the
checkpoint's `data_meta` (catches raw files changing underneath a run). Only operational knobs
(`config.KNOB_FIELDS`: device, threads, cadences, `max_steps`, `run_dir`, data paths,
`grad_checkpoint`, …) may differ — resume with the run's own `--config <run_dir>/config.json`
and change knobs from there. The running best selection score rides in every checkpoint, so a
resumed run continues the `best.pt` comparison where it left off instead of overwriting
`best.pt` at its first eval (pre-fix behavior); legacy checkpoints without the field fall back
to the old restart-at−∞ behavior.

`best.pt` tracks the highest held-out retrieval accuracy — the acceptance metric: tickers held
out of training entirely must stay consistent across windows and find themselves by nearest
neighbor against the trained gallery. `latest.pt` is written at every eval. Both store full
training state and are valid `--resume` targets.

Selection is switchable (`--best-metric`), the optional modes computed over `strat_eval_windows`
(32) windows spread across the sampler's M strata — the full timeline, the way training
samples — instead of the newest 16 (`eval_windows`, the always-on acceptance block):

- `consistency`: lowest held-out consistency median. Smoother than retrieval's 10-trial
  granularity, but scale-dependent and collapse-blind.
- `margin`: highest median margin ratio ρ = d(nearest impostor)/d(own key) — the continuous,
  scale-free form of retrieval (ρ > 1 iff top-1 hit; near-misses and total misses separate;
  collapse reads as ρ ≈ 1, never a win). `margin_ratio` is logged at every eval regardless;
  stratified modes add `*_stratified` columns.

Whatever the selector, the retrieval column stays the acceptance read.

## File map

| File | Responsibility |
|---|---|
| `config.py` | hyperparameters, paths, optimizer knobs |
| `data.py` | trading-day grid, window tiling, candle normalization + clip thresholds, holdout split |
| `sampling.py` | stratified window sampling, seeded group partitions |
| `model.py` | temporal encoder, three-view context module |
| `losses.py` | loss terms, anchor buffers, EMA loss normalizers |
| `train.py` | training step, observer dropout, LR schedule, checkpoints |
| `evaluate.py` | held-out consistency + retrieval, partition-redraw agreement |
| `export.py` | artifact: weights + context recipe + canonical embedding table |
| `inference.py` | `embed()` for arbitrary tickers from an artifact |

## Data backends and cross-market mode

Two input backends (`data_format`): `"csv"` — the historical per-ticker
`{TICKER}_daily.csv` files; `"parquet"` — long-format shards
(`ticker, exchange, date, OHLCV`) under `parquet_dir`, filtered to
`exchanges` (default US). `min_history_days` drops tickers with too few usable
days on their own market grid; its default is backend-dependent — 252
(~1 trading year) for parquet, 0 (off) for csv — so a default csv run keeps
the historical unfiltered 347-ticker universe. An explicit value applies to
either backend.

**Cross-market mode** (`cross_market=True` / `--cross-market`, parquet only,
**off by default** — the committed defaults are byte-identical to the
single-market pipeline, verified by hash):

- **Per-market grids.** `exchange_market_map` assigns each exchange to a market
  (default: US exchanges → "US", Shanghai/Shenzhen → "CN"). Each market gets
  its own trading-day calendar (US = SPY benchmark; secondary = union of own
  tickers' dates, kept only where ≥ `min_calendar_quorum` tickers have a bar,
  so one rogue date can't hole everyone) and its own backward N-day tiling,
  completeness mask, feature tensor, pool, and holdout (the US holdout draw is
  seed-identical to single-market mode; secondary markets draw their own).
- **Dominance-paired windows.** Each drawn US window derives one window per
  secondary market: the latest own-calendar window whose day-k date never falls
  after the US window's day-k date, for every k (endpoint alignment is not
  enough — the calendars' day-counting drifts within a span). Derived windows
  co-reside in the same training step; windows that predate a market's history
  are skipped (logged per step as `windows_mkt`).
- **Groups never span markets**, so the three-view leak closure, contemporaneity,
  and the ω noise model see same-calendar peers only. Cross-market coupling
  happens solely through the population terms — xsep separates different
  tickers across the union, util's variance floor and the syn/EMA pooling see
  both markets — which is exactly the union-distinctiveness effect the mode
  exists for. The encoder remains market-blind (no market id, no dates).
- **Eval**: legacy (US) columns keep their exact single-market definitions;
  secondary markets add prefixed columns (`cn_consistency_*`, `cn_retrieval_acc`,
  `cn_margin_ratio` — scored against the union gallery — and
  `cn_partition_agreement`). The stratified selection metrics become
  union-protocol (queries from every market's holdout, union gallery); the
  US-protocol values move to `*_stratified_us`. `market_centroid_acc`/`_dist`
  log whether "market" is a separating axis (the instrument that decides
  whether any binding term is ever needed).
- **Artifacts** carry one context recipe per market; the embedding table spans
  all markets; `inference.py` resolves a ticker's market from the recipe (or
  `--market`) and embeds it against its own calendar's context windows. Parquet
  artifacts bulk-load context candles from the shards.
- Memory: both markets' panels and features stay resident (full US+CN ≈ a few
  GB); per-step compute roughly doubles (both markets encoded per step).

**Secondary-market data semantics** (measured on the shipped CN slice): the feed
encodes market closures and stock halts as *carried bars* — rows with
O=H=L=C = last traded close and V=0, 99.65–99.98% bit-exact forward fills —
through 2023, and as *missing rows* from 2024. Read naively ("a row must be a
traded bar"), every holiday placeholder day poisons the window containing it
(68 of 78 CN windows had zero complete tickers). Three run-permanent ingestion
knobs, all default-off (defaults are byte-identical to the historical reader),
interpret the encoding instead:

- `calendar_validity="traded"` + `calendar_frac=0.5` (+ `calendar_frac_window=121`):
  a quorum-grid date is a market-open day only if enough tickers actually
  *traded* (absolute floor `min_calendar_quorum` AND ≥ frac of the rolling-max
  market size) — drops holiday placeholder days, market-wide V=0 glitch days,
  and severe vendor partial deliveries, across both encoding eras. CN: 4,701
  grid days, 73/73 windows usable.
- `halt_markets=("CN",)` + `halt_minfrac=0.9` + `max_ffill_days=14`: carried
  bars are admitted as halted days, and missing rows inside a ticker's listing
  span are synthesized to the same convention within 14 calendar days of its
  last traded bar (long holes — the B-share coverage gap, delisting reviews —
  stay excluded). A window is complete when every day is admitted and ≥ 58 of
  64 days traded. Halted days carry exact-zero price returns and the volume
  sentinel 0.0 (volume median over traded days only; clip quantiles estimated
  on traded-day returns only). Suspension-prone tickers stay in the population
  instead of being censored — halt propensity becomes learnable identity
  signal. `normalize_window(halt_aware=True)` mirrors the identical rule at
  inference; old artifacts default to the historical path byte-for-byte.
- `holdout_pool_windows=32`: secondary markets draw their holdout from tickers
  complete in the newest 32 windows (no CN ticker spans 19 unbroken years; the
  newest block is what the acceptance metrics read) — CN holdout ≈ 112 under
  the full config. The primary market always keeps the all-windows rule, so
  the US draw stays seed-identical in every mode.

Recommended staging (one variable per run, same `train_seed`):

```bash
# r12 — calendar repair + holdout pool (strict trading): the CN eval baseline
python -m StockIdentityModel.train --run-dir StockIdentityModel/runs/r12 \
    --data-format parquet --cross-market \
    --calendar-validity traded --calendar-frac 0.5 --holdout-pool-windows 32 \
    --context-checkpoint --loss-chunk 1024 --devices cuda:0,cuda:1 --eval-parallel

# r13 — halt tolerance on top (the one-variable comparison against r12)
python -m StockIdentityModel.train --run-dir StockIdentityModel/runs/r13 \
    --data-format parquet --cross-market \
    --calendar-validity traded --calendar-frac 0.5 --holdout-pool-windows 32 \
    --halt-markets CN \
    --context-checkpoint --loss-chunk 1024 --devices cuda:0,cuda:1 --eval-parallel
```

All seven ingestion fields are run-permanent (resume refuses drift) and bump
the data cache to v5.

## Design notes

- **Collapse guards** (post-r1, which collapsed totally — every ticker at one point, retrieval
  0/10): per-term EMA normalization equalizes loss *values*, not *gradients* — a quadratic
  consistency term's normalized gradient grows as 1/√L as it shrinks while a saturated hinge's
  stays constant, making collapse a stable attractor. Guards: (1) the bounded hinge terms
  (xsep/psep/util) are normalized by fixed ceilings (m_sep², v0+λ_cov) instead of their own
  EMA, and the ℓ1 anchor by the geometry unit √v0 — an ℓ1 gradient's norm doesn't shrink
  with its value, so EMA-normalizing it is a contraction ratchet (run r2); (2) the quadratic
  terms' (sc/tc) and syn's EMA denominators are floored at `kappa_floor` (0.01) × their
  first-step value, capping gradient self-amplification at 100×; (3) fresh runs rescale the
  final projection at init so per-dim var(z̄) starts at v0 (`calibrate_init`) — hinges begin
  satisfied, as fences rather than springs.
- **Force diagnostics**: every `grad_diag_every` steps (default 250) the train log records
  per-term λ·‖∂L/∂z‖/denom (`force`) and the live normalizer denominators (`denom`); every
  step records mean/min pairwise distance of the z̄ population (`zbar_dist`). Collapse is
  read in gradient units, not term values.
- **Per-epoch tiling offsets** (`window_offset`, on by default; `--no-window-offset`): training
  windows come from the base tiling shifted back by one random δ ∈ [0, N) per sampler epoch
  (all stratum queues refill synchronously; one δ per full pass). Post-r3 rationale: the fixed
  tiling yields only ~104×#tickers distinct input tensors, each repeated ~1500×/run — the
  memorization engine behind the train/holdout divergence. Offsets cut exact repeats to ~1;
  every span is still untouched real data. Eval/export/inference always use the δ=0 tiling.
  The train log records the live `offset`; `lambda_util` default also raised 1 → 3 here
  (r3 dimensional concentration — watch `var_spectrum`).
- **Calendar** = SPY trading days (the benchmark grid is immune to rogue dates in any single
  ticker's file). `calendar="union"` switches to the union of all tickers' dates.
- **Context trust in temporal consistency**: the weight ω(g) = (g−1)/(g−1+c_g) discounts
  embeddings computed with few peers (a thin context is noisy sampling, not signal). Group
  sizes differ across windows at the same scale because the universe grows over time, so each
  same-ticker window pair is weighted by ω(min(g, g′)) — a pair is trusted no more than its
  thinner context, and the weight reduces to a single per-scale ω whenever sizes match.
- **FF dropout masks** are drawn once per (window, scale) per token and reused across the
  three views — like the observer-dropout masks — so view differences are attributable to
  withheld evidence, not noise. Fresh draws per (window, scale) keep the self view varying
  across scales only through dropout, which is what lets scale consistency on the self view
  penalize dropout sensitivity.
- **Pre-norm stacks** end with a final LayerNorm before each output head (standard for
  pre-norm). FF width `d_ff = 4·d_model = 512`, GELU.
- **Weight decay** excludes biases, LayerNorm parameters, and the final `d_model → D`
  projection (decay there pushes against the variance floor `v0` for no benefit); everything
  else — including the positional embedding — decays.
- **Eval-time retrieval gallery** during training uses trained tickers' window-set-B means as
  the distractor table (same protocol as the holdout keys). The exported acceptance numbers
  use the canonical table itself.
- **Clip-threshold pool**: quantiles are computed over the four price log-return features of
  training tickers only (holdout exclusion is total), pooled across all windows, then frozen
  into the artifact.
- If a step draws the same window twice (possible at a queue refill boundary), the two draws
  occupy distinct slots with fresh partitions; the proximity weight κ — which up-weights
  same-ticker pairs from windows close in time — sees a window distance of 0 for that pair.
- **Capacity lesson (r10, reverted — code in commit d34dcc4)**: per-ticker ResidualMLP
  stacks at the two stage seams (+528k params, one variable vs this recipe, same seed)
  matched it to ~10.5k steps, then bled held-out margins while every train-side signal
  kept improving; the no-blocks run held 0.8/~1.5 with no slide, and ablating the trained
  blocks at inference cost ≤9% margin. The loss equilibrium, not the function class, sets
  the ceiling here: capacity beyond the loss's demand is spent on memorization the
  objective cannot distinguish from identity. Architecture changes need a demand-side
  mechanism first — separation weights, or ticker count (the one scaling axis with the
  field's preconditions attached).

## Sizing on this repo's data

347 tickers on the 1999-11-01 → 2026-04-23 SPY grid → 104 windows of 64 days; survivor pool
43 → holdout floored at 10. Universe per window grows 39 → 295; ladders span {1,2,4} (early)
to {1…32} (late). A training step (M·W = 8 windows, 3 views, exact peer view at g ≤ 64) takes
~10–30 s on a 4-core CPU; `grad_checkpoint=True` keeps peak RAM in budget by recomputing the
temporal encoder during backward.
