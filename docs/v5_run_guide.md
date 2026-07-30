# v5 run guide

How to execute a complete v5 run on the US universe, end to end, on the corrected pipeline (post `docs/v5_fix_implementation_plan.md`). The universe is US-only by construction: the loader filters the parquet shards to `US_EXCHANGES` and there is no non-US path.

Machine: the real runs need the multi-GPU box (4 GPUs assumed below; the scripts adapt to whatever `nvidia-smi` reports). CPU is only for `--smoke` and the unit tests.

## 0. Prerequisites and feed prep

Data on disk: `TrainingData/ohlcv_parts/*.parquet` (Tiingo long-format shards) and `TrainingData/indicators_data/raw/fear_greed.csv`.

Refresh `fear_greed.csv` through the OHLCV data end before any real run. The frame build prints a `FROZEN-FG SPAN` warning when it lags; a frozen fg degrades the `fg_corr` channel progressively. There is no automated fetch yet (P5.3 is specification-only), so refresh manually.

Regenerate the session census in the same operation as ANY feed append, then commit it:

    python3 tools/build_session_census.py

This writes `TrainingData/session_census.csv` (one row per feed date with its distinct US-name count) and prints its sha256 plus per-year universe death counts. The census drives the phantom-session filter at load (sessions with < 3 names are dropped: pre-1976 holidays, weekend rows, singleton prints), the rolling-origin seams, and the run manifest fingerprint. A stale census against a fresh feed mis-handles the new dates.

Disk: each trainer process writes a ~2.1 GB feature memmap into its own `cache/v5_gpu{k}`; the build refuses to start without 1.2x the memmap size free.

Sanity before anything: `python3 tests/run_tests.py` (dependency-free, CPU) and a smoke pass `python3 run_train_v5.py --smoke --run-dir /tmp/v5_smoke --cache-dir /tmp/v5_smoke_cache`. Never run `--smoke` with the default `--run-dir`: it would overwrite `models/v5/member_*.pt`.

## 1. One-time protocol steps (before the first corrected run)

Step zero: score the existing run-2 checkpoints on the OLD label semantics, before the P2 label changes apply. Valid only from a tree where `features_v5.py` is still the run-2 builder (last touched by commit `962bc0b`) while the new `tools/`, `v5/metrics.py`, and `TrainingData/session_census.csv` are present -- e.g. check out the pre-fix commit and copy those files in from the fix commit (`git checkout <fix-commit> -- tools v5/metrics.py TrainingData/session_census.csv` on top of a `962bc0b`-era `features_v5.py`). The scorer detects which builder generation it runs against. On the GPU box in that state:

    python3 tools/score_checkpoints.py --run-dir models/v5 --members 0,1,2,3 --ensemble --population val

It hard-asserts the 23,948,324 scaler-row fingerprint, records the git revision and the `features_v5.py` sha256 (what makes a too-late step zero detectable), runs untempered (run-2's finalize crashed before writing `v5_meta.json`), and writes `docs/v5_step0_results.md` with the ensemble-vs-baseline CE, the measured per-date IC dispersions (unfiltered + tradability-filtered), comparator ICs, and (GPU) the MambaRef<->CUDA parity number.

Pre-registration: freeze `docs/v5_preregistration.md` BEFORE the primary run, after step zero fills the numbers. Contents per the plan's P6.1: census sha256, block rule (4 contiguous ~63-session val blocks, roles [stop, gate, stop, gate], block 1 = stopping), the rolling-origin data-ends from `--print-seams`, the P6.3 gate definitions verbatim, step-zero dispersions and the derived detectable-IC figures, the tradability-filtered confirmation threshold, the stopping rule, the persistence rule, the 1d cost-gate operationalization, and the score-floor procedure.

    python3 tools/score_checkpoints.py --print-seams

prints the rolling-origin `--data-end` values (phantom-filtered sessions exactly k*254 before the primary data end; currently k=1: 2025-06-03, k=2: 2024-05-28, k=3: 2023-05-23).

## 2. Null-control run (gates the metric machinery, not the model)

One member trained on within-date-shuffled train labels; its finalize is kept because the gate read needs the null run's temperatures. Dedicated run dir is mandatory (default would overwrite `models/v5`):

    python3 run_train_v5.py --members 1 --label-shuffle-within-date --max-steps 40000 --run-dir models/v5_nullc --cache-dir cache/v5_nullc
    python3 tools/score_checkpoints.py --run-dir models/v5_nullc --members 0 --ensemble --population gating

A single-process 1-member run trains and finalizes in one invocation (the member range covers the whole ensemble), so the null run's `v5_meta.json` temperatures exist for the gate read without a separate `--forecast-only` pass.

PASS iff the gating-block ICs are ~0 with |t| < 2 across horizons and scores; a modest centered-CE beat is expected, not a failure. Nulls (a) shuffle and (b) shift run inside the metrics tests and can be applied to any scorer read via `--null shuffle|shift`. All three must pass before any 3-sigma verdict is read.

## 3. Primary run

    bash run_multi_gpu.sh 1 --eval-mode both --ticker-holdout-frac 0.1

One member per GPU (4 total), batch 256, lr 3e-4, warmup 1000, patience 20 evals, `--val-subsample 150000`, scheduled eval cadence (every 250 steps to 2k, 1000 to 30k, 5000 after). The script exports one `RUN_NONCE` per launch; the member-0 process writes `models/v5/run_manifest.json` (args, git revision, data fingerprint, scaler rows) and the other trainers refuse to run against a stale or mismatched manifest. After all members finish it deletes the duplicate per-GPU caches, reuses `cache/v5_gpu0`, and finalizes; the finalize refuses if the data, git revision, or any split-relevant arg changed since training.

The finalize fits per-horizon temperatures on the tag-0 {1d, 1w} stopping-side labels (1m/6m always persist T = 1.0), computes the per-horizon live score floors (p95 of the tempered ensemble Score over tradability-filtered tag-0 stopping-block-anchor samples), runs the parity check, and writes: `models/v5/{v5_meta.json, v5_norm.json, config.json, member_*.pt, eval_history_member*.jsonl}` and `forecasts/{*_forecast.csv, split_info.json, oos_start_date.txt, surface_manifest.json}`.

## 4. Rolling-origin runs (unconditional -- they are the adjudicating dataset)

Same args plus a data-end and a dedicated run dir per k; run regardless of the primary outcome:

    bash run_multi_gpu.sh 1 --eval-mode both --ticker-holdout-frac 0.1 --data-end 2025-06-03 --run-dir models/v5_ro1
    bash run_multi_gpu.sh 1 --eval-mode both --ticker-holdout-frac 0.1 --data-end 2024-05-28 --run-dir models/v5_ro2
    bash run_multi_gpu.sh 1 --eval-mode both --ticker-holdout-frac 0.1 --data-end 2023-05-23 --run-dir models/v5_ro3

`--data-end` truncates every ticker's OHLCV BEFORE the feature build, so labels, sigma, and features are computed as if the feed ended then. Distinct `--run-dir`s are mandatory: each run's scaler differs, and artifacts would clobber each other. Non-default run dirs get their own forecast dirs (`forecasts/<run-dir-basename>/`).

## 5. Gate reads and design freeze

Per run, score the finalize ensemble (the gating object; tempered) on the gating blocks and append the results to `docs/v5_gate_ledger.md`:

    python3 tools/score_checkpoints.py --run-dir models/v5 --ensemble --population gating --out models/v5/gate_read.json

Repeat with `--run-dir models/v5_ro{1,2,3}`. The output carries the tradability-filtered ICs, the top-of-ranking confirmation (including the return-units variant the 1d cost gate consumes), and the uncensored-outcome sensitivity. Decision gates are P6.3 of the plan: 1w/1m carry the standard 3-sigma gates, 1d is cost-conditioned only, 6m is report-only, and any qualifying result must persist at the +-2-eval neighbors of the selected checkpoint (read from the eval history, zero extra gating looks).

Design freeze, once per freeze, after the gates:

    python3 run_train_v5.py <primary args> --forecast-only --design-freeze
    python3 tools/score_checkpoints.py --run-dir models/v5 --ensemble --population oos --design-freeze

The OOS population refuses to run without `--design-freeze` and prints a one-shot warning; the pooled success verdict is additionally reported on the three rolling years alone (the primary val year is design-contaminated).

## 6. Consumers

Backtest (descriptive only -- the gates adjudicate, a single ~254-session single-position path cannot):

    python3 run_backtest_v5.py

Reads `forecasts/`, the score floors from `models/v5/v5_meta.json`, reports the {0, 10, 25} bp cost grid x the delisting-haircut sensitivity {CLI, 0.3}, replays 1000 protocol-identical nulls, and writes `videos/backtest_metrics_v5.json` + `trade_summary_v5.csv`. 6m is excluded from candidacy by default.

Live signals:

    python3 run_live_signals_v5.py --as-of-date YYYY-MM-DD

Defaults to the full trained parquet universe (`--stocklist` restores the legacy 350-name list), ranks and gates on Score against the stored floors, requires tradability for buy candidacy, and hard-fails when the feed's freshest session lags the decision date by more than 63 sessions. Until the P5.3 refresh contract is implemented, `--force-refresh` is a stub and the hard-fail is the guard.

## What changed since initial v5

Labels (`features_v5.py`): labels re-anchored to next-close entry -- z spans (t+1, t+1+d) so every horizon matches an executable fill; the label mask adds finiteness and per-horizon calendar-gap ceilings ({6, 12, 36, 193} days), so a "21-row" label across a halt is no longer a 1-month return; the EWMA re-seeds after > 90-day gaps instead of freezing pre-halt vol; frames carry `volume` / `tradable` (close >= $5 and 63-session median dollar volume >= $1M) / `vol_med63`; phantom sessions (census names < 3) are dropped at load.

Targets: labels are cross-sectionally centered per (session, horizon) -- the trained target is the market-relative z' = z - median, so P(up) now means "beats the date median by theta sigma-days", the per-date market component (~15-21% of label variance) can no longer be memorized from date fingerprints, and absolute-probability thresholds are meaningless (hence Score floors, below).

Trainer (`run_train_v5.py`): 1m/6m train labels are thinned (strides 4/25) to slow the long-horizon memorization clock; early stopping now runs on the smoothed rank-IC stopping score (t_1d + t_1w)/sqrt(2) over stopping blocks of the val year, with a calibrated-CE bootstrap while the score is degenerate; evals follow a step schedule dense inside warmup; every eval appends the full metric suite (both scores, block roles, tradability and uncensored variants, train-panel ICs) to a per-member history file, with gating-side values never printed or consulted; the judged CE, its baseline, and the temperature fit all live on one row set (tag-0 {1d, 1w} stopping-side labels under interval membership); `eval_mode=both` now routes eval tickers' val rows into the val fold (tag 1) and non-eval OOS rows onto the forecast surface (full universe); the class-rate probe measures the actually-trained population; sub-$1 anchors never train; `lam_cls` lives on the config; per-horizon train-CE EMAs print in the step log.

Provenance: every run writes `run_manifest.json` (args, git revision, data fingerprint incl. census sha256, scaler rows); multi-GPU launches are nonce-checked; `--forecast-only` refuses on any data/revision/arg drift. `--run-dir` isolates concurrent undertakings; `--data-start/--data-end` enable rolling origins and the survivorship ablation; `--label-shuffle-within-date` is the end-to-end leakage null; a disk preflight replaces the mid-write Bus error.

Finalize: temperatures are fit from an fp16 logit cache in 100k-row chunks (the float32 materialization killed the run-2 finalize); 1m/6m always persist T = 1.0; per-horizon Score floors are computed inside every finalize and shipped in `v5_meta.json`; the parity number is persisted; forecast CSVs gain `Volume`/`Tradable`/`VolMed63` and include every eval anchor (labels are not needed to forecast); `surface_manifest.json` tags holdout vs time-only tickers.

Consumers: `run_backtest_v5.py` replaces the v4 accounting for the v5 line -- gap-aware sign-locked split detection with volume corroboration and exact back-adjustment (no whole-ticker rejection, no directional look-ahead), t+1 fills, per-ticker exit calendars, forced delisting exits with a haircut sensitivity, a cost grid, and 1000 protocol-identical nulls (v4 had 50 always-invested ones). `run_live_signals_v5.py` ranks and gates on Score with per-horizon floors instead of an absolute P(up) threshold, requires tradability for candidacy, excludes 6m from the argmax by default, scores the full parquet universe, and hard-fails on a stale feed.

New modules: `v5/metrics.py` (per-date Spearman IC, block-clustered HAC t, top-of-ranking with the return-units variant, stopping score, uncertainty quality, evaluation nulls), `tools/score_checkpoints.py` (step zero / gate reads / one-shot OOS, manifest-driven arg reconstruction, `--print-seams`), `tools/build_session_census.py`. Tests: 87 across metrics, features, trainer, backtest, live.
