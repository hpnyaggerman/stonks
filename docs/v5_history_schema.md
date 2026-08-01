# v5 run history parquet schema

Column-level reference for the four per-member history tables every v5 training run maintains under its run dir (`models/v5/runs/<name>/`), written by `run_train_v5.train_member`. Sibling one-shot artifacts in the same dir -- `run_manifest.json` (args, git revision, data fingerprint, environment, scaler rows), `config.json` (resolved `V5Config`), `v5_norm.json` (training scaler), `split_meta.json` (cut dates, embargo, holdout list, block dates and roles, resolved schedule, baselines, comparator ICs) -- are launch-written JSON and are described in `docs/v5_run_guide.md`; this document covers only the parquet tables.

## Write mechanics

Each table is rewritten in full at every tick via temp file + atomic rename (`<file>.tmp` then `os.replace`), so the on-disk file is a complete, readable parquet at every instant of a run: a killed process -- the normal terminal state under the stopping protocol -- loses at most the in-flight tick, and a reader (including on a live run) always sees a consistent snapshot at most one tick stale. Stray `.tmp` files are leftovers of a kill mid-rewrite and are deleted at the next member start; they are never readable parquet.

`evals`, `ic`, and `top` are rewritten at every eval (scheduled cadence: every 250 steps to 2k, 1000 to 30k, 5000 after); `train` is rewritten every `--log-every` steps. Compression is zstd; the string key columns of `ic`/`top` are dictionary-encoded and come back as pandas categories. Scalars are float64/int64/bool except the per-horizon train EMAs (float32); absent values are NaN/null.

Reading: `pd.read_parquet(path)`. The `ic`/`top` column named `mask` collides with `DataFrame.mask`, so filter with `df["mask"] == "primary"`, not attribute access. Members are separate processes with separate files; for cross-member analysis concatenate with an added member column.

Runs older than r2 carry the same eval/train content as line-JSON (`eval_history_member{i}.jsonl` with nested record lists, `train_history_member{i}.jsonl`); the parquet tables are their flattened successors.

## `evals_member{i}.parquet` -- one row per eval

| column | meaning |
|---|---|
| `step` | Optimizer step of the eval. |
| `smoothed_score` | Median of the last 3 `stopping_score` values; the improvement tracker runs on this. `-inf` while the score is degenerate (bootstrap phase). |
| `improved` | Ruling improvement verdict at this eval: `smoothed_score > best + 0.05` (during bootstrap: stopping-side calibrated CE record instead). Drives checkpoint selection, `member_{i}_best.pt`, and the patience-20 early stop. |
| `bootstrap` | True while the stopping score has never been finite; improvement is CE-driven in that phase. |
| `raw_ce_tag0` | Full-fold raw (T=1) CE on tag-0 val rows. Report only. |
| `raw_ce_tag1` | Same on holdout-ticker val rows (tag 1); null when the run has no ticker holdout. Report only. |
| `cal_ce_stop` | Judged calibrated CE on the stopping-side score half ({1d, 1w} interval-membership labels, capped and parity-split). The bootstrap criterion, the tie-breaker, and the LR-gate CE input. |
| `raw_ce_stop` | Same rows at T=1. |
| `judged_T_{1d,1w,1m,6m}` | Transient in-loop half-sample temperatures behind `cal_ce_stop` -- not the persisted finalize temperatures; 1m/6m always 1.0, all 1.0 under `--no-calibrate-eval`. |
| `n_stop_score_rows` | Rows in the judged score half. |
| `base_stop` | Stopping-side constant-marginal baseline CE on the same rows; constant per run. `cal_ce_stop < base_stop` is the BEAT verdict. |
| `t_1d_stop`, `t_1w_stop` | HAC t of the consumer-score (`score`), primary-mask, stop-role, all-names IC -- copies of the two `ic` rows the stopping score reads. |
| `n_1d_stop`, `n_1w_stop` | Qualifying stopping dates behind those two t values. |
| `stopping_score` | `(t_1d + t_1w) / sqrt(2)`; ordinal only (the two t's share dates and the 1d path nests in 1w). `-inf` when either horizon has fewer than 10 dates or a non-finite t. |
| `lr_scale` | LR multiplier in force after this eval's drop decision (x0.5 per drop, floor 0.04). |
| `train_ema` | Train-loss EMA at the eval step (same value the `train` table carries). |
| `lr_gate_train_declining` | LR-drop gate 1: train EMA fell by more than `--lr-train-slope-min` over the trailing `--lr-patience` evals. |
| `lr_gate_ce_regressing` | LR-drop gate 2: the last `--lr-patience` `cal_ce_stop` values all sit more than `--lr-ce-margin` above the running best. A drop fires only with the no-improve streak >= `--lr-patience` and both gates true. |
| `ce_excess` | `min(last lr_patience cal_ce_stop) - best_ce`: the regression gate's margin scalar (fires when > `--lr-ce-margin`). NaN before `lr_patience` evals exist. |

## `ic_member{i}.parquet` -- long-format rank-IC records

One row per (eval `step` x population x horizon x score x mask x role x tradability). Per eval: 96 tag-0 rows (2 scores x 4 horizons x 2 masks x 3 roles x 2 trad), 8 tag-1 rows and 8 tag-2 rows when those populations exist (their mask/role/trad collapse to `primary`/`all`/`all`).

| column | values / meaning |
|---|---|
| `step` | Optimizer step of the eval. |
| `tag` | Population: 0 = val rows of non-holdout tickers (the primary evaluation population); 1 = holdout tickers' val-era rows (report-only); 2 = fixed train-era panel (report-only, present only when a panel is configured -- r1/r2 have none). |
| `horizon` | `1d` / `1w` / `1m` / `6m`. |
| `score` | `p_up` = P(up); `score` = P(up) - P(down), the consumer-aligned ranking signal -- the variant stopping and the gates read. |
| `mask` | `primary` = the trained label mask (spike censorship, calendar-gap ceilings, embargo, finiteness); `uncensored` = the alternative outcome mask that keeps spike-censored outcomes (the censorship sensitivity). |
| `role` | `all` = every val date; `stop` / `gate` = labels whose stored per-label target dates fall in stopping / gating blocks (block dates and roles are in `split_meta.json`). |
| `trad` | `all`, or `tradable` = close >= $5 and 63-session median dollar volume >= $1M at the anchor, causal. |
| `mean_ic` | Mean over qualifying dates of the per-date Spearman rank IC between score and realized centered z; a date qualifies with >= 30 mask-passing names (floor auto-scaled down for small dev universes). |
| `se`, `t` | Newey-West (Bartlett) HAC standard error and t = mean/se; lag `max(5, d_h - 1)`, computed within val blocks as independent clusters, calendar-hole-aware. |
| `n_dates` | Qualifying dates behind the mean. |
| `ic_std` | Plain across-date standard deviation of the per-date ICs. |

Discipline: `role = "gate"` rows are written for the pre-registered gate reads that happen after training and are never printed or consulted during a run. Reading them to steer stopping, tuning, or design decisions consumes the gating surface; leave them to the gate-read step of the protocol (`docs/v5_run_guide.md` section 5).

## `top_member{i}.parquet` -- top-of-ranking records

One row per (eval `step` x horizon x mask x trad); 16 rows per eval. The ranking score is always the consumer `score`; `mask`/`trad`/`n_dates` mean the same as in the `ic` table. Only dates passing the >= 30-name floor contribute; HAC clusters here are contiguous session runs rather than val blocks.

| column | meaning |
|---|---|
| `top1_z_mean`, `top1_z_se`, `top1_z_t` | Per-date realized centered z of the single top-scored name: mean over dates, HAC se, t. |
| `topdec_z_mean`, `topdec_z_se`, `topdec_z_t` | Same for the mean of the top decile (k = max(1, n/10) names per date). |
| `topdec_ret_mean`, `topdec_ret_se`, `topdec_ret_t` | Top-decile mean in return units: `z * sigma_hat * sqrt(d_h)`, the label's own denormalization. The variant the 1d cost gate compares against basis-point cost thresholds. |

## `train_member{i}.parquet` -- one row per `--log-every` steps

| column | meaning |
|---|---|
| `step` | Optimizer step. |
| `lr` | Actual optimizer LR at the step: `--lr` x warmup ramp x `lr_scale`. |
| `loss_ema` | 0.98/0.02 EMA of the total masked training loss (histogram CE + 0.1x class auxiliary), dropout active. |
| `loss_ema_{1d,1w,1m,6m}` | Per-horizon histogram-CE EMAs (float32; updated only on finite positive batch terms, so early rows can be null). |
| `it_s` | Mean steps/s since member start. |
| `elapsed_s` | Wall seconds since member start. |

Schema evolution is additive: new writer fields become new columns in later runs' files, and readers must select columns by name.
