Spot-check complete: my own independent replay reproduces D1's r11 milestone table exactly (1.737/1.530/1.127/1.045/1.019/1.251/0.981/1.368/1.085/0.931), syn frozen denominator 0.836982, and all six sc/tc frozen/floor ratios = 1.0 exactly. The verified fact base holds. Final specification follows.

---

# FINAL COMMITTED SPECIFICATION — `norm_freeze_step`: the stationary-loss branch

## Conflict resolutions (binding, one line each)

1. **Trainable freeze (C0/D1/D3) beat report-only (D2/D4).** The mandate's hard constraint ("only normalization/aggregation/weighting of the existing raws may change") and its demanded sections (force evolution, one-variable paired run vs the r14/r11 lineage) are only non-vacuous for an objective-touching branch; the "covert λ_syn anneal" objection is a predicted-dynamics concern, which per doctrine gets an instrument and a paired run, never a veto — and the measured fact that the entire objective delta is confined to syn (sc/tc frozen ≡ floor, bit-exact, 0 re-crossings in 3×~19.5k steps) makes the price one quantified force change with a pre-registered fallback.
2. **Freeze-at-T beat fixed-from-burn-in (D4's no-event constants).** Fixed-from-step-1 would either stay a second logged scalar (fails the headline mandate) or, promoted into the objective, change the fragile pre-460 opening race (fixed sc/tc units read ~20 at step 1 vs EMA's 1 — contraction-tilted during calibrate_init/warmup) and pin syn to a fiat unit 1.0; freeze-at-T keeps the r1-safe burn-in verbatim and freezes syn at a measured state.
3. **T = 1000, point-snapshot EMA, grounded in the drift data.** Floor engagement is measured at steps 371–460 (r10/r11/r14) with zero re-crossings ever, so any T ≥ ~500 makes the sc/tc freeze provably gradient-identical to history; warmup ends at 1000; indicator-curve T-sensitivity is corr ≥ 0.99936 over T∈[500, 2000]; syn's unit varies only −21%/+0% (EMA: 0.661 @500 — the 0.79 figure in D2 was an erratum — 0.837 @1000, 0.70 @2000). Windowed-median and plateau-detection snapshots rejected: β=0.99 already smooths ~100 steps, the sc/tc snapshot is irrelevant (max() resolves to the floor), and a data-dependent trigger makes T uncomparable across runs.
4. **No live shadow scalar ships, in either mode (D1 beat D3/D4 here).** D3's verifier proved the shadow is the one place determinism fails (a KNOB toggle at resume past shadow-T captures wrong-T constants); the offline backfill tool reproduces the identical curve at ≤2e-7 for any run, past or future — D4's entire capability is subsumed by it at zero state cost.
5. **Syn freezes (D1/D2/D3 converge over D4's dissent).** Unfrozen syn is the objective's only remaining live shrinking denominator (mild r1-class force-holding, denom 0.837→0.418 over r11) and a dead pinned-≈1 component worth ~25% of the late-run headline — it *is* the adulteration being repaired.
6. **Gate-the-EMA-update (D1) beat observe-only-EMA + frozen dict (D3/D2).** Identical frozen constants either way (both capture `max(EMA_T, κ·first)+eps`), but the gate keeps the checkpoint state-dict shape literally unchanged — no new key, no new legacy-load branch, nothing to shard — and its only benefit forgone (free `total_ema` logging) is recoverable offline.
7. **Smoothing conventions pinned (D2 verifier correction).** All quoted trajectory numbers are CENTERED 250-step rolling medians; the decomposition uses centered 250-step means. Acceptance checks written against trailing windows will spuriously fail (plateau crossing moves 10,504→10,791, flattening ratio 5.3×→4.1×).
8. **Backtest verdict on indicator quality, including r10:** the frozen scalar is a real train-side equilibrium indicator (r11: corr −0.943/−0.945 with stratified margin while progress existed, 5.3× slope break at the 10.5k margin plateau; r14: 100% monotone decline so far) and is **deliberately blind to memorization** (r10: declines 1.472→1.257 while margin bleeds 1.442→1.117) — this is documented as a property, not hidden.

---

## 1. Design statement

One run-permanent config field, `norm_freeze_step: int = 0` (CLI `--norm-freeze-step`), default 0 = byte-for-byte the historical computation. Set to T (recommended 1000), steps ≤ T execute today's EMA + κ-floor path verbatim (the burn-in and r1 guard stay); from the first step > T, `EmaNormalizer` simply **stops folding new raws into the EMA**, so the unchanged denominator formula `max(EMA, κ·first) + eps` becomes a per-run constant for every EMA-path term (sc_{full,self,peer}, tc_{full,self,peer}, syn — seven terms; the twelve fixed-scale terms were constants already). From T+1 the logged-and-optimized `total` is a stationary λ-weighted sum of the untouched raw terms: its level is a true distance-to-equilibrium read, and the gradient direction it produces differs from history in **exactly one coordinate — syn** (the sc/tc frozen constants are bit-equal to the κ-floor constants history already used from step ~460 onward, verified on 3×~19.5k real steps with zero EMA re-emergence). That one change — syn's late-run force decaying to 0.50× historical by 20k — is quantified, instrumented, and priced by a one-variable paired run with a pre-registered margin band and an invariant-preserving fallback (λ_syn 0.3→0.6 under freeze). Legacy and flag-off runs get the identical indicator from a fail-loud offline backfill tool validated to ≤2e-7 against r1/r10/r11/r14.

## 2. Normalizer mechanics

`losses.py:431-441`, with the only new logic marked:

```python
def normalize(self, name, raw, fixed_scale=None):
    if fixed_scale is not None:                       # xsep/psep ÷ m_sep²=8, util ÷ v0+λ_cov=2,
        self.last_denom[name] = fixed_scale           # anc ÷ √v0=1 — UNTOUCHED (already stationary)
        return raw / fixed_scale
    r = float(raw.detach())
    prev = self.ema.get(name, r)
    first = self.first.setdefault(name, r)
    denom = max(prev, self.kappa_floor * first) + self.eps   # UNCHANGED formula, eps placement identical
    self.last_denom[name] = denom
    frozen = self.freeze_step > 0 and self.step > self.freeze_step   # NEW (the only new logic)
    if name not in self.ema:
        self.ema[name] = r        # seeding always allowed; if it happens while frozen, the term
                                  # freezes at its first-seen value + one-time WARNING (R4)
    elif not frozen:
        self.ema[name] = self.beta * self.ema[name] + (1 - self.beta) * r
    return raw / denom
```

- **Constructor**: `EmaNormalizer(beta, eps, kappa_floor, freeze_step=0)`, with **`self.step = 0` initialized in `__init__`** (D1 verifier correction — without it, any pre-loop call with the flag on is an AttributeError). Built at `train.py:531` with `cfg.norm_freeze_step`.
- **Step plumbing**: the train loop sets `norm.step = step` (the same 1-based counter it logs/checkpoints) immediately before `combine(...)` at `train.py:573`. Transient attribute, never serialized; restored implicitly on resume by the loop.
- **Freeze boundary (use-then-update)**: step T uses EMA_{T−1} and folds raw_T → EMA_T; step T+1 uses EMA_T and never updates again. Frozen denominator = `max(EMA_T, κ·first) + eps` — exactly the value the historical path would have used at T+1, so the scalar is **continuous at the boundary** (measured: 1.737 at step 1001 in both schemes; pre-T identity exact on r11/r14).
- **κ-floor post-freeze**: still applied by the unchanged formula; with `ema` constant it is absorbed into the constant — C0's "yes by construction," confirmed.
- **`last_denom`/`grad_force_diag`/`rec["denom"]`**: untouched code (`losses.py:489`, `train.py:624`) automatically carries the active (frozen) denominators — force diagnostics reflect the real gradient scaling for free.
- **Freeze-event record (promoted to committed spec per D3 verifier)**: at the first step > T, after `combine`, append to train_log one record `{"step": step, "norm_freeze": {name: denom for the 7 EMA-path terms}, "floor_ratio": {name: denom/(κ·first[name]+eps) for the 6 sc/tc terms}}` and print one console INFO line. **Loud WARNING if any sc/tc floor_ratio > 1.0** (means T predates floor engagement — misconfiguration, see R3). On crash-resume across the boundary the record duplicates with identical content (deterministic); offline readers dedupe keep-last (r14's rewind precedent, steps 2751–2758).
- **Post-T anomaly warnings, split by term class (D1 verifier correction #1 — do NOT use one threshold)**: at diag steps, WARNING if any **sc/tc** normalized value > 1.0 (measured zero false positives across 21k+ post-T steps in r11+r14 — a genuine anomaly signal); **syn** warns only at normalized > 2.0 on **two consecutive diag steps** (syn exceeds 1.0 on 18.4% of healthy r14 post-T steps and 2/8 diag steps — a 1.0 threshold would train operators to ignore the design's one fail-loud channel).

## 3. Config fields, defaults, CLI

Exactly one field:

```python
# config.py, loss section, adjacent to kappa_floor
norm_freeze_step: int = 0   # >0: stop the EMA-path normalizer updates (sc/tc/syn) after this step;
                            # denominators freeze at max(EMA_T, kappa_floor*first)+eps and total
                            # becomes a stationary lambda-weighted objective from T+1 on (level =
                            # distance to training equilibrium; NOT a holdout signal — see r10 note).
                            # 0 = historical EMA path, byte-identical. Run-permanent (changes the
                            # objective): NOT a KNOB_FIELD. Recommended T=1000: floor engagement is
                            # measured at steps 371-460 (r10/r11/r14, zero re-crossings), so at
                            # T=1000 the sc/tc freeze is provably gradient-identical to the
                            # floor-pinned historical path; the only objective change is syn.
```

- **NOT in `KNOB_FIELDS`** (`config.py:112`) → the existing `_resume_config_diffs` (`train.py:407-428`) refuses drift automatically (invariant 2). Deliberately an explicit int, not coupled to `warmup_steps` (coupling would silently move the freeze when warmup is its own experiment).
- **CLI**: `ap.add_argument("--norm-freeze-step", type=int, default=None, help="freeze the EMA-path loss normalizers after this step (0 = historical EMA path); run-permanent")` — **and `"norm_freeze_step"` is explicitly added to the None-override setattr tuple at `train.py:773-777`** (D1+D2 verifier correction: without this line the flag parses and silently no-ops — the exact silent-skew invariant 7 forbids).
- **Validation (train() preamble, `loss_chunk` pattern at `train.py:475-476`)**: `norm_freeze_step < 0` → `ValueError`; `0 < norm_freeze_step >= max_steps` → loud WARNING ("freeze will not fire this session"; not a refusal, since max_steps is a knob and may be extended on a later resume).
- **No other fields.** No `shadow_total`, no `shadow_freeze_step` (resolution 4).
- **Backfill tool ships** as `StockIdentityModel/tools/backfill_total.py` (D3's spec, committed): `--run-dir`, `--freeze-step` (default 1000); keep-last dedup of resume-duplicated steps; rebuilds λ/fixed-scale tables from the run's own `config.json`; replays the EMA in host float64; **fail-loud validation** — replayed historical total vs logged `total` rel err > 1e-4 aborts (measured error is ≤2.0e-7, three orders of headroom); writes `{run_dir}/total_frozen.jsonl` with `{step, total_frozen, total_ema}`. This is how r1–r14 and all future flag-off runs get the indicator.

## 4. Checkpoint shape, legacy load, resume across the boundary

- **State-dict shape: UNCHANGED** — `{"ema": {...}, "first": {...}}` (`losses.py:443-444`). The frozen denominator *is* the stalled EMA flowing through the unchanged formula; there is nothing to store. No new key, no shard concern (host-side floats, device-0 loss path only — invariant 4), legacy flat-dict load (`losses.py:450-452`) untouched.
- **Old checkpoints + new code, flag off**: `config.json` lacks the field → `Config.load` (`config.py:386`) fills default 0 → historical path bit-for-bit; `_resume_config_diffs` compares 0 vs 0 (invariant 6).
- **Old checkpoint + flag on**: checkpoint-config 0 vs run 1000 → **refused** (correct: objectives can't splice).
- **Resume-replay exactness (invariant 3)**: the freeze is not an event — it is the comparison `step > freeze_step` of a checkpointed counter against a run-permanent field. Resume at S < T: `ema`/`first` restore bitwise, steps S+1..T replay the identical β-fold (CPU resume is bitwise, verified incl. kill -9 mid-interval), EMA_T identical. Resume at S ≥ T: `ema` is constant = EMA_T, carried in every checkpoint. kill -9 between T and the next checkpoint: replay re-derives the identical constants; the duplicate freeze record has identical content (keep-last dedup). **Bitwise gates are scoped to CPU** (D2 verifier: GPU baseline bitwise-ness is unestablished for unrelated reasons; the branch adds no new nondeterminism class).
- **Residual (R6)**: *old code* resuming a *new flag-on* checkpoint silently drops the field (old `Config.load` filters unknown keys) and splices objectives — not auto-detectable by old code; mitigated by run-ledger note + operational rule (resume with the repo at the run's commit). New-code-on-old is fully safe.
- **Byte-identity flag-off (invariant 1, deductive, verifier-walked)**: `freeze_step=0` ⇒ `frozen` False every step ⇒ identical float ops in identical order; no new log fields, no console change; one int comparison of added work. Same bar as REL-3, gated by acceptance run G1 below.

## 5. r1/r2 guard audit (with verifier-corrected prose — this wording goes in the ledger)

- **r1 (denominator shrinks with its own term → 1/√L gradient self-amplification)**: post-T every denominator in the objective is a constant — the mechanism class is **deleted**, not bounded: amplification factor exactly 1 (historical floor capped it at 1/κ = 20×; frozen is strictly tighter). Pre-T is the historical guard verbatim. The seeding-while-frozen branch matches historical first-sight behavior for any hypothetical post-T term debut; note (corrected): even without seeding, `first.setdefault` precedes the gate, so the counterfactual is κ-floor-capped at 1/κ, not unbounded — seeding is right for behavioral fidelity, not necessity.
- **r2 (l1 anchor ratchet)**: anc is on the fixed √v0 path in both modes (`losses.py:402-403`); never touches the gate.
- **Inverse direction (corrected — NOT "identical to historical")**: a frozen denominator cannot absorb a rising raw. sc/tc *transient* spikes read identically to history (both paths floor-pinned). A *sustained* collapse-grade reversion differs: the historical EMA would re-inflate the denominator within ~100–460 steps and soften the read back toward 1; frozen pins the floor forever and keeps screaming — strictly louder, the correct fail-loud direction. Syn likewise: historical EMA re-pins a runaway to ≈1 within ~100 steps; frozen reads raw/0.837. Gradient *magnitude* entering Adam stays capped by clip_norm=1.0 always-on (untouched — r9 load-bearing); only direction can tilt toward the screaming term.

## 6. Force evolution + paired-run protocol

- **sc/tc: provably zero change at T=1000.** max |log(hist_denom/frozen_denom)| = 0.0 exactly, all six terms, all 19,000 r11 + 2,168 r14 post-T steps; zero EMA re-emergence (re-emergence would need a sustained ~3.5–22× raw rise from T-levels). The fact base's "late-run force-holding lost" worry is structurally void: history lost EMA force-holding at step ~460 when the floor pinned; the freeze inherits the identical constants. Contraction/expansion ratio under freeze **equals** historical at every diag step (r11: 0.871/2.297/1.651/1.814/2.282/1.579 at 1.25k/2k/5k/10k/15k/20k — these measured values supersede the fact base's loose "1.2–1.9× throughout").
- **syn: the one real change — and the owner draft's predicted sign is wrong.** Syn's raw *falls* (0.84 @1k → 0.39 @20k smoothed), so the live EMA denominator falls below the frozen 0.837; the freeze **reduces** syn's force: multiplier 1.000 @1001, 0.806 @2.5k, 0.651 @5k, 0.561 @10k, **0.500 @20k** (r11; r14 0.87 @2.5k) — deductively equivalent to annealing λ_syn 0.3→0.15 over the run. Syn is the largest single per-term force late-run; its share drops from the historical climb 0.131→0.224 to ~0.126.
- **Observables**: `force_syn` at diag steps (predict 0.5–0.65× the flag-off twin late-run); `I_self/I_full`, `I_peer/I_full` drift off 3.4×/3.1× if the lost pressure mattered; c/e ratio essentially unchanged (syn is in neither bucket); hinge raws; `var_dim` spectrum; `zbar_dist`; margins.
- **Paired-run protocol (one variable)**: r14's REL-3 config, flag-off vs `--norm-freeze-step 1000`, same `train_seed`, same `devices` string, same data. Read `margin_ratio_stratified` + retrieval at matched steps plus the observables above. **Pre-registered decision rule**: recommend flag-on as default for new runs iff stratified margin at best-step is within r11's measured oscillation band (±0.021) or better with no collapse signals; if margins degrade beyond the band, the engineering fallback is a third run with **λ_syn 0.3→0.6 under the freeze** (restores syn's late-run force level while keeping every denominator constant — invariant-preserving, again one variable).
- **Engineering acceptance gates**: **G1** flag-off vs historical code, same seed, ~1.5k CPU steps → bitwise-identical train_log/eval_log/state (the REL-3 bar). **G2** flag-on vs flag-off, max_steps ~1.2k → bitwise identical through step T inclusive; first divergence, if any, strictly after T and only via the syn gradient path. **G3** flag-on run kill -9 between T and the next checkpoint, resumed on CPU → frozen denominators, train_log totals, and state bitwise equal to an uninterrupted twin; freeze records identical content.

## 7. Backtested indicator trajectories (real logs, independently replicated three times this session — designer, verifier, synthesizer)

Replay fidelity: ≤2.0e-7 vs logged `normalized`/`total` (float32 logging quantization), exactly 0 vs logged `denom`, on r10/r11/r14 (r14 after keep-last dedup of its resume seam) and r1 under its own config. Conventions: trajectory = centered 250-step rolling median; decomposition = centered 250-step mean.

**r11 (healthy 6,676-ticker run), T=1000** — per-step values at milestones, with smoothed line:

| step | hist total | frozen total | frozen (smoothed) | margin_strat |
|---|---|---|---|---|
| 1001 | 1.737 | 1.737 | — | — |
| 2500 | 1.601 | 1.530 | 1.437 | 0.621 |
| 5000 | 1.205 | 1.127 | 1.289 | 0.724 |
| 10000 | 1.125 | 1.019 | 1.155 | 0.861 |
| 10500 | 1.389 | 1.251 | 1.119 | 0.892 (peak) |
| 15000 | 1.631 | 1.368 | 1.042 | 0.830 |
| 20000 | 1.067 | 0.931 | 1.039 | 0.863 |

- Smoothed drop per 5k segment: −0.489 / −0.134 / −0.113 / **−0.003** — clean approach-to-equilibrium; **5.3× slope break at the 10.5k margin plateau** (−0.047/1k → −0.009/1k), with the residual slope honestly reflecting that r11's consistency raws were still falling 0.58× over the last 10k (equilibrium genuinely not reached at horizon — final flattening lands ~15k).
- Correlation with stratified margin: **−0.945 from T to the 10.5k plateau (38 evals), decoupling to −0.166 after** holdout saturates; full-run Pearson −0.943 (vs −0.933 for the legacy scalar, n=76).
- Level honesty: @20k the legacy syn component reads 0.272 (normalized 0.908 — EMA-pinned, meaningless); frozen reads 0.136 (normalized 0.454 — syn's raw genuinely improved 2.2× since T). Decomposition 1k→20k: anc 0.466→0.132 (the true convergence carrier), syn 0.300→0.150, consistency 0.545→0.437, util 0.353→0.220, hinges **rising** 0.108→0.138 (fences engaging at scale — real state).
- Noise: per-step sd 0.146–0.151 (CV envelope 0.076–0.146; one r10 window touches 0.313); at the 250-median grain noise is ~0.011, so the 10.5k→20k drop (~0.07–0.09) is a 6–7× noise signal. **Reading contract: single steps need ±0.3 error bars; trends are called on ≥3 consecutive 250-step medians moving ≥0.033.**
- T-sensitivity: smoothed-curve correlation 0.99936 (T=500) / 0.99964 (T=2000) vs T=1000.

**r14 (REL-3, mid-run @3,168)**: floor engagement 406–455; syn EMA_T 0.9735; frozen smoothed 1.864 @1250 → 1.599 @3000, **100% monotone-declining** at the 250-step grid so far; margin corr −0.819 Pearson / −0.881 Spearman (n=8 — thin, directional only).

**r10 (the honesty check — slide-window memorization run)**: frozen total declines **1.472 @10.5k → 1.257 @20k while stratified margin bleeds 1.442 → 1.117**; its plateau-10% crossing lands ~3,700 steps *after* its margin peak (vs r11's crossing at ~10.5k, on the margin peak). **What the scalar indicates: distance to the training objective's equilibrium — declining means optimization is still moving toward it, flattening means it has arrived. What it does not indicate: holdout quality.** It is blind to r10-class memorization by construction; the r11-vs-r10 contrast (flattening at the margin peak vs grinding far past it) is readable only from the indicator *paired with* eval curves, which remain the sole selection authority (doctrine point 4 survives unmodified).

**r1 (collapse, from D4's verified probe)**: the stationary read bottoms ~4.57 then rises ~11% with the signature cons→0.000 + hinge→3.998 (its λ-ceiling 4.0) + util→0.499 (its 0.5 ceiling) — collapse crashes the indicator loudly *and* the fixed-unit decomposition names the failure mode.

## 8. Logging/schema changes

- **Flag off**: zero changes — train_log byte-identical (no new keys, no console delta).
- **Flag on**: (a) one one-time JSONL record `{"step": T+1, "norm_freeze": {...7 denoms...}, "floor_ratio": {...6 sc/tc ratios...}}` (additive-key precedent: `sdpa_ctx_backend`, `rel`, `mem_gb`); (b) `total`/`normalized`/`denom`/`force` fields keep their exact meanings and now describe the stationary path — no renamed or repurposed fields, existing parsers unaffected; (c) console INFO at freeze; WARNINGs per §2 (sc/tc >1.0 at diag; syn >2.0 at two consecutive diags; post-T term debut; floor_ratio>1 at freeze).
- **Offline**: `tools/backfill_total.py` output `total_frozen.jsonl` (§3) for legacy/flag-off runs.

## 9. Risk register

| # | Risk | Detection signal |
|---|---|---|
| R1 | Halving syn's late-run force (largest per-term force; λ_syn is the principal tuning knob) shifts the masked-vs-full equilibrium, costs holdout margin | Paired run (§6) with ±0.021 pre-registered band; per-run: `force_syn` 0.5–0.65× twin, I_self/I_full drift off 3.4×; fallback run λ_syn=0.6 under freeze |
| R2 | sc/tc EMA re-emerges above the κ-floor in a future regime (frozen ≠ historical there) | Freeze record's `floor_ratio` (≈1 expected) + post-T sc/tc normalized >1.0 WARNING (measured zero false positives in 21k+ steps) |
| R3 | T set before floor engagement (e.g. 200) → sc/tc freeze at EMA_T ≫ floor, real force decay vs history | Freeze-time `floor_ratio` > 1 → loud WARNING; field comment pins T=1000 to the measured 371–460 engagement window |
| R4 | New term debuts post-T (shouldn't exist; cross-market adds no names) | Seeded-freeze at first-seen value + one-time WARNING naming the term |
| R5 | Resume drift on the field | Automatic — not in KNOB_FIELDS, `_resume_config_diffs` refuses |
| R6 | Old code resumes a new flag-on checkpoint, silently splices objectives | Not auto-detectable by old code; run-ledger note + resume-at-run's-commit rule; backfill validation fails loudly on a spliced log |
| R7 | Collapse-grade raw reversion post-freeze reads ≈20 and tilts gradient direction | That *is* the detection (fail-loud; r1 backtest shows the signature); magnitude capped by clip_norm=1.0 always-on |
| R8 | Operator reads the scalar as a generalization signal | r10 numbers documented beside the field (declined 1.47→1.26 through a 1.44→1.12 margin bleed); eval curves remain the authority |
| R9 | Per-step noise (sd ~0.15) misread as trend | Documented reading contract: 250-step centered medians, ≥0.033 move over ≥3 medians |
| R10 | Cross-run level comparison (units are per-run: 0.05×first-step draw for sc/tc, EMA_T for syn) | Documented with D4's measured counterexample (r14 1.59 vs r11 1.35 at 3k while r14 *led* margins); freeze record makes units auditable; only fixed-unit parts (anc/util/hinge) compare across runs |
| R11 | Implementation leaks into flag-off path | Gate G1 (bitwise vs historical code); flag silently no-ops if the setattr-tuple line is forgotten — covered by gate G2's required post-T divergence check |

## 10. Documentation language (verbatim, for README/config docstring — claims limited to what the backtests support)

> **`norm_freeze_step` (stationary-loss branch, default 0 = off).** With T>0, the EMA-path loss normalizers (sc/tc/syn) stop updating after step T (recommended 1000); every denominator becomes a per-run constant and the logged `total` becomes a stationary objective from T+1. Backtested against r11/r14/r10 logs: post-T it declines while optimization approaches the terms' equilibrium and flattens when it arrives (r11: 5.3× slope break at the 10.5k eval plateau; correlation with stratified holdout margin −0.94 while margins were improving), and the fixed-unit decomposition carries real level meaning (anc 0.47→0.13 = true anchor convergence; the legacy scalar pinned syn at ≈1 forever and hid its 2.2× improvement). **It is a train-side indicator only**: in r10 it declined smoothly through a holdout-margin bleed (1.47→1.26 vs margin 1.44→1.12) — it cannot detect memorization; eval curves remain the only model-quality signal. Read it as centered 250-step rolling medians (per-step draw noise sd ≈ 0.15); levels compare within a run, not across runs. At T=1000 the sc/tc freeze is measured gradient-identical to the historical κ-floor-pinned path (frozen/floor = 1.0 exactly, zero EMA re-crossings in r10/r11/r14); the one objective change is syn, whose late-run force decays to ~0.5× historical — priced by a paired run before flag-on becomes the recommended default. Run-permanent: resume refuses drift. Steps ≤ T are the historical burn-in and are not indicator-grade. Historical/flag-off runs get the identical indicator via `tools/backfill_total.py` (replay validated ≤2e-7; aborts loudly above 1e-4). *Ledger correction carried with this branch: a sustained post-freeze sc/tc reversion is NOT identical to historical behavior — the historical EMA re-absorbs it within ~100–460 steps while frozen denominators keep reporting it at full scale (strictly louder, by design).*

**Honest weaknesses (carried into the README):** blind to memorization by construction (r10); per-step noise requires smoothed reads; the syn force change is a quantified-but-open dynamics conjecture until the paired run; units are per-run constants inheriting first-step draws and EMA_T (syn's unit is "syn at warmup end" — arbitrary but stationary and logged); flattening trails the margin plateau (~15k vs 10.5k in r11) — the indicator answers "has optimization equilibrated," not "when to stop."