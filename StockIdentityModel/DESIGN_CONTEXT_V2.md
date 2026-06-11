# ContextModule v2 — FINAL COMMITTED SPECIFICATION ("REL-3")

Three pre-norm blocks, eight heads, GELU FF, and a single shared **relational logit-bias tensor** per (window, scale) that fuses day-level co-movement evidence with a learned non-bilinear pair-scorer — all capacity injected at the attention-logit level, where the self route (single-key softmax) is provably blind and 100% of function lands in the relational circuits.

## Element adjudication (winner + one-line why)

| Decision | Winner | Why |
|---|---|---|
| Day-level evidence channel (deficit a) | **D4's 3 planes** (corr0 masked+shrunk, beta_prod, dlogvol) over D1's learned Φ-Gram bank | D1 measured 15.2 GB OOM-class transient + 6-site phi threading + 2 of its 3 init channels measured dead (lead/lag SNR 0.15 at N=64); D4's estimators measured unbiased under halts. |
| beta_prod | **Dropped from the run set** (post-implementation measurement; supported opt-in) | Its group-mean factor includes the observer's returns ⇒ an observer→peer-row *weight* channel on the exact peer path — measured: planes(i,j≠o) move 4.5e-2 under r_o perturbation, corr0/dlogvol exactly 0. Compounds with D4's own findings (β std ≈0.10 at g=16; train/eval grouping shift), all concentrated at the fine scales where 1/g dilution is weakest; corr0 spans factor structure (≈ ββᵀ + residual) so the information loss is small. Run set: `("corr0", "dlogvol")`. |
| corr_lead1/corr_lag1 | **Dropped** (D4 measurement) | SNR 0.15 ⇒ ~98% noise; errors-in-variables attenuates the learned slope to ~2% — pay 2 planes for nothing. |
| Bias scope | **D4's one tensor, shared across blocks and full/peer views** over D1/D2 per-block | Per-block biases at L=4,848 are the deterministic OOM in both verifier reports; sharing is the verified memory discipline. |
| Non-bilinear scoring (deficit b) | **D2's PSN, hoisted**: computed once per segment from Hg (block-0 inputs), folded into the shared bias | Keeps the expressivity-class escape (GELU ridges ⊃ bilinear) while resolving D2-verifier's 27 GB FAIL; closure simplifies — no per-copy PSN on the exact path. |
| FF type | **D2's lean GELU** over C0/D4's SwiGLU | SwiGLU is view-invariant pointwise capacity — the exact r10 shape; the nonlinear *relational* class is already delivered at the logit level. |
| Depth | **L_ctx=3** (D2/D3/D4 unanimous) over C0's 4 | Query-channel magnitude grows with depth (0.36→0.48 measured); block 4 is pure self-route FF stacking for ×1.6 step. |
| Heads | **n_heads_ctx=8**, context-only (all designers) | Zero params, 8 routing channels for the 2-group bias. |
| Temperature | **D2's per-block per-head log_tau** | Zero memory, fused-preserving, attacks softmax diffusion over ~4,848 keys, self-route invisible. |
| D3's GST hub | **Excluded** | Attacks neither diagnosed deficit; adds a new approx-path leak funnel + block-0 self-content channel (its own verifier's finding) + the heaviest plumbing diff. |

All load-bearing verifier corrections are applied: conditional module construction, fail-loud bias threading, out-of-place mask fills, inner-checkpointed pair chunks, `edge_chunk` in `KNOB_FIELDS`, math-backend memory planned as a real branch.

## 1. Architecture statement

The ContextModule becomes a 3-block, 8-head, GELU-FF pre-norm stack whose attention logits at every block are additively modulated by **one** per-(window, scale) bias tensor `bias ∈ [B, G=2, L, L]` (2 head-groups of 4 heads), computed inside the checkpointed segment as `bias = edge_cap · tanh((EdgeMLP(planes) + PSN(Hg)) / edge_cap)`: `planes` are two day-level pairwise statistics in the run set (shrunk masked-Pearson `corr0`, vol asymmetry `dlogvol`; `beta_prod` implemented but opt-in — see its adjudication row) built from the close-return channel and a halt-validity mask by bmms; `PSN` is a zero-init additive-ridge pair scorer `W2·GELU(P_q·LN(H_i) + P_k·LN(H_j))` that provably escapes the rank-≤head_dim bilinear family. The bias is shared across all three blocks and across the full and peer views (masked per view, out-of-place), never touches the value path, and is exactly zero at init; each block additionally carries a zero-init per-head log-temperature scaling q. The self view keeps its bool-mask path — single-key softmax is bias- and scale-invariant, so every new parameter is structurally invisible to the self route. Deficit (a) is attacked by injecting day-level joint structure the summaries cannot carry; deficit (b) by both the data-side bias and the non-bilinear PSN; capacity placement is the anti-r10 extreme: 4,468 of the new params are logit-level relational mechanism, and the only pointwise addition is block 3's standard FF. With the run set, the exact peer path's content closure is total: peer-pair planes are measured exactly invariant to observer perturbation.

## 2. Module-level spec (tensor shapes)

**`model.py — compute_edge_planes(Rg, Vg, n0, min_overlap) → [B, L, L, 3]`** (pure function, no params, no RNG). Inputs `Rg [B,L,N]` = close-return channel (feats `[...,3]`) scattered like `Hg`; `Vg [B,L,N]` bool = `feats.abs().amax(-1) > 0`.
- `corr0`: masked demean over valid days; `A = bmm(R̃,R̃ᵀ)`, `B2 = bmm(R̃²,Vgᵀ)`, `n = bmm(Vg,Vgᵀ)`; `ρ̂ = A/sqrt(B2∘B2ᵀ+ε)`; shrink `ρ̂·n/(n+n0)`; zero where `n < min_overlap`. All ε/`max(n,1)` guards mandatory (pad-diag is visible in the NaN guard — bias(pad,pad) must stay finite; it does: every op is bounded, `tanh` caps it).
- `beta_prod`: group-mean factor `m [B,N]` over valid entries; `β_i = corr_masked(r_i, m)` `[B,L]`; plane `= β βᵀ` (outer product).
- `dlogvol`: `v_i = ½log(var_valid(r_i)+1e-8)`; plane `= v_i − v_j` (outer difference).
Cost: 3 bmms `[B,L,N]×[B,N,L]` ≈ 9 GMAC/window summed over the ladder — <1% of context compute.

**`model.py — PairScore`** (new submodule, built iff `cfg.psn_rank > 0`):
```
ln:  LayerNorm(128)                     # pre-norm convention; buffer-free
P_q: Linear(128 → r=16, bias=True)      a = P_q(ln(Hg))   [B, L, 16]
P_k: Linear(128 → r=16, bias=False)     b = P_k(ln(Hg))   [B, L, 16]
W2:  Linear(16 → G=2, bias=True)        # weight AND bias zero-init
raw_psn[:, c] = W2(gelu(a[:, c, None, :] + b[:, None, :, :]))   # [B,|c|,L,2], per row-chunk
```
Computed **once per (window, scale)** from `Hg` — block-0 inputs, identical in all views; never per block, never per copy.

**`ContextModule.__init__`** — conditional construction (D4-verifier correction #1; preserves strict state_dict load of old checkpoints and init-RNG byte-identity):
```
if cfg.edge_stats:  self.edge = Sequential(Linear(len(cfg.edge_stats), cfg.edge_hidden), GELU(), Linear(cfg.edge_hidden, cfg.edge_head_groups))  # last layer zero-init
if cfg.psn_rank:    self.psn  = PairScore(cfg)
blocks built with n_heads=cfg.resolved_n_heads_ctx(); each block: log_tau = nn.Parameter(zeros(H)) iff cfg.attn_temp_ctx
```
All truthiness checks are `if cfg.edge_stats:` — never `== ()` (JSON round-trips tuples to lists).

**Bias build (inside the segment, before the block loop)**, under a nested non-reentrant checkpoint per row-chunk of `edge_chunk` rows (verified grad-exact; bounds saved-for-backward to ~0 for the pair/hidden tensors):
```
raw_c   = edge(planes[:, c]) + raw_psn_c                 # [B,|c|,L,2]
store[:, :, c, :L] = (edge_cap * tanh(raw_c / edge_cap)).permute(0,3,1,2)
```
`store` is a `[B, G, L, ⌈L/8⌉·8]` −inf-prefilled backing tensor sliced `[..., :L]` (cutlass bias-grad stride alignment; padding is lost on the exact path's expand→reshape copy — harmless, ≤56 MB scales).

**Masking (aliasing fix — both fills out-of-place):**
```
mask_full = bias.masked_fill(~vis_full[:, None], -inf)   # [B,G,L,L]
mask_peer = bias.masked_fill(~vis_po[:,  None], -inf)    # approx path
# exact path: bias[:,None].expand(B,L,G,L,L).reshape(B·L,G,L,L)  (expand→reshape materializes a
# fresh copy) then masked_fill_(~vis_c[:,None], -inf) in place on that copy
```
Built once per segment, the same tensor passed to all 3 blocks of its view.

**`Attention.forward(q_x, kv_x, vis, bias=None)`**: after the head split, `q = q * exp(log_tau)[None,:,None,None]` (if present). `bias=None` ⇒ shipped bool path, byte-identical. Else head-group merge: `q.view(B,G,H/G,L,hd).reshape(B·G,H/G,L,hd)`, mask `.reshape(B·G,1,L,L)`, one SDPA call, un-merge (verified numerically equal to per-head broadcast, and zero-bias equal to the bool path). `Block.forward` and `_run` gain a `bias=None` pass-through (D2-verifier note: `_run` is shared by full_view and z_self, so the flag lives there).

**View wiring**: full = `mask_full` every block; self = bool `vis_self`, `bias=None` (value- and grad-exact skip); `_peer_approx` = `mask_peer`, kv = full pass's block-l inputs, unchanged logic; `_peer_exact` = expanded copy-masked bias, column-o kill / row-o `vis_po` / `gate1` all carried by the existing `vis_c` machinery, untouched. PSN reads `Hg` only ⇒ no per-copy divergence issue; column o's PSN/plane entries are −inf-overwritten in copy o (zero gradient); row o's bias row uses `H_o`/`r_o` on the query/weights side only — the sanctioned constitutive class.

**`train.py` seams**: `run_step` computes once per window, outside the scale loop: `R = x[...,3]`, `V = x.abs().amax(-1) > 0`. Segment signature becomes `_context_views_segment(ctx, H, R, V, order_t, gid_t, pos_t, real, vis_peers, *ff_masks)`; `Rg/Vg` scattered inside with the same `gid_t/pos_t` scatter as `Hg`. Deterministic, recomputed exactly in backward. `ff_masks` loop already ranges over `cfg.L_ctx` — depth 3 needs zero mask-code changes. Non-checkpoint branch mirrors it.

## 3. Config fields (all byte-identical defaults; committed run values in parens)

```python
n_heads_ctx:      int | None = None    # None -> n_heads.  (8)   zero params, context-only
attn_temp_ctx:    bool = False         # per-block per-head log-temperature.  (True)
edge_stats:       tuple[str, ...] = () # () = no edge module built.  (("corr0","dlogvol") — beta_prod
                                       # supported but opt-in; see the beta_prod adjudication row)
edge_hidden:      int = 8              # edge-MLP hidden width (inert while edge_stats falsy)
edge_head_groups: int = 2              # bias channels; must divide resolved_n_heads_ctx()
edge_cap:         float = 4.0          # bias = cap*tanh(raw/cap) — deductive softmax-saturation bound
edge_shrink_n0:   float = 8.0          # corr0 shrinkage rho*n/(n+n0)
edge_min_overlap: int = 16             # joint-valid-day floor; below it pair stats are zeroed
psn_rank:         int = 0              # PairScore ridge width; 0 = not built.  (16)
edge_chunk:       int = 512            # inner-checkpoint row-chunk for the bias MLPs; value-exact
# L_ctx: existing field, default 2.  (3)
```
`edge_chunk` is added to `KNOB_FIELDS` (value-exact recompute plumbing — the `loss_chunk` precedent; an OOM mid-run must be fixable on resume; D2-verifier correction). Every other field is run-permanent automatically. `_resume_config_diffs` reads absent fields as today's defaults ⇒ old checkpoints resume as v1; splicing refused.

## 4. Init, optimizer, calibration

- **Zero-init chain**: `edge[2].weight/bias = 0`; `psn.W2.weight/bias = 0`; `log_tau = 0` ⇒ bias ≡ `cap·tanh(0)` = 0 and temp ≡ ×1 ⇒ **v2 ≡ v1-at-(depth 3, 8 heads) at init**, with exact, separable inference-time ablation forever (zero either output layer). `P_q/P_k`, `edge[0]`, `psn.ln` keep default inits.
- **Optimizer** (rule untouched): all new 2-D weights (`P_q, P_k, W2`, edge linears) → decayed (zero output layers stay zero under multiplicative decay until grads move them); `log_tau`, LN params, biases (ndim<2) → no decay — gain-like params correctly undecayed. Nothing named `context.out.*` ⇒ calibration exclusion untouched.
- **Calibration** (`calibrate_output_scale`): runs only at init where bias ≡ 0; it threads `rv=(R,V)` (the feats are already in scope at train.py:90) and still rescales `context.out` weight/bias only — a pre-softmax bias is invariant to output scaling. No escape hatch needed anywhere: fail-loud is universal.
- **Engagement**: output layers at zero ⇒ `P_q/P_k`/`edge[0]` get zero grad until `W2`/`edge[2]` move — standard ramp, instrumented (§8); engineering fallback: σ=1e-3 re-init of the zero layers if both share scalars are ~0 at step 2k.

## 5. Invariant threading (all 8)

| # | Invariant | How REL-3 satisfies it |
|---|---|---|
| 1 | Shared weights; view diffs = withheld evidence; no RNG in segment; ff-masks as tensor args | One weight set; one deterministic bias per (window, scale), identical for full and peer — only the visibility masks differ. `R/V` enter as tensor args; planes/PSN/tanh and the nested chunk checkpoints are RNG-free; ff-mask convention `[B,L,d_ff]` per block untouched, loop already `cfg.L_ctx`-sized. |
| 2 | Peer content closure, exact + approx | Bias is logit-only — **no value-path term exists**. Exact: column-o −inf in every copy ⇒ `corr0(·,o)`/`PSN(·,o)` unread, zero grad; row o's bias row rides `H_o/r_o` on the query/weights side — the verified constitutive class (H_o→wq). Approx: `vis_po` semantics and kv-from-full-pass unchanged; no new leak order. Temperature scales logits only. |
| 3 | Buffer-free; parameters()-only mirror sync | New state = Linears, LayerNorm, Parameters; zero buffers. Planes computed per device from the already-resident `x`. `_sync_mirrors`/`_merge_mirror_grads` unchanged. |
| 4 | Byte-identical defaults; run-permanence | Conditional construction ⇒ defaults build nothing: identical state_dict keys, init-RNG stream, forward values. Old checkpoints/artifacts strict-load. Only `edge_chunk` is a knob (value-exact). |
| 5 | calibrate rescales `context.out` only; decay rule | §4 — both rules untouched; bias ≡ 0 at calibration time. |
| 6 | Embedder serves context + unseen targets | PSN needs only `H` (already kept). Edge needs `R/V`: retained per recipe window from feats already computed (~1.6 MB/window, memory only); unseen target's `R_t/V_t` from its `normalize_window` output. `embed_rows(H, rv)`/`full_view` **raise** when `cfg.edge_stats` is truthy and `rv is None` — silent bias-off skew impossible by construction. |
| 7 | No re-ingestion | Planes derive from cached feats at runtime; `_cache_key` untouched. |
| 8 | Eval protocol unchanged | Same windows/metrics/append; `window_embeddings` (evaluate.py:77-85) and `_partition_agreement` (:204-213) build `rv` from the `feats_at` arrays already in hand; export inherits via `window_embeddings`; `eval_parallel` byte-identity preserved (deterministic per device). Eval wall ×~1.3–1.5; context↔context plane caching is a non-blocking optimization. |

Enumerated fail-loud call sites: inference.py:123, inference.py:130, evaluate.py:78, evaluate.py:85, evaluate.py:213, train.py:91.

## 6. Cost arithmetic (U≈4,848, 16 windows/step, 2×4090, `--context-checkpoint --loss-chunk 1024`)

**Params** (verified base: block 198,272; context 400,928; model 806,816):

| Piece | Count |
|---|---|
| v2 block: attn 66,048 + GELU FF 131,712 + 2×LN 512 + log_tau 8 | 198,280 |
| 3 blocks | 594,840 |
| edge MLP (2→8: 24; 8→2: 18) | 42 |
| PSN (LN 256 + P_q 2,064 + P_k 2,048 + W2 34) | 4,402 |
| ln_f 256 + out 4,128 | 4,384 |
| **Context** | **603,668** |
| **Model** (temporal 405,888 + context) | **1,009,556 (+202,740, +25.1%)** |

New-mechanism params: **4,468**, all logit-level ⇒ 100% relational-route. The only r10-shaped addition is block 3's FF (131,712), flagged and watched via the existing holdout-margin cadence.

**Step time**: depth 2→3 ×1.33 (measured); planes <1%; chunked bias MLPs (triple-computed under nested+outer checkpointing, memory-traffic-bound) +3–5%; float-mask/bias-grad handling +~5% ⇒ **×1.4–1.5 committed; ×~1.8 worst case under math-backend fallback at scale 1** (vs C0's ×1.8–2.0 baseline case).

**Memory** (per card, on the 10–13 GB envelope): held — block-3 ff-masks +~0.8 GB/card (+1.59 GB step total, verified), `R/V` ~25 MB. Scale-1 segment backward transient — planes 282 MB + bias 188 MB + mask_full 188 MB + mask_peer 188 MB + chunk recompute ≤300 MB ≈ **+1.2 GB** on the mem-efficient bias-grad branch; the math branch adds ~4.5 GB of saved softmax (6 relational calls). **Committed peak: ~13–15 GB (efficient) / ~17–20 GB (math)** — fits 24 GB on both branches; the startup probe (§8) reports which branch is live. Fallback ladder (engineering, in order): `edge_chunk` 512→256; fp16 planes+pair-hidden inside chunks with **fp32 bias** (SDPA mask dtype must equal query dtype — deductive); `loss_chunk` 1024→512; last resort `psn_rank` 16→8. Exact-path scales: expanded bias ≤56 MB — trivial.

## 7. Inference path + export/artifact compat

`Embedder.__init__`: alongside `H_ctx`, retain per (market, recipe window) `R_ctx [C,N]` fp32 + `V_ctx [C,N]` bool from the feats already computed (~13 MB total at K_inf=4 × 2 markets; **artifact files unchanged in format**). `embed_candles`: canonical path → planes over `[C]`; unseen-target path → prepend the target's `R_t/V_t` from `normalize_window`, planes over `[1+C]`; both call `embed_rows(H, rv)` (fail-loud). PSN/temperature recompute internally from `H` — zero retention. `weights.pt` gains `context.edge.*`, `context.psn.*`, `context.blocks.*.log_tau` automatically; `config.json` carries the fields via `asdict`. Old artifacts load and run bit-identically under new code (defaults ⇒ modules absent ⇒ strict key match); new artifacts refuse old code (expected). Embed cost: +3 bmms (~9 GFLOP/window) + one no-grad 188 MB bias per window — sub-second on GPU.

## 8. Run command + new log scalars

Write `StockIdentityModel/runs/v2rel3/config.json` = the r7 recipe + current data settings (parquet, cross_market, halt_markets, calendar flags as the active run) + `{"L_ctx": 3, "n_heads_ctx": 8, "attn_temp_ctx": true, "edge_stats": ["corr0","dlogvol"], "psn_rank": 16}` (architecture fields are config-file-only; the CLI exposes no `--set`). Then:

```
~/venvs/stockid/bin/python -m StockIdentityModel.train \
  --config StockIdentityModel/runs/v2rel3/config.json \
  --run-dir StockIdentityModel/runs/v2rel3 \
  --context-checkpoint --loss-chunk 1024 \
  --devices cuda:0,cuda:1 --eval-parallel
```

New log scalars (all no_grad, outside the segment, at `grad_diag_every`=250 cadence on the first US job's coarsest scale; one-time startup line `sdpa_ctx_backend` records efficient-vs-math dispatch for a grad-bearing float mask at the largest L and L+1):
1. `edge_bias_abs` — mean |capped bias| over real visible pairs (engagement meter; read against holdout margins = the r10 detector).
2. `psn_share` — RMS(PSN raw) / RMS(post-temperature bilinear logits) over the same pairs (channel attribution: summary-side vs data-side engagement; `edge_share` logged as its complement in the same dict).
3. `edge_sat_frac` — fraction of real pairs with |bias| > 0.9·edge_cap (cap-binding / saturation watch).

## 9. Risk register (risk → detection signal)

1. Math-backend dispatch at scale 1 (memory +4.5 GB, step →×1.8) → startup `sdpa_ctx_backend` line + per-scale segment timing in train_log; respond with the §6 fallback ladder.
2. r10-class memorization through stable pair statistics / PSN → `edge_bias_abs`/`psn_share` climbing while holdout margins bleed past ~10k steps; quantify with the exact W2-zero ablation at eval and re-run the H_o-perturbation probe (the 0.36 measurement) on the trained model.
3. `beta_prod` (only if opted back in — off the default run set): exact-path observer→peer-row weight channel (measured 4.5e-2 plane shift) + train/eval grouping shift (9-stock factor at fine scales vs market factor at eval scale 1) → channel engaged (`edge_share` up) but holdout margins flat or peer-route train/holdout scissoring; settle with per-channel ablation (zero its edge[0] input column) and the r_o-perturbation probe.
4. Zero-init engagement stall → both share scalars ≈0 at step 2k → σ=1e-3 re-init of the zero output layers.
5. Bias dominance / cap saturation → `edge_sat_frac` → 1 or `psn_share` ≫ 1; the tanh cap deductively bounds |bias| < 4, so the failure shape is a plateau, not divergence — retune `edge_cap` on the next run, never mid-run.