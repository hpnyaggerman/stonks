#!/usr/bin/env bash
# Train a v5 deep ensemble across every visible GPU in parallel -- MEMBERS_PER_GPU
# members per card -- then run one finalize pass that writes temperatures, metadata,
# and forecasts over the full ensemble.
#
#   bash run_multi_gpu.sh <members_per_gpu> [extra run_train_v5.py args...]
#
# Examples:
#   bash run_multi_gpu.sh 2                       # 2 members per GPU
#   bash run_multi_gpu.sh 1 --eval-mode both --ticker-holdout-frac 0.1 \
#       --seed 7 --batch-size 512                # extra args apply to every process
#
# GPUs: all that nvidia-smi reports, or the set in CUDA_VISIBLE_DEVICES if you export
# it before calling. Total members = (#GPUs) * members_per_gpu.
#
# Do NOT pass these via the extra args -- the script manages them:
#   --members --member-start --member-count --cache-dir --skip-forecast --forecast-only --device
set -euo pipefail
cd "$(dirname "$0")"

PER_GPU="${1:?usage: run_multi_gpu.sh <members_per_gpu> [extra run_train_v5.py args...]}"
shift || true
EXTRA=("$@")

# Numbered side-by-side run dirs under models/v5/runs (stock-identity convention):
# every launch gets its own dir -- config, manifest, histories, checkpoints, logs
# together -- and no run ever overwrites a previous one. An explicit --run-dir
# (protocol runs nullc/ro1..ro3, smoke) wins; otherwise claim the next rN ONCE
# here and pass it to every trainer and the finalize, because per-process
# auto-claiming would scatter the members across different run dirs.
RUN_DIR=""
for ((i = 0; i < ${#EXTRA[@]}; i++)); do
    case "${EXTRA[$i]}" in
        --run-dir)   RUN_DIR="${EXTRA[$((i + 1))]:-}" ;;
        --run-dir=*) RUN_DIR="${EXTRA[$i]#--run-dir=}" ;;
    esac
done
if [ -z "$RUN_DIR" ]; then
    RUN_DIR="$(python -c 'import run_train_v5 as t; print(t.next_run_dir().relative_to(t.PROJECT_ROOT))')"
    EXTRA+=(--run-dir "$RUN_DIR")
fi
mkdir -p "$RUN_DIR"
# The launcher's own stream lands in the run dir too, so a detached launch
# (screen/nohup) needs no external redirect and the run dir is self-contained.
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
echo "[multi-gpu] run dir: $RUN_DIR"

# Resolve the GPU id list: an explicit CUDA_VISIBLE_DEVICES wins; otherwise use all.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
    echo "[multi-gpu] no GPUs found (set CUDA_VISIBLE_DEVICES or install nvidia-smi)" >&2
    exit 1
fi
NGPU=${#GPU_IDS[@]}
[ "$NGPU" -ge 1 ] || { echo "[multi-gpu] no GPUs resolved" >&2; exit 1; }
MEMBERS=$(( NGPU * PER_GPU ))

export PYTHONUNBUFFERED=1   # unbuffered so per-GPU logs stream live, not in 8 KB chunks
# One nonce per launch: the member-0 trainer embeds it in the run manifest and every
# other trainer requires an exact match, so a stale manifest from a previous launch
# can never pass the reader check.
export RUN_NONCE="$(date +%s)-$$"
# Shared across every process so the scaler / splits / val-fold are identical and the
# ensemble is valid. Eval cadence is the trainer's built-in schedule (no fixed
# --eval-every-steps); eval mode comes from the caller's extra args.
COMMON=(--members "$MEMBERS" --seed 42 --batch-size 256 --num-workers 8
        --epochs 3 --patience 20 --val-subsample 150000
        --log-every 100 --device cuda)

echo "[multi-gpu] GPUs=${GPU_IDS[*]} | members/GPU=$PER_GPU | total members=$MEMBERS | nonce=$RUN_NONCE"
echo "[multi-gpu] args: ${COMMON[*]} ${EXTRA[*]+${EXTRA[*]}}"

pids=()
for g in "${!GPU_IDS[@]}"; do
    start=$(( g * PER_GPU ))
    log="${RUN_DIR}/train_gpu${g}.log"
    echo "[multi-gpu] phys GPU ${GPU_IDS[$g]} -> members [$start, $((start + PER_GPU))) -> $log"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$g]}" python run_train_v5.py \
        "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
        --member-start "$start" --member-count "$PER_GPU" \
        --cache-dir "cache/v5_gpu${g}" --skip-forecast > "$log" 2>&1 &
    pids+=("$!")
done

# Wait for every trainer; only finalize if all succeeded (else a member is missing).
fail=0
for g in "${!pids[@]}"; do
    if ! wait "${pids[$g]}"; then
        echo "[multi-gpu] trainer for GPU index $g FAILED (see train_gpu${g}.log)" >&2
        fail=1
    fi
done
[ "$fail" -eq 0 ] || { echo "[multi-gpu] a trainer failed; skipping finalize" >&2; exit 1; }

# Finalize reuses GPU0's cache dir (the memmap opens mode=w+, an in-place overwrite,
# not a fast reuse) and deletes the other per-GPU copies first -- they are identical
# by construction, and the deletion is what actually frees the disk the crashed
# run-2 finalize ran out of.
for g in "${!GPU_IDS[@]}"; do
    if [ "$g" -ne 0 ]; then
        rm -rf "cache/v5_gpu${g}"
    fi
done
echo "[multi-gpu] all $MEMBERS members trained; finalizing (temperatures + artifacts + forecasts)"
FINALIZE_CMD=(python run_train_v5.py "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
    --cache-dir cache/v5_gpu0 --forecast-only)
echo "[multi-gpu] finalize command: ${FINALIZE_CMD[*]}" | tee "$RUN_DIR/finalize.log"
"${FINALIZE_CMD[@]}" 2>&1 | tee -a "$RUN_DIR/finalize.log"
echo "[multi-gpu] done"
