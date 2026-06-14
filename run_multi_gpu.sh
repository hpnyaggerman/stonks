#!/usr/bin/env bash
# Train a v5 deep ensemble across every visible GPU in parallel — MEMBERS_PER_GPU
# members per card — then run one finalize pass that writes temperatures, metadata,
# and forecasts over the full ensemble.
#
#   bash run_multi_gpu.sh <members_per_gpu> [extra run_train_v5.py args...]
#
# Examples:
#   bash run_multi_gpu.sh 2                       # 2 members per GPU
#   bash run_multi_gpu.sh 2 --seed 7 --batch-size 512 --lr 4.2e-4 \
#       --max-steps 100000 --t-max 100000        # extra args apply to every process
#
# GPUs: all that nvidia-smi reports, or the set in CUDA_VISIBLE_DEVICES if you export
# it before calling. Total members = (#GPUs) * members_per_gpu.
#
# Do NOT pass these via the extra args — the script manages them:
#   --members --member-start --member-count --cache-dir --skip-forecast --forecast-only --device
set -euo pipefail
cd "$(dirname "$0")"

PER_GPU="${1:?usage: run_multi_gpu.sh <members_per_gpu> [extra run_train_v5.py args...]}"
shift || true
EXTRA=("$@")

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
# Shared across every process so the scaler / splits / val-fold are identical and the
# ensemble is valid. Override anything here by appending to the script's extra args.
COMMON=(--members "$MEMBERS" --eval-mode time --seed 42 --batch-size 256 --num-workers 8
        --epochs 3 --eval-every-steps 2000 --patience 20 --val-subsample 150000
        --log-every 100 --device cuda)

echo "[multi-gpu] GPUs=${GPU_IDS[*]} | members/GPU=$PER_GPU | total members=$MEMBERS"
echo "[multi-gpu] args: ${COMMON[*]} ${EXTRA[*]+${EXTRA[*]}}"

pids=()
for g in "${!GPU_IDS[@]}"; do
    start=$(( g * PER_GPU ))
    log="train_gpu${g}.log"
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

echo "[multi-gpu] all $MEMBERS members trained; finalizing (temperatures + artifacts + forecasts)"
python run_train_v5.py "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} --forecast-only 2>&1 | tee finalize.log
echo "[multi-gpu] done"
