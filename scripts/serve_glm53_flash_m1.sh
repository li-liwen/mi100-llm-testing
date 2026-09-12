#!/bin/bash
# GLM-5.3-Flash (Glm5Next, 355B MoE, W4A16 AutoRound GS128) on 8x MI100 (gfx908), TP4xPP2.
# Image glm53f:latest = li-liwen/vllm feat/glm53-flash-gfx908 @ e7f562cf2 built on the pinned
# runtime btbtyler09/vllm-rocm-gfx908:v0.28.0rc7.dev-q38fn
# @ sha256:03f325eb9fb40f21482972d30ade52ba1b47e2223dca66c71d1d3a057d0f0a67
# (ROCm 7.2.4, torch 2.12.0+git6bbd260, triton 3.7.1+gitf0b55c07, transformers 5.16.1).
#
# Validated env (gfx908 defaults baked into the fork's rocm.py auto-config, kept explicit here):
#   HSA_ENABLE_SVM=0, HSA_NO_SCRATCH_RECLAIM=1, HIP_FORCE_DEV_KERNARG=1 (DSV4-stability carryovers),
#   TORCH_BLAS_PREFER_HIPBLASLT=0 (rocBLAS addmm fallback; hipBLASLt 7.2.4 heuristics slow MTP verify),
#   VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 (UA state-corruption reports on gfx908; Triton attention).
# Loading: --load-format safetensors with '{"safetensors_use_index": true}' — the checkpoint holds
#   stale intra-shard duplicates (e.g. [24576,1,4] conv1d copies) while the index maps each tensor
#   to its repaired copy; the index-respecting iterator yields exactly the index tensors.
# MTP depth 2: the draft (checkpoint layer 45) loads 28/30 quantized params; the 2 untouched
#   qzeros are expected (symmetric uint4b8 ignores them). Depth 3 hits HSA memory-access faults.
# Usage: MODEL=/path/to/GLM-5.3-Flash-W4A16-AutoRound scripts/serve_glm53_flash_m1.sh [extra vllm args]
set -euo pipefail
IMG=${IMG:-glm53f:latest}
MODEL=${MODEL:-/mnt/flash-inference/models/GLM-5.3-Flash-W4A16-AutoRound}
NAME=${NAME:-vllm-glm53f}
API_KEY=${VLLM_API_KEY:?set VLLM_API_KEY in env}
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --ipc=host --cpuset-cpus=0-23 --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd \
  --device=/dev/dri/renderD128 --device=/dev/dri/renderD129 --device=/dev/dri/renderD130 \
  --device=/dev/dri/renderD131 --device=/dev/dri/renderD132 --device=/dev/dri/renderD133 \
  --device=/dev/dri/renderD134 --device=/dev/dri/renderD135 \
  -e HSA_OVERRIDE_GFX_VERSION=9.0.8 -e HSA_ENABLE_SVM=0 -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HIP_FORCE_DEV_KERNARG=1 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_API_KEY="$API_KEY" -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -p 8006:8000 \
  -v "$MODEL":/model:ro \
  "$IMG" vllm serve /model --served-model-name glm5.3-flash-autoround \
    --tensor-parallel-size 4 --pipeline-parallel-size 2 --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.93 --max-num-seqs 4 \
    --max-num-batched-tokens 2048 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --load-format safetensors \
    --model-loader-extra-config '{"safetensors_use_index": true}' \
    --trust-remote-code "$@"
echo "started $NAME ($IMG); wait: until curl -sf localhost:8006/health; do sleep 20; done  (~14 min: 169 GB indexed load + Triton JIT warmup + graph capture)"
