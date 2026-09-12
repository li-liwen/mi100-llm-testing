<!--
Milestone Report: GLM-5.3-Flash W4A16 AutoRound on 8x MI100 (gfx908) — Milestone 1
Date: 2026-09-12
Branch: li-liwen/vllm feat/glm53-flash-gfx908 @ e7f562cf2 (base 22258a26bc)
Stack: ROCm 7.2.4, torch 2.12.0+git6bbd260, triton 3.7.1+gitf0b55c07, transformers 5.16.1
Serving profile: scripts/serve_glm53_flash_m1.sh
Approved plan: docs/glm53_flash_m1_plan.md
-->

# GLM-5.3-Flash W4A16 AutoRound on 8× MI100 — Milestone 1 deployment report

**Model:** GLM-5.3-Flash W4A16 AutoRound (Glm5NextForConditionalGeneration, 355B MoE, sym INT4 GPTQ packing, group 128, 45 base layers: 34 KDA + 11 sparse MLA + 1 MTP layer, vision tower)

**Hardware:** 8× AMD Instinct MI100 (gfx908), XGMI hives 0–3 / 4–7, PCIe between hives

**Date:** September 12, 2026

**Status:** ✅ **Milestone 1 achieved** — the exact checkpoint loads correctly and serves with native MTP (depth 2) + HIP graphs on TP4×PP2: text, reasoning, tool calling, and image understanding all verified. c=1 pooled decode **38.8 tok/s** vs the ≥50 tok/s acceptance gate (Phase 6 continues). 1M context is measured-infeasible with BF16 KV on either topology; the FP8-cache fallback is the remaining path.

## Executive summary

This deployment ran the *exact* AutoRound W4A16 checkpoint (no requantization, no repacking) on an 8-card MI100 node through a fork of upstream vLLM (`feat/glm53-flash-gfx908`, 70 commits on top of main @ `22258a26bc`). Bring-up took four working sessions from a cold start to serving: the index-respecting loader defeated the checkpoint's stale intra-shard duplicate tensors; the INC/auto-round dispatch was re-routed from Marlin-only to Triton W4A16 on ROCm; the MTP draft was fixed to load its quantized experts, embedding and output head; and gfx908-specific gaps (fp8 MQA-logits kernels, oversized prefill workspaces, missing platform defaults) were closed.

Measured c=1 decode with HIP graphs and MTP: **code 46.6 / math 19.1 / prose 38.8, pooled median 38.8 tok/s** (plan gate: ≥50 pooled). Graph capture was worth 2.4× over eager; MTP depth 2 another ~30% over depth 1. Depth 3 boots and serves one request but then hits HSA memory-access faults on GPUs 5/6/8 — documented with a repro, root-causing pending.

The 1M-token context goal is **memory-bound, not kernel-bound**: measured KV/token/rank (~6.9 KiB on the six-sparse-layer PP0 rank) means TP4×PP2 would need ~35 GiB of KV per rank at 1M — impossible on 32 GiB cards. TP2×PP4 (partition 12,12,12,9) with BF16 KV caps at ~526k tokens on the tightest rank at 0.97 utilization; the plan's fallback #5 (explicit software FP8 MLA cache with its own quality gate) is the remaining path and is the headline item of Milestone 2.

## System configuration

| Component | Value |
|---|---|
| GPU(s) | 8× AMD Instinct MI100 (gfx908), ~31.984 GiB usable VRAM each |
| Host | 503 GiB RAM, 96 CPUs, 893 GiB free local disk |
| Kernel | 6.8.0-139-generic |
| ROCm | 7.2.4 |
| Interconnect | XGMI within GPUs 0–3 and 4–7; PCIe between hives (no P2P) |
| vLLM | li-liwen/vllm `feat/glm53-flash-gfx908` @ `e7f562cf2` (base `22258a26bc`), version `0.28.0rc7.dev0+glm53.gfx908` |
| Base image | `btbtyler09/vllm-rocm-gfx908:v0.28.0rc7.dev-q38fn` @ `sha256:03f325eb9fb40f21482972d30ade52ba1b47e2223dca66c71d1d3a057d0f0a67` |
| PyTorch / Triton / Transformers | 2.12.0+git6bbd260 / 3.7.1+gitf0b55c07 / 5.16.1 (pinned from the base image, untouched) |
| Checkpoint | `GLM-5.3-Flash-W4A16-AutoRound` (NFS original; verified local copy, all 113,074 indexed tensors, 169.010 GiB, sha256 manifest) |

## Launch command

```bash
#!/bin/bash
# GLM-5.3-Flash (Glm5Next, 355B MoE, W4A16 AutoRound GS128) on 8x MI100 (gfx908), TP4xPP2.
# Full profile with env rationale: scripts/serve_glm53_flash_m1.sh (in this repo).
# Requires: checkpoint at $MODEL, VLLM_API_KEY set.
set -euo pipefail
IMG=${IMG:-glm53f:latest}
MODEL=${MODEL:-/mnt/flash-inference/models/GLM-5.3-Flash-W4A16-AutoRound}
NAME=vllm-glm53f
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
  -p 8006:8000 -v "$MODEL":/model:ro \
  "$IMG" vllm serve /model --served-model-name glm5.3-flash-autoround \
    --tensor-parallel-size 4 --pipeline-parallel-size 2 --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.93 --max-num-seqs 4 \
    --max-num-batched-tokens 2048 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --load-format safetensors \
    --model-loader-extra-config '{"safetensors_use_index": true}' \
    --trust-remote-code
echo "started; ~14 min to serving (169 GB indexed load + Triton JIT warmup + graph capture)"
```

The image is not on Docker Hub (fork-local); rebuild with the vLLM branch's
`deploy/glm53-gfx908/scripts/build.sh` — it pins the base image by digest, builds the
wheel with `PYTORCH_ROCM_ARCH=gfx908`, and installs only `tilelang`/`apache-tvm-ffi`
on top of the base (everything else already satisfies the requirements; torch/triton/
transformers are installed `--no-deps` and never upgraded). The build purges the base
image's stale vllm package before and after the editable install — its leftover
namespace stubs shadow the editable finder and break submodule imports.

## Deployment results

### Loading correctness (the checkpoint's duplicate-tensor trap)

The AutoRound repair added two extra safetensors files (`model_extra_conv.safetensors`,
`model_extra_tensors.safetensors`, 2,719 tensors) whose copies supersede stale tensors
*inside* the regular shards — e.g. conv1d weights stored twice as `[24576,1,4]` stale
and `[8192,1,4]` repaired. File-level dedup cannot fix intra-file duplicates, so an
index-respecting iterator was added (`--model-loader-extra-config
'{"safetensors_use_index": true}'`): each tensor is yielded exactly once from its
`weight_map` file, unindexed extras are ignored, missing files/tensors raise. All
113,074 indexed tensors load exactly once; 8 loader unit tests pass.

Quantization conventions verified against the real checkpoint tensors: `qzeros` words
are `0x77777777` (nibble 7), which under GPTQ's offset convention is zero point 8 —
exactly the `uint4b8` bias the TritonW4A16 kernel applies (dequant `w = (q-8)·scale`).
No `g_idx` anywhere (non-activation-ordered). Exclusions verified per-layer: routed
experts quantize to 4-bit while shared experts, conv1d, router gate and dense
down_proj stay BF16 (tests pass against real checkpoint tensors).

### Serving milestones (bring-up order, one session each)

| Step | Result |
|---|---|
| Hardware probes (compute, RCCL collectives, graphs) | All pass; cross-hive P2P absent — direct device copies segfault, so PP uses RCCL send/recv (verified working) |
| Full model boot, TP4×PP2, 8K, eager | 8/8 workers: 20–21.7 GiB weights/rank, ~200 s load, TRITON WNA16 MoE backend auto-selected (Marlin rejected) |
| Native MTP depth 1 | Draft layer 45 loads 28/30 quantized params per rank (2 untouched qzeros are expected — symmetric uint4b8 ignores them); draft embed + output head loaded from the checkpoint; coherent outputs |
| HIP graphs (no `--enforce-eager`) | Graph capture passes on all 8 workers with MTP; decode 2.4× vs eager |
| Depth 2 | +30% over depth 1 — **current production depth** |
| Depth 3 | Boots and serves one 200-OK request, then HSA memory-access faults on GPUs 5/6/8 kill the engine (repro: `scripts/serve_glm53_flash_m1_d3.sh` equivalent — see Known issues) |
| Multimodal | Image understanding verified (synthetic red/green PNG correctly described through the GLM vision tower); video timestamp fix from vLLM PR #55647 ported (placeholder/pixel sampler agreement) |
| Parsers | `glm45` reasoning parser verified live (reasoning/content split); `glm47` tool parser wired and the model emits well-formed native tool calls |

### Decode performance (C1 benchmark, plan §4 shape: 3 prompt families × ~4k input tokens, 1024 forced output tokens, 1 warmup + 5 reps, median; streaming chars/3.2 as the tok/s estimate)

| Config | code | math | prose | **pooled median** |
|---|---:|---:|---:|---:|
| MTP depth 1, eager | — | — | — | ~13 |
| MTP depth 1, graphs | ~32 | ~14 | ~30 | ~30 |
| **MTP depth 2, graphs** | **46.6** | **19.1** | **38.8** | **38.8** |

**Gate: ≥50 pooled — not yet met (38.8).** The math family (deep reasoning chains) is
the pooled blocker: its MTP acceptance is visibly lower than code/prose. The remaining
Phase 6 levers, in the order the plan prescribes: dense W4A16 GEMV at M ≤ 8 (the MoE
GEMV port is in and verified but perf-neutral at depth 2 — the dense q_b/kv_b/o_proj
GEMV is the more promising port), skinny-BF16 dense path, split-KV sparse decode,
depth-3 fault root-cause (unlocking depth 3 raises acceptance headroom).

### Correctness spot checks (plan §4 model-quality subset)

17×23=391; 15% of 80 = 12; 3x=27 → 9; 25%-discount pricing → 30; coherent Chinese
instructions with correct technical terms; coherent quantum-computing summary;
image colors correctly identified. Full GSM8K subset + logprob-based
optimized-vs-eager comparison are Phase 4 acceptance work (Milestone 2).

## Engineering work landed on the fork (70 commits on `feat/glm53-flash-gfx908`)

**Platform/gfx908 enablement** (ports from `btbtyler09/vllm-gfx908@mi100-optimized`
d3bab5eb0 / 2ae323c98 / d2e868f3, adapted): `_ON_GFX908` arch flag + `on_gfx908()`;
per-feature AITER env defaults (CK ops off — they crash on gfx908; Triton paths on);
CK→Triton redirects for rms_norm/rmsnorm2d/flash_attn_varlen; NCCL Tree+LL defaults;
`DISABLE_ADDMM_HIP_LT=1` (ROCm 7.2.4 hipBLASLt heuristics slow MTP verify GEMMs);
unified-attention default OFF (state-corruption reports in the reference's later
defaults supersede the initial ON).

**Loader**: index-respecting safetensors iteration (above); extra-config allowlist;
Dockerfile purges the base image's stale vllm package before *and* after the editable
install (its leftover namespace stubs shadow the editable finder and break submodule
imports).

**GLM model** (ports from `promisezackr/glm53-flash-170hx-pp8` 0003/0005/0017 +
vLLM PR #55647, adapted to main): PP intermediate-tensor factories with mHC `hc_post`
materialization at stage boundaries; MTP draft embedding/output-head loading from the
checkpoint (under PP>1 the runner's target-weight sharing is skipped, which previously
left the draft random-init at 0% acceptance); shared head built unquantized matching
the BF16 `lm_head`; untouched-parameter warning after draft load; video placeholder
timestamps taken from the pixel path's sampler.

**Quantization dispatch**: `AutoGPTQLinearMethod` Marlin verification deferred until
after kernel selection so `TritonW4A16LinearKernel` is reachable on ROCm; INC
Marlin-less fallback to the kernel-choosing method; INC parser root remap so the MTP
draft's shortened `model.layers.45.*` parameter paths resolve against the checkpoint's
quantize block (the draft routed experts previously resolved unquantized and draft
loading failed).

**gfx908 kernels**: W4A16 GEMV MoE path for M ≤ 8 (port of `cfac8d0d9`; numerics
verified, split-count divisibility fix for K=4096, perf-neutral at depth 2 — the
dense-projection GEMV is the more promising follow-up); vectorized chunked
capture-safe paged MQA-logits fallback (aiter's Triton kernels need fp8 `tl.dot`,
unsupported on gfx908 — exact vs reference on GPU); MLA chunked-prefill workspace
capped at `max_num_batched_tokens` (the 64k-token cap reserved multiple GiB per
sparse-MLA layer at 1M and starved KV).

## Known issues and negative results

1. **MTP depth 3 → HSA memory-access faults** after serving one 200-OK request
   ("Page not present or supervisor privilege", GPUs 5/6/8). Repro: boot with
   `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` + any chat
   request. Depth 2 is the working production depth; root-cause pending (suspect a
   kpool/MTP-shape kernel at depth-3 shapes).
2. **Cross-hive direct device copies segfault** (no P2P across the PCIe-separated
   hives; `can_device_access_peer(0,4)=False`; HSA SVM/SDMA/TRANSFER knobs don't
   help). vLLM's PP transfers use RCCL send/recv, which works cross-hive — verified
   with an 8-GPU all-reduce + send/recv probe.
3. **aiter Triton MQA-logits kernels are gfx908-incompatible** (fp8 `tl.dot` →
   "Unsupported lhs dtype fp8e4nv"). Replaced by the vectorized torch fallback on
   gfx908; a native dequantizing Triton kernel is Phase 6 work.
4. **1M context, TP4×PP2: impossible with BF16 KV** — measured ~6.9 KiB KV/token/rank
   on the six-sparse-layer PP0 rank → ~35 GiB/rank needed at 1M vs 32 GiB total.
5. **1M context, TP2×PP4 (partition 12,12,12,9): BF16 KV caps at ~526k tokens** —
   the tightest rank has 1.62 GiB KV available at util 0.97 vs 3.16 GiB needed for
   1M. The plan's fallback #5 (explicit software FP8 MLA cache with its own quality
   gate) is the remaining path.
6. The base image's `prebuild_gfx908_exts.py` reports failures in this image —
   expected: those extension modules are Qwen4-fork-only and absent from the GLM
   branch.
7. The W4A16 GEMV MoE path is perf-neutral at depth 2 (pooled 38.8 with vs 39.5
   without, within run noise); kept enabled since numerics-verified. The reference
   repo's gains were on a smaller-expert model.

## Next steps (Milestone 2 candidates, per the approved plan)

1. Phase 6: dense W4A16 GEMV at M ≤ 8 (q_b/kv_b/o_proj) + skinny-BF16 dense path;
   math-family acceptance investigation; depth-3 root-cause; re-run C1 after each.
2. Phase 5: FP8 MLA-cache fallback for 1M (TP2×PP4) with its own quality gate; then
   the 8K→…→1M long-context ladder.
3. Phase 5: video caps (32 frames / 4096 vision tokens) + mixed-modality tests.
4. Phase 4: 200-question GSM8K subset + logprob-based optimized-vs-eager comparison;
   forced-MTP-rejection state tests; 1000-request/2h soak.
5. Phase 7: immutable image, restart policy, kernel-cache persistence, handoff record.
