# DeepSeek-V4.1-Flash on GB300: optimization branch

Branch `dsv41-flash-gb300-opt` of `github.com/nvzhihanj/vllm` is CentML vLLM
`mlperf-end-multiturn-v1.0-dsv4.1-flash` at commit
`cbc89c23e101e36a441c8f0692f673993d2ee0a3`, plus one commit per optimization for
serving DeepSeek-V4.1-Flash on GB300 NVL72 (one 4-GPU node) behind NVIDIA Dynamo
in the MLPerf Endpoints agentic benchmark. Every commit changes Python or Triton
code only; the C++/CUDA sources, and the kernels vLLM builds from them, are those
of `cbc89c23`.

The serving recipe (Slurm + nv-sflow), the container build, and the measured
results are in the `mlperf-endpoints` repository under
`NVIDIA/src/deepseek-v4.1-flash/` (README.md and BUILD.md).

## Commits

All new behavior is off unless its environment variable is set, except the two
kernel changes, which produce identical results.

| Commit | Change | Enable | Effect (DEP4, C2048 unless noted) |
| --- | --- | --- | --- |
| `85dd986b` | Inverse-RoPE + MXFP8 quant kernel: 4 heads x 2 warps per program instead of one warp per (token, head). Bitwise-identical output. | always on (`VLLM_INV_ROPE_HEADS_PER_PROG`, `VLLM_INV_ROPE_NUM_WARPS` override) | kernel 2.2x; +1.4% end to end (2xTP2 C1024) |
| `dc7a70ee` | `VLLM_DP_RANK_LOCAL_OVERRIDE`: co-located external-LB DP ranks each select their own GPU while all node GPUs stay visible (torch symmetric memory for DeepGEMM MegaMoE rejects every rank being device 0). | `VLLM_DP_RANK_LOCAL_OVERRIDE=<rank>` per rank | enables DEP4 behind Dynamo (one dynamo.vllm per DP rank) |
| `9e2fe8d8` | Candidate-block (CED) selection bounds block scores and top-k by the batch's longest row instead of `max_model_len`. Same selection. | always on | -2.9 ms per decode call on the candidate-source layer |
| `5acada2a` | Decoder SWA bounded replay (DeepSeek-V4.1 tech report 2.2 / 3.2.2) also in piecewise CUDA-graph steps: layers past the last KV-source layer run on each prefill's last window rows only, via a separately captured replay graph. | `VLLM_DSV41_GRAPH_BOUNDED_REPLAY=1` | +15% |
| `9dd28724` | Engine core defers full (oldest-generation) Python GC collections to idle; young-generation collection is unchanged. | `VLLM_ENGINE_DEFER_FULL_GC=1` | removes 0.5-0.6 s scheduler pauses every 15-20 s; +9.9% |
| `c78e748f` | Block-verification residual-mass kernel bounds its next-draft read (one-element out-of-bounds read; intermittent illegal memory access after 10-60 min with adaptive verification). Results unchanged. | always on | stability |
| `40bed00e` | Engram tables sharded across DP ranks return their rows with an all-to-all (pynccl grouped send/recv, CUDA-graph capturable) instead of an all-gather. Bitwise identical. | `VLLM_DSV41_ENGRAM_ALL_TO_ALL=1` | -2.0 ms/step |

The GB300 points set all three variables:

```bash
export VLLM_DSV41_GRAPH_BOUNDED_REPLAY=1
export VLLM_ENGINE_DEFER_FULL_GC=1
export VLLM_DSV41_ENGRAM_ALL_TO_ALL=1
```

Server topology and flags (DEP4: four `dynamo.vllm` processes, one per DP rank,
EP4 DeepGEMM MegaMoE, FlashMLA mega attention with the `fp8_ds_mla` KV cache,
Engram tables sharded across the DP ranks on transparent huge pages, per-point
`--max-num-batched-tokens`) are in the mlperf-endpoints point configurations.

## Dependencies

No dependency is modified. The image is built from this branch with vLLM's own
`docker/Dockerfile` (target `vllm-openai`), which compiles the kernels vLLM pins
in `cmake/external_projects/` (DeepGEMM `vllm-project/DeepGEMM@e1f418c2`,
FlashMLA `vllm-project/FlashMLA@0eee43b1`, vllm-flash-attention, MSA, DeepSelect,
FlashKDA, qutlass, CUTLASS) and installs `requirements/rubin-prerelease.txt`
(PyTorch nightly `2.15.0.dev20260908+cu134`, FlashInfer `0.7.0.post1` with its
cu134 JIT cache, `nvidia-cutlass-dsl 4.8.0.dev0`, `nixl 1.4.1`), Triton built from
`triton-lang/triton@3f6e4113`, and DeepEP `deepseek-ai/DeepEP@d4f41e4e`. All of
them are public.

## Building the image

The canonical build is `mlperf-endpoints/NVIDIA/src/deepseek-v4.1-flash/BUILD.md`
(build arguments in `configs/vllm-build-args.txt`, revision in
`configs/dependencies.json`). With Docker on a native arm64 host it amounts to:

```bash
git clone --branch dsv41-flash-gb300-opt https://github.com/nvzhihanj/vllm.git && cd vllm
mapfile -t args < <(grep -vE '^[[:space:]]*(#|$)' <mlperf-endpoints>/NVIDIA/src/deepseek-v4.1-flash/configs/vllm-build-args.txt)
docker buildx build --platform linux/arm64 --target vllm-openai \
  "${args[@]/#/--build-arg=}" --build-arg "VLLM_BUILD_COMMIT=$(git rev-parse HEAD)" \
  --tag <registry>/<image>-vllm -f docker/Dockerfile .
docker buildx build --platform linux/arm64 \
  --build-arg VLLM_IMAGE=<registry>/<image>-vllm --build-arg DYNAMO_VERSION=1.6.0.dev20260924 \
  --tag <registry>/<image> <mlperf-endpoints>/NVIDIA/src/deepseek-v4.1-flash/configs
```

## Developing on top of this branch

Changes that touch only Python or Triton code can be tested without rebuilding:
check out the branch with your commits, put the checkout first on `PYTHONPATH`
in the workers (the mlperf-endpoints workflow's `VLLM_OVERLAY` variable does
this), and copy the image's compiled extensions into the checkout once
(files the installed package has and the tree lacks: `vllm/*.so`,
`vllm/_version.py`, `vllm/third_party/...`). Rebuild the image for C++/CUDA
changes and for any result you publish.
