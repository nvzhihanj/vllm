# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DEBUG ONLY: wrap selected worker / engine-core functions in NVTX ranges (VLLM_DEBUG_NVTX=1).

nsys then shows the host-side phases of every step (DP sync, input / attention / replay prep, forward launch,
sampling, output wait, scheduling) on the CPU timeline, aligned with the GPU kernels.
"""
import functools
import importlib
import os

import torch

WORKER_TARGETS = [
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.execute_model",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.sample_tokens",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.prepare_inputs",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.add_requests",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.update_requests",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.gather_batch_req_state",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.sample",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.postprocess_sampled",
    "vllm.v1.worker.gpu.model_runner:dispatch_cg_and_sync_dp",
    "vllm.models.deepseek_v41.nvidia.model_state:DeepseekV41ModelState.prepare_inputs",
    "vllm.models.deepseek_v41.nvidia.model_state:DeepseekV41ModelState.prepare_attn",
    "vllm.models.deepseek_v41.nvidia.model_state:DeepseekV41ModelState._prepare_replay_batch",
    "vllm.v1.worker.gpu.cudagraph_utils:CudaGraphManager.run_pw_graph",
    "vllm.v1.worker.gpu.cudagraph_utils:CudaGraphManager.run_fullgraph",
    "vllm.v1.worker.gpu.cudagraph_utils:ModelCudaGraphManager.run_fullgraph",
    "vllm.v1.worker.gpu.async_utils:AsyncOutput.get_output",
    "vllm.distributed.device_communicators.shm_broadcast:MessageQueue.dequeue",
]
ENGINE_TARGETS = [
    "vllm.v1.core.sched.scheduler:Scheduler.schedule",
    "vllm.v1.core.sched.scheduler:Scheduler.update_from_output",
    "vllm.v1.engine.core:EngineCore.step_with_batch_queue",
    "vllm.v1.executor.multiproc_executor:FutureWrapper.result",
    "vllm.distributed.device_communicators.shm_broadcast:MessageQueue.dequeue",
]


def _wrap(fn, name):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        torch.cuda.nvtx.range_push(name)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    wrapper.__nvtx_debug__ = True
    return wrapper


def install(role: str) -> None:
    if os.environ.get("VLLM_DEBUG_NVTX") != "1":
        return
    done = []
    for target in WORKER_TARGETS if role == "worker" else ENGINE_TARGETS:
        mod_name, qual = target.split(":")
        try:
            obj = importlib.import_module(mod_name)
            parts = qual.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            fn = getattr(obj, parts[-1])
        except Exception:  # noqa: BLE001
            continue
        if getattr(fn, "__nvtx_debug__", False):
            continue
        setattr(obj, parts[-1], _wrap(fn, f"{role}:{qual}"))
        done.append(qual)
    print(f"[nvtx_debug] {role}: wrapped {len(done)} targets: {done}", flush=True)
