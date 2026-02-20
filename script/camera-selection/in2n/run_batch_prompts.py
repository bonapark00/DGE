#!/usr/bin/env python3
"""
Run launch_and_metrics for multiple prompt combinations in parallel across GPUs.
Each task runs NUM_RUNS=2 (launch + metrics, twice). One task per GPU.
Starts processes with 10s interval to avoid cache/disk collision.
"""

import time
import sys
from multiprocessing import Process

import launch_and_metrics as lm

# ========== 사용할 GPU ==========
GPU_IDS = [0, 1, 2]
# ========== 프로세스 시작 간격 (초) - 캐시 충돌 방지 ==========
START_INTERVAL = 10
# ========== 프롬프트 조합 (각각 NUM_RUNS=2 회 실행) ==========
TASKS = [
    {"name": "smile", "PROMPT": "Make his mouth smile", "SEG_PROMPT": "face of the man",
     "TARGET_PROMPT": "Smiling mouth of the man", "STYLE_TARGET_PROMPT": "A man without smiling",
     "STYLE_SOURCE_PROMPT": "A Man with smiling"},
    {"name": "mustache", "PROMPT": "Give him a mustache", "SEG_PROMPT": "face of the man",
     "TARGET_PROMPT": "Mustache of the man", "STYLE_TARGET_PROMPT": "A man without mustache",
     "STYLE_SOURCE_PROMPT": "A Man with mustache"},
    {"name": "leather_jacket", "PROMPT": "Change the fleece jacket into a leather jacket",
     "SEG_PROMPT": "fleece jacket", "TARGET_PROMPT": "Leather jacket",
     "STYLE_TARGET_PROMPT": "A man with a leather jacket", "STYLE_SOURCE_PROMPT": "A man with a fleece jacket"},
    {"name": "mondigliani", "PROMPT": "Turn him into a Mondigliani painting", "SEG_PROMPT": "man",
     "TARGET_PROMPT": "man", "STYLE_TARGET_PROMPT": "A real photo of a man",
     "STYLE_SOURCE_PROMPT": "A Mondigliani painting of a man"},
    {"name": "pixar", "PROMPT": "Turn the man into a stylized 3D Pixar-like character.", "SEG_PROMPT": "man",
     "TARGET_PROMPT": "man", "STYLE_TARGET_PROMPT": "A real photo of a man",
     "STYLE_SOURCE_PROMPT": "A 3D Pixar-like man"},
    {"name": "clown", "PROMPT": "Turn the man into a clown.", "SEG_PROMPT": "man",
     "TARGET_PROMPT": "clown", "STYLE_TARGET_PROMPT": "man", "STYLE_SOURCE_PROMPT": "clown"},
]
NUM_RUNS_PER_TASK = 2


def _worker(task: dict, gpu_id: int) -> int:
    import launch_and_metrics as _lm
    _lm.GPU = str(gpu_id)
    _lm.NUM_RUNS = NUM_RUNS_PER_TASK
    _lm.PROMPT = task["PROMPT"]
    _lm.SEG_PROMPT = task["SEG_PROMPT"]
    _lm.MMR_SEG_PROMPT = task["SEG_PROMPT"]
    _lm.TARGET_PROMPT = task["TARGET_PROMPT"]
    _lm.STYLE_TARGET_PROMPT = task.get("STYLE_TARGET_PROMPT", "")
    _lm.STYLE_SOURCE_PROMPT = task.get("STYLE_SOURCE_PROMPT", "a Photo")
    name = task.get("name", task["PROMPT"][:30])
    print(f"[GPU {gpu_id}] Starting task: {name}", flush=True)
    code = _lm.run_batch()
    print(f"[GPU {gpu_id}] Task {name} finished with code {code}", flush=True)
    return code


def main():
    if not TASKS or not GPU_IDS:
        print("TASKS or GPU_IDS is empty.")
        return 1
    print("=" * 60)
    print("Batch run: multiple prompts, NUM_RUNS=2 each, multi-GPU")
    print("=" * 60)
    print(f"GPU_IDS: {GPU_IDS}")
    print(f"START_INTERVAL: {START_INTERVAL}s between process starts")
    print(f"Tasks: {len(TASKS)}")
    print("=" * 60)

    pending = list(enumerate(TASKS))
    running = {}
    exit_codes = []

    def start_next_on_gpu(gpu_id: int) -> bool:
        if not pending:
            return False
        idx, task = pending.pop(0)
        name = task.get("name", task["PROMPT"][:30])
        p = Process(target=_worker, args=(task, gpu_id))
        p.start()
        running[gpu_id] = (p, name)
        print(f"[Scheduler] Started '{name}' on GPU {gpu_id} ({len(pending)} pending)", flush=True)
        return True

    for gpu_id in GPU_IDS:
        start_next_on_gpu(gpu_id)
        if not pending:
            break
        time.sleep(START_INTERVAL)

    while running:
        for gpu_id in list(running.keys()):
            p, name = running[gpu_id]
            if not p.is_alive():
                code = p.exitcode if p.exitcode is not None else -1
                exit_codes.append((name, code))
                del running[gpu_id]
                time.sleep(START_INTERVAL)
                if not start_next_on_gpu(gpu_id):
                    pass
        time.sleep(2)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, code in exit_codes:
        status = "OK" if code == 0 else "FAILED"
        print(f"  {name}: {status} (exit {code})")
    failed = sum(1 for _, c in exit_codes if c != 0)
    print("=" * 60)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
