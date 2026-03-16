# gpu_variable_load.py
import torch
import random
import time
import os
import signal
import sys

running = True
buffers = []

def stop(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

device = "cuda"
dtype = torch.float16

def mem_gb():
    return torch.cuda.memory_reserved() / (1024**3)

def alloc(mb):
    n = (mb * 1024 * 1024) // 2
    t = torch.empty(n, device=device, dtype=dtype)
    buffers.append(t)

def free_some():
    if buffers:
        buffers.pop(random.randrange(len(buffers)))
        torch.cuda.empty_cache()

# compute tensors
A = torch.randn(4096,4096, device=device, dtype=dtype)
B = torch.randn(4096,4096, device=device, dtype=dtype)

print(f"PID {os.getpid()} running")

while running:

    target = random.uniform(8,12)

    while mem_gb() < target:
        alloc(random.randint(200,600))

    if mem_gb() > target + 1:
        free_some()

    # compute intensity 변화
    loops = random.randint(5,80)

    for _ in range(loops):
        C = A @ B
        A = C[:4096,:4096]

    # idle time 랜덤
    time.sleep(random.uniform(0.2,2.0))

print("stopping")