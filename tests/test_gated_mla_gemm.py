import sys
import os
import statistics
from pathlib import Path
import time

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch

from utils import allclose


import pytest

import torch


def gated_mla_gemm_ref(x, w, x2):
    return x2 * torch.sigmoid(torch.matmul(x, w))


def addmm(x, w, x2):
    return torch.addmm(x2, x, w, beta=0.5, alpha=1.0)


def bench_stats_us(fn, iters=10, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()

    del graph
    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    return statistics.mean(times_us), statistics.median(times_us), min(times_us), max(times_us)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip non sm100")
@pytest.mark.parametrize("m", [16, 32, 64, 128, 256, 16384])
@pytest.mark.parametrize("n", [256 * 64])
@pytest.mark.parametrize("k", [6144])
def test_gated_mla_gemm_bf16(m, n, k):
    dtype = torch.bfloat16

    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((n, k), dtype=torch.float, device="cuda").to(dtype)
    x2 = torch.randn((m, n), dtype=torch.float, device="cuda").to(dtype)

    my = hpc.gated_mla_gemm(x, w, x2)
    gt = gated_mla_gemm_ref(x, w.t(), x2)

    print("gt")
    print(gt)

    print("my")
    print(my)

    assert allclose(gt.to(torch.float32), my.to(torch.float32), rtol=0.08, atol=0.01)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip non sm100")
@pytest.mark.parametrize("m", [16, 32, 64, 128, 256, 512, 8192, 16384])
@pytest.mark.parametrize("n", [256 * 64])
@pytest.mark.parametrize("k", [6144])
def test_gated_mla_gemm_bf16_benchmark(m, n, k):
    dtype = torch.bfloat16

    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    w2 = w.t().contiguous()
    x2 = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")

    a_b_c_stats = bench_stats_us(lambda: addmm(x, w2, x2))
    ref_stats = bench_stats_us(lambda: gated_mla_gemm_ref(x, w2, x2))
    my_stats = bench_stats_us(lambda: hpc.gated_mla_gemm(x, w, x2))

    def fmt(label, stats):
        mean_us, median_us, min_us, max_us = stats
        print(
            f"{label:12s} mean: {mean_us:.2f} us, median: {median_us:.2f} us, "
            f"min: {min_us:.2f} us, max: {max_us:.2f} us"
        )

    print(f"mnk=({m},{n},{k})")
    fmt("a*b+c", a_b_c_stats)
    fmt("gated_mla_gemm_ref", ref_stats)
    fmt("gated_mla_gemm", my_stats)
    print(f"speedup: {ref_stats[0] / my_stats[0]:.2f}x")
