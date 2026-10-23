#!/usr/bin/env python3
"""Benchmark FP16 RMSNorm for row length 4096 on this Mac.

The interesting case is one 4096-element vector: at 8 KiB, GPU launch
overhead dominates completely, so a hand-written NEON kernel on the CPU wins.
For large batches the comparison shifts towards the GPU.

Run:
    python3 benchmark.py
    python3 benchmark.py --rows 1 8 64 256 1024 4096 16384
    python3 benchmark.py --json results.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "build" / "librmsnorm.dylib"
EPS = 1e-5
N = 4096


def load_custom():
    if not LIB.exists():
        raise SystemExit(f"missing {LIB}; run `make` first")
    lib = ctypes.CDLL(str(LIB))
    lib.rmsnorm_f16_row.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    ]
    lib.rmsnorm_f16_batch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
    ]
    return lib


def nthreads_for(rows: int) -> int:
    # dispatch_apply costs more than it saves for a handful of rows.
    if rows < 64:
        return 1
    return 4


def timer(fn, iters: int, rounds: int = 5, sync=None, sync_each: bool = False):
    for _ in range(min(50, max(5, iters // 10))):
        fn()
    if sync is not None:
        sync()
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn()
            if sync_each and sync is not None:
                sync()
        if sync is not None:
            sync()
        samples.append((time.perf_counter_ns() - t0) / iters)
    return statistics.median(samples)


def iters_for(rows: int, latency: bool = True) -> int:
    if latency:
        if rows <= 8:
            return 200
        if rows <= 64:
            return 100
        if rows <= 256:
            return 50
        if rows <= 1024:
            return 20
        if rows <= 4096:
            return 10
        return 4
    return max(3, min(3000, 400_000 // rows))


def make_tensors(rows: int, weight: bool):
    x = torch.randn(rows, N, dtype=torch.float16)
    w = torch.randn(N, dtype=torch.float16) if weight else None
    xm = x.to("mps")
    wm = w.to("mps") if w is not None else None
    return x, w, xm, wm


def reference(x: torch.Tensor, w: torch.Tensor | None):
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    if w is not None:
        y = y * w.float()
    return y


def bench_custom(lib, x, w, y, rows, threads, latency=True):
    xp, yp = x.data_ptr(), y.data_ptr()
    wp = w.data_ptr() if w is not None else None
    fn = lib.rmsnorm_f16_batch

    def call():
        fn(xp, wp, yp, rows, N, EPS, threads)

    return timer(call, iters_for(rows, latency), 5, None)


def bench_torch(x, w, device, latency=True):
    def call():
        F.rms_norm(x, (N,), w, EPS)

    sync = torch.mps.synchronize if device == "mps" else None
    return timer(call, iters_for(x.shape[0], latency), 5, sync, sync_each=latency)


def bench_mlx(x, w, use_gpu: bool, latency=True):
    try:
        import mlx.core as mx
    except ImportError:
        return None, None
    mx.set_default_device(mx.gpu if use_gpu else mx.cpu)
    xm = mx.array(x.numpy())
    wm = mx.array(w.numpy()) if w is not None else None
    mx.eval(xm, wm) if wm is not None else mx.eval(xm)

    def call():
        mx.eval(mx.fast.rms_norm(xm, wm, EPS))

    mx.synchronize()
    return timer(call, iters_for(x.shape[0], latency), 5, mx.synchronize), mx


def fmt_us(ns: float) -> str:
    return f"{ns / 1000:.2f}"


def fmt_bw(ns: float, rows: int, weight: bool) -> str:
    nbytes = rows * N * 2 * (3 if weight else 2)
    return f"{nbytes / ns:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+",
                    default=[1, 8, 64, 256, 1024, 4096, 16384])
    ap.add_argument("--weight", action="store_true", help="also benchmark learned weight")
    ap.add_argument("--queued", action="store_true",
                    help="queue all iterations and sync once (GPU throughput, not latency)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    lib = load_custom()
    latency = not args.queued
    torch.set_num_threads(4)
    torch.mps.synchronize()

    print(f"torch {torch.__version__}  mps={torch.backends.mps.is_available()}  "
          f"cpu_threads={torch.get_num_threads()}  N={N}  eps={EPS}")
    print("mode:", "single-call latency (sync every call)" if latency else "queued throughput")

    # Correctness check against an FP32 reference.
    for weight in (False, True):
        x, w, _, _ = make_tensors(4, weight)
        y = torch.empty_like(x)
        lib.rmsnorm_f16_batch(x.data_ptr(), w.data_ptr() if w is not None else None,
                              y.data_ptr(), 4, N, EPS, 1)
        err = (y.float() - reference(x, w)).abs().max().item()
        assert err < 0.02, f"custom kernel error {err}"

    all_results = {}
    for weight in ([False, True] if args.weight else [False]):
        print("\n" + ("=" * 78))
        print("with weight" if weight else "without weight")
        print(f"{'rows':>6} | {'custom NEON':>12} | {'torch CPU':>11} | "
              f"{'torch MPS':>10} | {'MLX GPU':>9} | best")
        print("-" * 78)
        rows_out = []
        for rows in args.rows:
            x, w, xm, wm = make_tensors(rows, weight)
            y = torch.empty_like(x)
            threads = nthreads_for(rows)

            t_custom = bench_custom(lib, x, w, y, rows, threads, latency)
            assert (y.float() - reference(x, w)).abs().max().item() < 0.02
            t_cpu = bench_torch(x, w, "cpu", latency)
            t_mps = bench_torch(xm, wm, "mps", latency)
            t_mlx, _ = bench_mlx(x, w, True, latency)

            entries = {"custom": t_custom, "cpu": t_cpu, "mps": t_mps}
            if t_mlx is not None:
                entries["mlx"] = t_mlx
            best = min(entries, key=entries.get)
            print(f"{rows:6d} | {t_custom/1000:7.2f} us {fmt_bw(t_custom, rows, weight):>3} | "
                  f"{t_cpu/1000:7.2f} us | {t_mps/1000:7.2f} us | "
                  f"{(t_mlx or math.nan)/1000:7.2f} us | {best}")
            rows_out.append({
                "rows": rows,
                "threads": threads,
                "custom_us": t_custom / 1000,
                "torch_cpu_us": t_cpu / 1000,
                "torch_mps_us": t_mps / 1000,
                "mlx_gpu_us": None if t_mlx is None else t_mlx / 1000,
                "best": best,
            })
        all_results["weight" if weight else "no_weight"] = rows_out

    if args.json:
        args.json.write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
