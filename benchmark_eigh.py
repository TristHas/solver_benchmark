import argparse
import csv
import math
from typing import Any

import cupy as cp
import torch

from cusolver_torch import eigh as cusolver_eigh

try:
    from eigen_g_torch import buffer_size, eigen_batch
except ImportError:
    buffer_size = None
    eigen_batch = None


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda_error_out_of_memory" in msg
        or "cuda out of memory" in msg
        or "cusolver_status_alloc_failed" in msg
    )


def _clear_cuda_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _safe_run(fn, backend: str) -> tuple[float, str, str]:
    try:
        t = fn()
        return float(t), "ok", ""
    except Exception as exc:
        if _is_oom(exc):
            _clear_cuda_mem()
            return float("nan"), "oom", str(exc).replace("\n", " ")[:500]
        return float("nan"), "error", str(exc).replace("\n", " ")[:500]


def _time_torch_eigh(n: int, b: int, dtype: torch.dtype, iters: int, lib: str) -> float:
    torch.backends.cuda.preferred_linalg_library(lib)
    a = torch.randn(b, n, n, device="cuda", dtype=dtype)
    a = (a + a.transpose(-1, -2)) * 0.5

    torch.linalg.eigh(a)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.linalg.eigh(a)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _time_cusolver_ext(
    n: int,
    b: int,
    dtype: torch.dtype,
    iters: int,
    driver: str,
    tol: float = 1e-7,
    max_sweeps: int = 100,
    sort_eig: bool = True,
) -> float:
    a = torch.randn(b, n, n, device="cuda", dtype=dtype)
    a = (a + a.transpose(-1, -2)) * 0.5

    cusolver_eigh(
        a,
        compute_vectors=True,
        driver=driver,
        tol=tol,
        max_sweeps=max_sweeps,
        sort_eig=sort_eig,
    )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        cusolver_eigh(
            a,
            compute_vectors=True,
            driver=driver,
            tol=tol,
            max_sweeps=max_sweeps,
            sort_eig=sort_eig,
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _time_cupy(n: int, b: int, dtype: cp.dtype, iters: int) -> float:
    a = cp.random.standard_normal((b, n, n), dtype=dtype)
    a = (a + a.transpose(0, 2, 1)) * 0.5

    cp.linalg.eigh(a)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(iters):
        cp.linalg.eigh(a)
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / iters


def _time_eigeng(n: int, b: int, dtype: torch.dtype, iters: int) -> float:
    if eigen_batch is None or buffer_size is None:
        return float("nan")

    nm = n
    m = n
    a = torch.randn(b, nm, n, device="cuda", dtype=dtype)
    a = (a + a.transpose(-1, -2)) * 0.5
    w = torch.empty(b, n, device="cuda", dtype=dtype)

    lwork = buffer_size(a, b, nm, n, m)
    work = torch.empty(lwork // a.element_size(), device=a.device, dtype=a.dtype)

    eigen_batch(a, w, nm, n, m, work=work)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        eigen_batch(a, w, nm, n, m, work=work)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _pick_iters(n: int, b: int) -> int:
    if n <= 64:
        return 5
    if n <= 256:
        return 3
    return 2


def run_sweep(ns: list[int], bs: list[int], dtype: torch.dtype) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for b in bs:
        for n in ns:
            iters = _pick_iters(n, b)
            print(f"\nRunning N={n}, B={b}, iters={iters}")

            tests = [
                ("torch_cusolver", lambda: _time_torch_eigh(n, b, dtype, iters, "cusolver")),
                ("torch_magma", lambda: _time_torch_eigh(n, b, dtype, iters, "magma")),
                ("cupy", lambda: _time_cupy(n, b, cp.float32, iters)),
                ("eigeng", lambda: _time_eigeng(n, b, dtype, iters)),
                ("ext_syevd", lambda: _time_cusolver_ext(n, b, dtype, iters, "syevd")),
                (
                    "ext_syevj",
                    lambda: _time_cusolver_ext(
                        n,
                        b,
                        dtype,
                        iters,
                        "syevj",
                        tol=1e-7,
                        max_sweeps=100,
                        sort_eig=True,
                    ),
                ),
                ("ext_xsyev_batched", lambda: _time_cusolver_ext(n, b, dtype, iters, "xsyev_batched")),
            ]

            for backend, fn in tests:
                t_ms, status, err = _safe_run(fn, backend)
                row = {
                    "N": n,
                    "B": b,
                    "dtype": str(dtype).replace("torch.", ""),
                    "iters": iters,
                    "backend": backend,
                    "time_ms": t_ms,
                    "time_per_matrix_ms": (t_ms / b) if math.isfinite(t_ms) else float("nan"),
                    "throughput_mat_per_s": ((b * 1000.0) / t_ms) if math.isfinite(t_ms) else float("nan"),
                    "status": status,
                    "error": err,
                }
                rows.append(row)
                print(f"  {backend:18s} {t_ms:10.3f} ms  status={status}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--bs", type=str, default="16,64,128,256,1024,4096")
    parser.add_argument("--out", type=str, default="benchmark_eigh_results.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    bs = [int(x) for x in args.bs.split(",") if x.strip()]

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    rows = run_sweep(ns=ns, bs=bs, dtype=torch.float32)

    if not rows:
        raise RuntimeError("No results generated")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
