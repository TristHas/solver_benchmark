import time
import torch
import cupy as cp
import numpy as np

from cusolver_torch import eigh as cusolver_eigh

try:
    from eigen_g_torch import eigen_batch, buffer_size
except ImportError:
    eigen_batch = None
    buffer_size = None


def time_eigh_torch(N, B=8, dtype=torch.float32, lib="cusolver", iters=10):
    torch.backends.cuda.preferred_linalg_library(lib)
    A = torch.randn(B, N, N, device="cuda", dtype=dtype)
    A = (A + A.transpose(-1, -2)) * 0.5  # symmetric
    # warmup
    torch.linalg.eigh(A)
    torch.cuda.synchronize()

    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        torch.linalg.eigh(A)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def time_eigh_cusolver_ext(
    N,
    B=8,
    dtype=torch.float32,
    iters=10,
    driver="syevd",
    compute_vectors=True,
    lower=True,
    tol=1e-7,
    max_sweeps=100,
    sort_eig=True,
):
    A = torch.randn(B, N, N, device="cuda", dtype=dtype)
    A = (A + A.transpose(-1, -2)) * 0.5  # symmetric

    # warmup
    cusolver_eigh(
        A,
        compute_vectors=compute_vectors,
        lower=lower,
        driver=driver,
        tol=tol,
        max_sweeps=max_sweeps,
        sort_eig=sort_eig,
    )
    torch.cuda.synchronize()

    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        cusolver_eigh(
            A,
            compute_vectors=compute_vectors,
            lower=lower,
            driver=driver,
            tol=tol,
            max_sweeps=max_sweeps,
            sort_eig=sort_eig,
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def time_eigh_cupy(N, B=4096, dtype=cp.float32, iters=20):
    # Create symmetric batch
    A = cp.random.standard_normal((B, N, N), dtype=dtype)
    A = (A + A.transpose(0, 2, 1)) * 0.5

    # Warmup (important for cuSOLVER handle creation + JIT kernels)
    cp.linalg.eigh(A)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iters):
        cp.linalg.eigh(A)
    end.record()

    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / iters  # ms


def time_eigh_eigeng_batch(N, B=128, dtype=torch.float32, iters=10):
    if eigen_batch is None:
        return float("nan")
    # EigenG-Batched expects A shape (L, nm, n) with nm>=n
    nm = N
    m = N
    A = torch.randn(B, nm, N, device="cuda", dtype=dtype)
    A = (A + A.transpose(-1, -2)) * 0.5
    w = torch.empty(B, N, device="cuda", dtype=dtype)

    # workspace
    lwork = buffer_size(A, B, nm, N, m)
    work = torch.empty(lwork // A.element_size(), device=A.device, dtype=A.dtype)

    # warmup
    eigen_batch(A, w, nm, N, m, work=work)
    torch.cuda.synchronize()

    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        eigen_batch(A, w, nm, N, m, work=work)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


B = 128

for N in [16, 32, 64, 128, 256, 512, 1024]:
    t_ext_syevd = time_eigh_cusolver_ext(N, B=B, driver="syevd", iters=2)
    t_ext_syevj = time_eigh_cusolver_ext(
        N, B=B, driver="syevj", iters=2, tol=1e-7, max_sweeps=100, sort_eig=True
    )
    t_cu = time_eigh_torch(N, B=B, lib="cusolver", iters=2)
    t_ma = time_eigh_torch(N, B=B, lib="magma", iters=2)
    t_py = time_eigh_cupy(N, B=B, iters=2)
    t_eg = time_eigh_eigeng_batch(N, B=B, iters=2)
    print(
        f"N={N:4d}  torch-cusolver={t_cu:8.2f} ms  torch-magma={t_ma:8.2f} ms  cupy={t_py:8.3f} ms  eigeng={t_eg:8.3f} ms  ext-syevd={t_ext_syevd:8.3f} ms  ext-syevj={t_ext_syevj:8.3f} ms"
    )
