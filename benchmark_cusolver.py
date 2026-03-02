import csv
import math
from dataclasses import dataclass

import torch

from cusolver_torch import eigh


@dataclass
class Case:
    tag: str
    driver: str
    compute_vectors: bool
    copy_input: bool
    tol: float = 1e-7
    max_sweeps: int = 100
    sort_eig: bool = True
    k: int | None = None
    which: str = "largest"  # for syevdx


def make_symmetric(batch: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(batch, n, n, device="cuda", dtype=dtype)
    return (a + a.transpose(-1, -2)) * 0.5

def pick_iters(n: int) -> int:
    if n <= 64:
        return 20
    if n <= 128:
        return 10
    if n <= 256:
        return 5
    return 3

def pick_batch(n: int) -> int:
    if n <= 64:
        return 256
    if n <= 128:
        return 128
    if n <= 256:
        return 64
    return 16

def run_case(a_ref: torch.Tensor, case: Case, iters: int) -> dict:
    n = a_ref.shape[-1]
    a_work = a_ref.clone()

    il, iu = 1, -1
    if case.driver == "syevdx":
        k = case.k if case.k is not None else n
        k = max(1, min(k, n))
        if case.which == "largest":
            il, iu = n - k + 1, n
        else:
            il, iu = 1, k

    def call_solver(return_meig: bool = False):
        # cuSOLVER overwrites A for these routines. Keep input identical per iteration.
        if case.copy_input:
            target = a_ref
        else:
            a_work.copy_(a_ref)
            target = a_work

        return eigh(
            target,
            compute_vectors=case.compute_vectors,
            driver=case.driver,
            tol=case.tol,
            max_sweeps=case.max_sweeps,
            sort_eig=case.sort_eig,
            il=il,
            iu=iu,
            copy_input=case.copy_input,
            return_meig=return_meig,
        )

    # Warmup
    call_solver(return_meig=False)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        call_solver(return_meig=False)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters

    # Collect status from one run for convergence diagnostics.
    w, v, info, meig = call_solver(return_meig=True)
    torch.cuda.synchronize()

    info_host = info.detach().cpu().reshape(-1)
    nonzero_info = int((info_host != 0).sum().item())
    max_info = int(info_host.max().item()) if info_host.numel() else 0

    meig_host = meig.detach().cpu().reshape(-1)
    meig_min = int(meig_host.min().item()) if meig_host.numel() else n
    meig_max = int(meig_host.max().item()) if meig_host.numel() else n

    solved_k = n
    if case.driver == "syevdx":
        solved_k = meig_max

    return {
        "N": n,
        "B": int(a_ref.shape[0]),
        "dtype": str(a_ref.dtype).replace("torch.", ""),
        "iters": iters,
        "tag": case.tag,
        "driver": case.driver,
        "compute_vectors": case.compute_vectors,
        "copy_input": case.copy_input,
        "tol": case.tol,
        "max_sweeps": case.max_sweeps,
        "sort_eig": case.sort_eig,
        "which": case.which,
        "k_request": case.k if case.k is not None else n,
        "il": il,
        "iu": iu,
        "k_solved_min": meig_min,
        "k_solved_max": meig_max,
        "time_ms": float(ms),
        "time_per_matrix_ms": float(ms / a_ref.shape[0]),
        "throughput_mat_per_s": float((a_ref.shape[0] * 1000.0) / ms),
        "nonzero_info": nonzero_info,
        "max_info": max_info,
    }


def build_cases(n: int) -> list[Case]:
    ks = [k for k in [8, 16, 32, 64, 128] if k <= n]

    cases: list[Case] = []

    # Exact solvers.
    for compute_vectors in [False, True]:
        for copy_input in [False, True]:
            cases.append(
                Case(
                    tag=f"exact_syevd_vec{int(compute_vectors)}_copy{int(copy_input)}",
                    driver="syevd",
                    compute_vectors=compute_vectors,
                    copy_input=copy_input,
                )
            )

    for driver in ["syevj", "syevj_batched"]:
        for tol in [1e-5, 1e-7]:
            for sweeps in [20, 50, 100]:
                for sort_eig in [False, True]:
                    cases.append(
                        Case(
                            tag=f"exact_{driver}_vec0_tol{tol:g}_sw{sweeps}_sort{int(sort_eig)}_copy0",
                            driver=driver,
                            compute_vectors=False,
                            copy_input=False,
                            tol=tol,
                            max_sweeps=sweeps,
                            sort_eig=sort_eig,
                        )
                    )

        # Include one vector-producing setting for each jacobi backend.
        cases.append(
            Case(
                tag=f"exact_{driver}_vec1_tol1e-7_sw100_sort1_copy0",
                driver=driver,
                compute_vectors=True,
                copy_input=False,
                tol=1e-7,
                max_sweeps=100,
                sort_eig=True,
            )
        )

    # Truncated exact solves via syevdx.
    for k in ks:
        for which in ["largest", "smallest"]:
            for compute_vectors in [False, True]:
                cases.append(
                    Case(
                        tag=f"trunc_syevdx_k{k}_{which}_vec{int(compute_vectors)}_copy0",
                        driver="syevdx",
                        compute_vectors=compute_vectors,
                        copy_input=False,
                        k=k,
                        which=which,
                    )
                )

    return cases


def print_top(results: list[dict], n: int, title: str, predicate) -> None:
    subset = [r for r in results if r["N"] == n and predicate(r)]
    subset.sort(key=lambda x: x["time_ms"])
    print(f"\\n{title} (N={n})")
    print("rank  time_ms   per_mat_ms  mats/s   nonzero_info  tag")
    for i, r in enumerate(subset[:8], start=1):
        print(
            f"{i:>4d}  {r['time_ms']:8.3f}  {r['time_per_matrix_ms']:10.4f}  {r['throughput_mat_per_s']:7.1f}"
            f"  {r['nonzero_info']:12d}  {r['tag']}"
        )


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.float32
    ns = [64, 128, 256, 512]
    results: list[dict] = []

    for n in ns:
        b = pick_batch(n)
        iters = pick_iters(n)
        a = make_symmetric(b, n, dtype)
        cases = build_cases(n)

        print(f"\\nRunning N={n}, B={b}, iters={iters}, cases={len(cases)}")

        for idx, case in enumerate(cases, start=1):
            r = run_case(a, case, iters)
            results.append(r)
            print(
                f"[{idx:>3d}/{len(cases)}] {case.tag:<58s}"
                f" {r['time_ms']:9.3f} ms  info_nonzero={r['nonzero_info']}"
            )

    out_csv = "cusolver_benchmark_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\\nWrote {out_csv} with {len(results)} rows")

    for n in ns:
        print_top(results, n, "Top exact configs", lambda r: not r["tag"].startswith("trunc_"))
        print_top(results, n, "Top truncated configs", lambda r: r["tag"].startswith("trunc_"))

    # Global best summary.
    exact = [r for r in results if not r["tag"].startswith("trunc_")]
    trunc = [r for r in results if r["tag"].startswith("trunc_")]
    best_exact = min(exact, key=lambda x: x["time_per_matrix_ms"])
    best_trunc = min(trunc, key=lambda x: x["time_per_matrix_ms"])

    print("\\nGlobal best exact:")
    print(best_exact)
    print("\\nGlobal best truncated:")
    print(best_trunc)


if __name__ == "__main__":
    main()
