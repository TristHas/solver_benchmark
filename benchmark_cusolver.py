import argparse
import csv
import math
from dataclasses import dataclass
from typing import Any

import torch

from cusolver_torch import eigh


@dataclass
class Case:
    tag: str
    driver: str
    lower: bool = True
    deterministic_mode: int = 0
    tol: float = 1e-5
    max_sweeps: int = 20
    sort_eig: bool = True
    k: int | None = None
    which: str = "largest"
    chunk_size: int | None = None


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


def make_symmetric(batch: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(batch, n, n, device="cuda", dtype=dtype)
    return (a + a.transpose(-1, -2)) * 0.5


def pick_iters(n: int) -> int:
    if n <= 128:
        return 8
    if n <= 256:
        return 5
    return 3


def run_case(a_ref: torch.Tensor, case: Case, iters: int) -> dict[str, Any]:
    n = a_ref.shape[-1]
    b = a_ref.shape[0]
    a_work = a_ref.clone()

    il, iu = 1, -1
    if case.driver == "syevdx":
        k = case.k if case.k is not None else n
        k = max(1, min(k, n))
        if case.which == "largest":
            il, iu = n - k + 1, n
        else:
            il, iu = 1, k

    chunk = case.chunk_size if case.chunk_size is not None else b
    chunk = max(1, min(chunk, b))

    def call_once(return_meig: bool = False):
        infos = []
        meigs = []
        out_w = []
        out_v = []

        for start in range(0, b, chunk):
            end = min(b, start + chunk)
            target = a_work[start:end]
            target.copy_(a_ref[start:end])

            w, v, info, meig = eigh(
                target,
                compute_vectors=True,
                lower=case.lower,
                driver=case.driver,
                tol=case.tol,
                max_sweeps=case.max_sweeps,
                sort_eig=case.sort_eig,
                il=il,
                iu=iu,
                copy_input=False,
                deterministic_mode=case.deterministic_mode,
                return_meig=True,
            )

            if return_meig:
                infos.append(info.detach().cpu().reshape(-1))
                meigs.append(meig.detach().cpu().reshape(-1))
                out_w.append(w)
                out_v.append(v)

        if return_meig:
            return (
                torch.cat(out_w, dim=0),
                torch.cat(out_v, dim=0),
                torch.cat(infos, dim=0),
                torch.cat(meigs, dim=0),
            )
        return None

    call_once(return_meig=False)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        call_once(return_meig=False)
    end_ev.record()
    torch.cuda.synchronize()
    ms = start_ev.elapsed_time(end_ev) / iters

    _, _, info, meig = call_once(return_meig=True)
    torch.cuda.synchronize()

    nonzero_info = int((info != 0).sum().item())
    max_info = int(info.max().item()) if info.numel() else 0
    meig_min = int(meig.min().item()) if meig.numel() else n
    meig_max = int(meig.max().item()) if meig.numel() else n

    return {
        "N": n,
        "B": b,
        "dtype": str(a_ref.dtype).replace("torch.", ""),
        "iters": iters,
        "tag": case.tag,
        "driver": case.driver,
        "lower": case.lower,
        "deterministic_mode": case.deterministic_mode,
        "tol": case.tol,
        "max_sweeps": case.max_sweeps,
        "sort_eig": case.sort_eig,
        "k_request": case.k if case.k is not None else n,
        "which": case.which,
        "il": il,
        "iu": iu,
        "chunk_size": chunk,
        "time_ms": float(ms),
        "time_per_matrix_ms": float(ms / b),
        "throughput_mat_per_s": float((b * 1000.0) / ms),
        "nonzero_info": nonzero_info,
        "max_info": max_info,
        "k_solved_min": meig_min,
        "k_solved_max": meig_max,
        "status": "ok",
        "error": "",
    }


def nan_row(n: int, b: int, iters: int, case: Case, status: str, err: str) -> dict[str, Any]:
    return {
        "N": n,
        "B": b,
        "dtype": "float32",
        "iters": iters,
        "tag": case.tag,
        "driver": case.driver,
        "lower": case.lower,
        "deterministic_mode": case.deterministic_mode,
        "tol": case.tol,
        "max_sweeps": case.max_sweeps,
        "sort_eig": case.sort_eig,
        "k_request": case.k if case.k is not None else n,
        "which": case.which,
        "il": 1,
        "iu": -1,
        "chunk_size": case.chunk_size if case.chunk_size is not None else b,
        "time_ms": float("nan"),
        "time_per_matrix_ms": float("nan"),
        "throughput_mat_per_s": float("nan"),
        "nonzero_info": -1,
        "max_info": -1,
        "k_solved_min": -1,
        "k_solved_max": -1,
        "status": status,
        "error": err,
    }


def build_cases(n: int, b: int) -> list[Case]:
    ks = [k for k in [8, 16, 32, 64, 128] if k <= n]
    chunks = sorted(set([b, max(1, b // 2), max(1, b // 4)]))

    out: list[Case] = []

    for lower in [True, False]:
        for det in [0, 1, 2]:
            out.append(Case(tag=f"syevd_vec_uplo{int(lower)}_det{det}", driver="syevd", lower=lower, deterministic_mode=det))

    for lower in [True, False]:
        for tol in [1e-5, 1e-6, 1e-7]:
            for sweeps in [20, 50, 100]:
                out.append(
                    Case(
                        tag=f"syevj_batched_vec_uplo{int(lower)}_tol{tol:g}_sw{sweeps}",
                        driver="syevj_batched",
                        lower=lower,
                        tol=tol,
                        max_sweeps=sweeps,
                        sort_eig=True,
                    )
                )

    for lower in [True, False]:
        for det in [0, 1, 2]:
            out.append(Case(tag=f"xsyev_batched_vec_uplo{int(lower)}_det{det}", driver="xsyev_batched", lower=lower, deterministic_mode=det))

    for chunk in chunks:
        out.append(Case(tag=f"syevd_chunk{chunk}", driver="syevd", chunk_size=chunk))
        out.append(Case(tag=f"xsyev_batched_chunk{chunk}", driver="xsyev_batched", chunk_size=chunk))

    for k in ks:
        for which in ["largest", "smallest"]:
            out.append(Case(tag=f"syevdx_vec_k{k}_{which}", driver="syevdx", k=k, which=which))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--bs", type=str, default="16,64,128,256,1024,4096")
    parser.add_argument("--out", type=str, default="cusolver_vectors_knobs_results.csv")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    bs = [int(x) for x in args.bs.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []

    for b in bs:
        for n in ns:
            iters = pick_iters(n)
            a = make_symmetric(b, n, torch.float32)
            cases = build_cases(n, b)
            print(f"\nRunning vectors benchmark N={n}, B={b}, iters={iters}, cases={len(cases)}")

            for i, c in enumerate(cases, start=1):
                try:
                    r = run_case(a, c, iters)
                except Exception as exc:
                    if _is_oom(exc):
                        _clear_cuda_mem()
                        r = nan_row(n, b, iters, c, "oom", str(exc).replace("\n", " ")[:500])
                    else:
                        r = nan_row(n, b, iters, c, "error", str(exc).replace("\n", " ")[:500])
                rows.append(r)
                t = r["time_ms"]
                t_str = f"{t:9.3f}" if math.isfinite(t) else "      nan"
                print(f"[{i:>3d}/{len(cases)}] {c.tag:<45s} {t_str} ms status={r['status']}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
