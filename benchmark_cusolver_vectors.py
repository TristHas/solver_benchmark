import csv
from dataclasses import dataclass

import torch

from cusolver_torch import eigh


@dataclass
class Case:
    tag: str
    driver: str
    lower: bool = True
    deterministic_mode: int = 0  # 0=as-is, 1=deterministic, 2=allow-nondeterministic
    tol: float = 1e-5
    max_sweeps: int = 20
    sort_eig: bool = True
    k: int | None = None
    which: str = "largest"
    chunk_size: int | None = None


def make_symmetric(batch: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(batch, n, n, device="cuda", dtype=dtype)
    return (a + a.transpose(-1, -2)) * 0.5


def pick_iters(n: int) -> int:
    if n <= 128:
        return 8
    if n <= 256:
        return 5
    return 3


def pick_batch(n: int) -> int:
    if n <= 128:
        return 128
    if n <= 256:
        return 64
    return 16


def run_case(a_ref: torch.Tensor, case: Case, iters: int) -> dict:
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
        if case.driver in {"syevd", "syevdx", "syevj", "syevj_batched", "xsyev_batched"}:
            pass
        else:
            raise ValueError(f"Unknown driver {case.driver}")

        infos = []
        meigs = []
        out_w = None
        out_v = None

        for start in range(0, b, chunk):
            end = min(b, start + chunk)
            if start == 0 and end == b:
                target = a_ref if False else a_work
            else:
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
                if out_w is None:
                    out_w = []
                    out_v = []
                out_w.append(w)
                out_v.append(v)

        if return_meig:
            w_full = torch.cat(out_w, dim=0)
            v_full = torch.cat(out_v, dim=0)
            info_full = torch.cat(infos, dim=0)
            meig_full = torch.cat(meigs, dim=0)
            return w_full, v_full, info_full, meig_full
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

    w, v, info, meig = call_once(return_meig=True)
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
    }


def build_cases(n: int, b: int) -> list[Case]:
    ks = [k for k in [8, 16, 32, 64, 128] if k <= n]
    chunks = sorted(set([b, max(1, b // 2), max(1, b // 4)]))

    out: list[Case] = []

    # Baselines and uplo/deterministic sweeps.
    for lower in [True, False]:
        for det in [0, 1, 2]:
            out.append(Case(tag=f"syevd_vec_uplo{int(lower)}_det{det}", driver="syevd", lower=lower, deterministic_mode=det))

    # Jacobi batched for vector case.
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

    # 64-bit batched SYEVD API.
    for lower in [True, False]:
        for det in [0, 1, 2]:
            out.append(Case(tag=f"xsyev_batched_vec_uplo{int(lower)}_det{det}", driver="xsyev_batched", lower=lower, deterministic_mode=det))

    # Batch chunking strategy for two likely winners.
    for chunk in chunks:
        out.append(Case(tag=f"syevd_chunk{chunk}", driver="syevd", chunk_size=chunk))
        out.append(Case(tag=f"xsyev_batched_chunk{chunk}", driver="xsyev_batched", chunk_size=chunk))

    # Truncated vector solves.
    for k in ks:
        for which in ["largest", "smallest"]:
            out.append(Case(tag=f"syevdx_vec_k{k}_{which}", driver="syevdx", k=k, which=which))

    return out


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    ns = [64, 128, 256, 512]
    dtype = torch.float32
    rows: list[dict] = []

    for n in ns:
        b = pick_batch(n)
        iters = pick_iters(n)
        a = make_symmetric(b, n, dtype)
        cases = build_cases(n, b)

        print(f"\nRunning vectors benchmark N={n}, B={b}, iters={iters}, cases={len(cases)}")
        for i, c in enumerate(cases, start=1):
            r = run_case(a, c, iters)
            rows.append(r)
            print(f"[{i:>3d}/{len(cases)}] {c.tag:<45s} {r['time_ms']:9.3f} ms info_nonzero={r['nonzero_info']}")

    out_csv = "cusolver_vectors_knobs_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out_csv} ({len(rows)} rows)")

    for n in ns:
        subset = [r for r in rows if r["N"] == n]
        subset.sort(key=lambda x: x["time_ms"])
        print(f"\nTop configs N={n}")
        print("rank  time_ms   per_mat_ms  mats/s   info_nz  tag")
        for rank, r in enumerate(subset[:10], start=1):
            print(
                f"{rank:>4d}  {r['time_ms']:8.3f}  {r['time_per_matrix_ms']:10.4f}"
                f"  {r['throughput_mat_per_s']:7.1f}  {r['nonzero_info']:7d}  {r['tag']}"
            )


if __name__ == "__main__":
    main()
