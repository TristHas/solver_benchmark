from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import panel as pn
import hvplot.pandas  # noqa: F401

pn.extension("tabulator")


def _load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)


def _safe_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df.copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out


def dashboard_eigh(csv_path: str | Path = "benchmark_eigh_results.csv") -> pn.Column:
    df = _load_csv(csv_path)
    df = _safe_metric(df, "time_ms")

    backends = sorted(df["backend"].dropna().unique().tolist())
    statuses = sorted(df["status"].dropna().unique().tolist())

    backend_w = pn.widgets.Select(name="Backend", options=backends, value=backends[0] if backends else None)
    status_w = pn.widgets.MultiChoice(name="Status", options=statuses, value=statuses)
    metric_w = pn.widgets.Select(name="Metric", options=["time_ms", "time_per_matrix_ms", "throughput_mat_per_s"], value="time_ms")

    @pn.depends(backend_w, status_w, metric_w)
    def _view(backend, statuses_sel, metric):
        d = df.copy()
        if backend is not None:
            d = d[d["backend"] == backend]
        if statuses_sel:
            d = d[d["status"].isin(statuses_sel)]
        d = _safe_metric(d, metric)

        heat = d.hvplot.heatmap(
            x="N",
            y="B",
            C=metric,
            cmap="viridis",
            colorbar=True,
            title=f"Eigh Sweep Heatmap ({backend})",
            width=700,
            height=350,
        )

        lines = d.sort_values(["B", "N"]).hvplot.line(
            x="N",
            y=metric,
            by="B",
            title=f"{metric} vs N grouped by B",
            width=700,
            height=300,
            legend="right",
        )

        table_cols = [
            "N",
            "B",
            "backend",
            "status",
            "time_ms",
            "time_per_matrix_ms",
            "throughput_mat_per_s",
            "error",
        ]
        table = pn.widgets.Tabulator(d[table_cols].sort_values(["B", "N"]), pagination="remote", page_size=20)
        return pn.Column(heat, lines, table)

    return pn.Column(
        pn.pane.Markdown("## benchmark_eigh results"),
        pn.Row(backend_w, status_w, metric_w),
        _view,
    )


def dashboard_cusolver(csv_path: str | Path = "cusolver_vectors_knobs_results.csv") -> pn.Column:
    df = _load_csv(csv_path)

    drivers = sorted(df["driver"].dropna().unique().tolist())
    statuses = sorted(df["status"].dropna().unique().tolist()) if "status" in df.columns else ["ok"]

    driver_w = pn.widgets.MultiChoice(name="Driver", options=drivers, value=drivers)
    status_w = pn.widgets.MultiChoice(name="Status", options=statuses, value=statuses)
    metric_w = pn.widgets.Select(name="Metric", options=["time_ms", "time_per_matrix_ms", "throughput_mat_per_s"], value="time_ms")
    tag_filter_w = pn.widgets.TextInput(name="Tag contains", value="")

    @pn.depends(driver_w, status_w, metric_w, tag_filter_w)
    def _view(drivers_sel, statuses_sel, metric, tag_sub):
        d = df.copy()
        d = _safe_metric(d, metric)

        if drivers_sel:
            d = d[d["driver"].isin(drivers_sel)]
        if "status" in d.columns and statuses_sel:
            d = d[d["status"].isin(statuses_sel)]
        if tag_sub.strip():
            d = d[d["tag"].str.contains(tag_sub, case=False, na=False)]

        # Best configuration per (N, B) for current filters.
        best = (
            d.sort_values(metric, na_position="last")
            .groupby(["N", "B"], as_index=False)
            .first()
        )

        heat = best.hvplot.heatmap(
            x="N",
            y="B",
            C=metric,
            cmap="plasma",
            colorbar=True,
            title="Best filtered config per (N,B)",
            width=700,
            height=350,
        )

        scatter = d.hvplot.scatter(
            x="N",
            y=metric,
            by="driver",
            hover_cols=["B", "tag", "status"] if "status" in d.columns else ["B", "tag"],
            title=f"{metric} by driver",
            width=700,
            height=300,
            legend="right",
            alpha=0.6,
        )

        table_cols = [
            "N",
            "B",
            "driver",
            "tag",
            "status",
            "time_ms",
            "time_per_matrix_ms",
            "throughput_mat_per_s",
            "nonzero_info",
            "error",
        ]
        table_cols = [c for c in table_cols if c in d.columns]
        table = pn.widgets.Tabulator(d[table_cols].sort_values(["B", "N", "time_ms"]), pagination="remote", page_size=20)

        return pn.Column(heat, scatter, table)

    return pn.Column(
        pn.pane.Markdown("## benchmark_cusolver results"),
        pn.Row(driver_w, status_w, metric_w, tag_filter_w),
        _view,
    )


def app(
    eigh_csv: str | Path = "benchmark_eigh_results.csv",
    cusolver_csv: str | Path = "cusolver_vectors_knobs_results.csv",
) -> pn.Tabs:
    return pn.Tabs(
        ("Eigh", dashboard_eigh(eigh_csv)),
        ("CuSolver", dashboard_cusolver(cusolver_csv)),
    )


if __name__ == "__main__":
    # Usage: python plot.py then open served app by panel if needed, or import app() in notebooks.
    pane = app()
    pane.show()
