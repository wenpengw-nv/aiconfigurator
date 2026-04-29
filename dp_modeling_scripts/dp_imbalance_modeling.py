#!/usr/bin/env python3
"""
DP Imbalance Modeling — DeepSeek-R1 wide-EP decode (sglang 0.5.9, flashmla).

Reads measured per-bucket step times, predicts step time with the DP imbalance
model (attention max-rank + MoE total-batch), and writes static / dynamic
comparison CSVs and plots.

Requires aiconfigurator installed (e.g., `pip install -e <aic-src>`).

    python dp_imbalance_modeling.py [--data-csv path]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_CSV = SCRIPT_DIR / "data" / "decode_step_aligned.csv"

DP_SIZE = 16
MOE_PROBE_ISL = 8192
WARMUP_S = 30
STATIC_INTERVAL_S = 180
DYNAMIC_INTERVAL_S = 30

ATTENTION_OPS = [
    "generation_embedding", "generation_add_norm_1",
    "generation_qkv_a_proj_gemm", "generation_downscale_gemm",
    "generation_attention",
]
MOE_OPS = [
    "generation_gate_ffn1_gemm", "generation_act_gate",
    "generation_ffn2_gemm", "generation_moe_pre_dispatch", "generation_moe",
]

GROUP_ORDER = ["G0", "G1", "G2a", "G2b", "G2c", "G2d", "G3", "G4", "G5"]
STATIC_MEAN_GROUPS = ("G0", "G1", "G2a", "G2b", "G2c", "G2d")
STATIC_SWEEP_GROUPS = ("G3", "G4")
DYNAMIC_GROUPS = ("G3", "G4", "G5")


# ---------------------------------------------------------------------------
# AIC

def build_session():
    import logging
    logging.basicConfig(level=logging.WARNING)

    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator.sdk.config import ModelConfig
    from aiconfigurator.sdk.common import (
        GEMMQuantMode, MoEQuantMode, KVCacheQuantMode, FMHAQuantMode, CommQuantMode,
    )
    from aiconfigurator.sdk.perf_database import get_database
    from aiconfigurator.sdk.backends.factory import get_backend
    from aiconfigurator.sdk import models

    mc = ModelConfig(
        tp_size=1, pp_size=1, attention_dp_size=DP_SIZE,
        moe_tp_size=1, moe_ep_size=DP_SIZE,
        gemm_quant_mode=GEMMQuantMode.fp8_block,
        moe_quant_mode=MoEQuantMode.fp8_block,
        kvcache_quant_mode=KVCacheQuantMode.fp8,
        fmha_quant_mode=FMHAQuantMode.fp8_block,
        comm_quant_mode=CommQuantMode.half,
        moe_backend="deepep_moe",
        attention_backend="flashmla",
    )
    db = get_database("h20", "sglang", "0.5.9")
    if db is None:
        sys.exit("ERROR: failed to load PerfDatabase for h20/sglang/0.5.9")
    be = get_backend("sglang")
    model = models.get_model("deepseek-ai/DeepSeek-R1", mc, be.name.value)
    return InferenceSession(model, db, be)


def aic_per_step(sess, batch: int, isl: int, osl: int = 500) -> dict:
    """One AIC static_gen call, returns op→per-step latency dict."""
    from aiconfigurator.sdk.config import RuntimeConfig
    rt = RuntimeConfig(isl=isl, osl=osl, batch_size=batch)
    summary = sess.run_static(rt, mode="static_gen")
    return {op: lat / (osl - 1) for op, lat in summary.get_generation_latency_dict().items()}


def predict_step_ms(sess, max_b: int, max_c: int, total_b: int):
    """DP imbalance step time = attention(max-rank) + MoE(total-batch / DP_SIZE)."""
    if max_b <= 0 or total_b <= 0:
        return None
    isl = max_c // max_b
    moe_b = max(1, round(total_b / DP_SIZE))
    attn = aic_per_step(sess, max_b, isl)
    moe = aic_per_step(sess, moe_b, MOE_PROBE_ISL)
    t = sum(attn.get(op, 0) for op in ATTENTION_OPS) + sum(moe.get(op, 0) for op in MOE_OPS)
    return round(t, 3)


# ---------------------------------------------------------------------------
# Data aggregation

def load_buckets(csv_path: Path) -> dict[str, list[dict]]:
    by_scen: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            by_scen[r["scenario"]].append(r)
    return by_scen


def _valid(b):
    return (int(b["num_active_ranks"]) > 0
            and int(b["max_rank_batch"]) > 0
            and int(b["total_batch"]) > 0)


def _window_mean(buckets: list[dict]) -> dict | None:
    """Per-window aggregate: mean of all four load/itl fields."""
    if not buckets:
        return None
    n = len(buckets)
    return {
        "group": buckets[0]["group"],
        "max_rank_batch": max(1, round(sum(int(b["max_rank_batch"]) for b in buckets) / n)),
        "max_rank_ctx": max(1, round(sum(int(b["max_rank_ctx"]) for b in buckets) / n)),
        "total_batch": max(1, round(sum(int(b["total_batch"]) for b in buckets) / n)),
        "measured_ms": sum(float(b["global_step_itl_max"]) for b in buckets) / n,
    }


def steady_state(buckets: list[dict]) -> dict | None:
    """G0-G2 aggregate: mean of load fields and ITL across warmup-filtered buckets."""
    s = [b for b in buckets if float(b["time_bucket_center_s"]) >= WARMUP_S and _valid(b)]
    return _window_mean(s)


def windows(buckets: list[dict], interval_s: int):
    """Yield consecutive [WARMUP+k*interval, ...) window aggregates (mean of all fields)."""
    s = [b for b in buckets if float(b["time_bucket_center_s"]) >= WARMUP_S and _valid(b)]
    if not s:
        return
    t_max = max(float(b["time_bucket_center_s"]) for b in s)
    t = float(WARMUP_S)
    while t < t_max:
        win_buckets = [b for b in s if t <= float(b["time_bucket_center_s"]) < t + interval_s]
        agg = _window_mean(win_buckets)
        if agg is not None:
            agg["t_lo"], agg["t_hi"] = t, t + interval_s
            yield agg
        t += interval_s


# ---------------------------------------------------------------------------
# Modeling

def _row(scen, agg, marker, predicted_ms):
    return {
        "scenario": scen,
        "group": agg["group"],
        "marker": marker,
        "max_rank_batch": agg["max_rank_batch"],
        "max_rank_ctx": agg["max_rank_ctx"],
        "total_batch": agg["total_batch"],
        "measured_ms": round(agg["measured_ms"], 3),
        "predicted_ms": predicted_ms,
        "error_pct": round((predicted_ms - agg["measured_ms"]) / agg["measured_ms"] * 100, 1),
    }


def build_static(sess, by_scen) -> list[dict]:
    """G0-G2 single steady-state mean, G3/G4 sampled every STATIC_INTERVAL_S."""
    out = []
    for scen in sorted(by_scen):
        grp = by_scen[scen][0]["group"]
        if grp in STATIC_MEAN_GROUPS:
            agg = steady_state(by_scen[scen])
            if agg is None:
                continue
            pred = predict_step_ms(sess, agg["max_rank_batch"], agg["max_rank_ctx"], agg["total_batch"])
            if pred is not None:
                out.append(_row(scen, agg, "mean", pred))
        elif grp in STATIC_SWEEP_GROUPS:
            for w in windows(by_scen[scen], STATIC_INTERVAL_S):
                pred = predict_step_ms(sess, w["max_rank_batch"], w["max_rank_ctx"], w["total_batch"])
                if pred is not None:
                    out.append(_row(scen, w, f"{int(w['t_lo'])}-{int(w['t_hi'])}s", pred))
    return out


def build_dynamic(sess, by_scen) -> list[dict]:
    """G3/G4/G5 traced every DYNAMIC_INTERVAL_S seconds."""
    out = []
    for scen in sorted(by_scen):
        grp = by_scen[scen][0]["group"]
        if grp not in DYNAMIC_GROUPS:
            continue
        for w in windows(by_scen[scen], DYNAMIC_INTERVAL_S):
            pred = predict_step_ms(sess, w["max_rank_batch"], w["max_rank_ctx"], w["total_batch"])
            if pred is None:
                continue
            row = _row(scen, w, "", pred)
            row["time_s"] = (w["t_lo"] + w["t_hi"]) / 2
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Plotting

_BLUE, _RED = "#1565C0", "#EF5350"


def _grid(n):
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    return fig, axes, ncols, nrows


def _draw_steady(ax, grp, gr):
    """G0-G2 panel: one continuous line across scenarios sorted by load."""
    gr = sorted(gr, key=lambda r: (r["max_rank_batch"], r["max_rank_ctx"]))
    x = np.arange(len(gr))
    ax.plot(x, [r["measured_ms"] for r in gr], "o-", color=_BLUE, lw=2, ms=5, label="Measured")
    ax.plot(x, [r["predicted_ms"] for r in gr], "s-", color=_RED, lw=2, ms=5, label="DP Model")
    ax.set_xticks(x)
    ax.set_xticklabels([r["scenario"].replace(grp + "_", "") for r in gr],
                       rotation=45, ha="right", fontsize=7)


def _draw_sweep(ax, grp, gr):
    """G3/G4 panel: per-scenario line segments separated by gaps."""
    scens = sorted({r["scenario"] for r in gr})
    x_pos = 0
    tick_x, tick_l = [], []
    for i, scen in enumerate(scens):
        pts = sorted([r for r in gr if r["scenario"] == scen],
                     key=lambda r: int(r["marker"].split("-")[0]))
        xs = list(range(x_pos, x_pos + len(pts)))
        x_pos += len(pts) + 1  # gap before next scenario
        ax.plot(xs, [p["measured_ms"] for p in pts], "o-", color=_BLUE, lw=2, ms=5,
                label="Measured" if i == 0 else None)
        ax.plot(xs, [p["predicted_ms"] for p in pts], "s-", color=_RED, lw=2, ms=5,
                label="DP Model" if i == 0 else None)
        scen_short = scen.replace(grp + "_", "")
        for x, p in zip(xs, pts):
            tick_x.append(x)
            tick_l.append(f"{scen_short}\n{p['marker']}")
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_l, rotation=45, ha="right", fontsize=6)


def plot_static(rows: list[dict], out_path: Path):
    groups = sorted({r["group"] for r in rows}, key=GROUP_ORDER.index)
    fig, axes, ncols, nrows = _grid(len(groups))
    for i, grp in enumerate(groups):
        ax = axes[i // ncols][i % ncols]
        gr = [r for r in rows if r["group"] == grp]
        if grp in STATIC_SWEEP_GROUPS:
            _draw_sweep(ax, grp, gr)
        else:
            _draw_steady(ax, grp, gr)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Step time (ms)")
        err = np.array([r["error_pct"] for r in gr])
        ax.set_title(f"{grp} (n={len(gr)}, err {err.mean():+.1f}%)", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    for j in range(len(groups), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("DP Imbalance Modeling — Static (measured vs predicted)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dynamic(rows: list[dict], out_path: Path):
    by_scen: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scen[r["scenario"]].append(r)
    fig, axes, ncols, nrows = _grid(len(by_scen))
    for i, scen in enumerate(sorted(by_scen)):
        ax = axes[i // ncols][i % ncols]
        trace = sorted(by_scen[scen], key=lambda r: r["time_s"])
        t = [r["time_s"] for r in trace]
        ax.plot(t, [r["measured_ms"] for r in trace], "o-", color=_BLUE, lw=1.5, ms=3, label="Measured")
        ax.plot(t, [r["predicted_ms"] for r in trace], "s-", color=_RED, lw=1.5, ms=3, label="DP Model")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Step time (ms)")
        err = np.array([r["error_pct"] for r in trace])
        ax.set_title(f"{scen} (err {err.mean():+.1f}%)", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    for j in range(len(by_scen), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("DP Imbalance Modeling — Dynamic (measured vs predicted)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main

def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-csv", default=str(DEFAULT_DATA_CSV))
    args = ap.parse_args()

    data_csv = Path(args.data_csv).resolve()
    if not data_csv.is_file():
        sys.exit(f"ERROR: --data-csv not found: {data_csv}")

    out_dir = SCRIPT_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_csv} ...")
    by_scen = load_buckets(data_csv)
    print(f"  {len(by_scen)} scenarios")

    print("Building AIC session ...")
    sess = build_session()

    print(f"Static modeling (G0-G2 mean + G3/G4 every {STATIC_INTERVAL_S}s) ...")
    static_rows = build_static(sess, by_scen)
    write_csv(static_rows, out_dir / "static.csv")
    plot_static(static_rows, out_dir / "static.png")
    print(f"  {len(static_rows)} rows -> output/static.{{csv,png}}")

    print(f"Dynamic modeling (G3/G4/G5 every {DYNAMIC_INTERVAL_S}s) ...")
    dynamic_rows = build_dynamic(sess, by_scen)
    write_csv(dynamic_rows, out_dir / "dynamic.csv")
    plot_dynamic(dynamic_rows, out_dir / "dynamic.png")
    print(f"  {len(dynamic_rows)} rows -> output/dynamic.{{csv,png}}")

    print("Done.")


if __name__ == "__main__":
    main()
