#!/usr/bin/env python3
"""Post-hoc run analysis for cnnLstmModel_pytorch outputs.

This script reproduces the quick ad-hoc checks we used in review:
- split sizes and seizure/non-seizure event counts
- threshold trade-offs (TPR/FPR/Youden)
- approximate confidence intervals at a selected threshold
- training-history trend summary and best epochs by key criteria

Usage:
    python user_tools/nnTraining2/analyze_run_posthoc.py --run-id 8
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze model run outputs for quick post-hoc review")
    parser.add_argument("--run-id", required=True, help="Run folder name/id under output/<model>/")
    parser.add_argument(
        "--model",
        default="cnnLstmModel_pytorch",
        help="Model output folder name under output/ (default: cnnLstmModel_pytorch)",
    )
    parser.add_argument(
        "--output-root",
        default="user_tools/nnTraining2/output",
        help="Base output folder (default: user_tools/nnTraining2/output)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold to highlight (default: 0.5)",
    )
    return parser.parse_args()


def resolve_columns(df: pd.DataFrame) -> Tuple[str, str]:
    event_col = "eventId" if "eventId" in df.columns else "EventID"
    type_col = "type" if "type" in df.columns else "Type"
    return event_col, type_col


def split_summary(run_dir: Path) -> None:
    print("\n=== Split Summary ===")
    files = [
        "trainData.csv",
        "valData.csv",
        "testData.csv",
        "trainFeatures.csv",
        "valFeatures.csv",
        "testFeatures.csv",
    ]

    any_found = False
    for fname in files:
        fpath = run_dir / fname
        if not fpath.exists():
            continue
        any_found = True
        df = pd.read_csv(fpath, low_memory=False)
        event_col, type_col = resolve_columns(df)
        grouped = df.groupby(event_col)[type_col].first()
        seizure_events = int((grouped == 1).sum())
        non_seizure_events = int((grouped != 1).sum())
        print(
            f"{fname:18s} rows={len(df):8d}  events={len(grouped):6d}  "
            f"seizure_events={seizure_events:5d}  nonseizure_events={non_seizure_events:6d}"
        )

    if not any_found:
        print("No split/feature CSV files found in run folder.")


def compute_ci(success: int, total: int, z: float = 1.96) -> Tuple[float, float, float]:
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    p = success / total
    se = math.sqrt(p * (1.0 - p) / total)
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return p, lo, hi


def _load_threshold_rows(path: Path) -> Optional[List[Tuple[float, float, float, float, int, int, int, int]]]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    thresholds: List[float] = data.get("thresholds", [])
    tpr: List[float] = data.get("tpr", [])
    fpr: List[float] = data.get("fpr", [])
    tp: List[int] = data.get("tp", [])
    fp: List[int] = data.get("fp", [])
    tn: List[int] = data.get("tn", [])
    fn: List[int] = data.get("fn", [])

    if not thresholds:
        return []

    rows = []
    for i, th in enumerate(thresholds):
        youden = tpr[i] - fpr[i]
        rows.append((th, tpr[i], fpr[i], youden, int(tp[i]), int(fp[i]), int(tn[i]), int(fn[i])))
    return rows


def _summarize_threshold_rows(label: str, rows: List[Tuple[float, float, float, float, int, int, int, int]],
                              selected_threshold: float) -> Optional[Tuple[float, float, float, float, int, int, int, int]]:
    if rows is None:
        print(f"{label}: file not found")
        return None
    if len(rows) == 0:
        print(f"{label}: threshold data file is empty")
        return None

    print(f"\n{label}")
    rows_sorted = sorted(rows, key=lambda x: x[3], reverse=True)
    best = rows_sorted[0]
    print(
        f"  Best Youden threshold: {best[0]:.3f}  "
        f"TPR={best[1]:.4f}  FPR={best[2]:.4f}  Youden={best[3]:.4f}"
    )

    selected = min(rows, key=lambda r: abs(r[0] - selected_threshold))
    if abs(selected[0] - selected_threshold) > 1e-9:
        print(f"  Requested threshold {selected_threshold:.3f} not found; using nearest {selected[0]:.3f}")

    print(
        f"  Selected threshold: {selected[0]:.3f}  "
        f"TPR={selected[1]:.4f}  FPR={selected[2]:.4f}  Youden={selected[3]:.4f}"
    )

    tpr_p, tpr_lo, tpr_hi = compute_ci(selected[4], selected[4] + selected[7])
    fpr_p, fpr_lo, fpr_hi = compute_ci(selected[5], selected[5] + selected[6])
    print(
        "  Approx 95% CI @ selected threshold: "
        f"TPR={tpr_p:.4f} [{tpr_lo:.4f}, {tpr_hi:.4f}], "
        f"FPR={fpr_p:.4f} [{fpr_lo:.4f}, {fpr_hi:.4f}]"
    )

    print("  Top 5 thresholds by Youden:")
    for th, tpr_v, fpr_v, y, _, _, _, _ in rows_sorted[:5]:
        print(f"    th={th:.3f}  TPR={tpr_v:.4f}  FPR={fpr_v:.4f}  Youden={y:.4f}")

    return selected


def threshold_analysis(run_dir: Path, model: str, selected_threshold: float) -> None:
    print("\n=== Threshold Analysis ===")

    outputs = {
        "Event-level (all seizures)": run_dir / f"{model}_event_threshold_data.json",
        "Production-level (all seizures)": run_dir / f"{model}_production_threshold_data.json",
        "Event-level (tonic-clonic seizures)": run_dir / f"{model}_event_threshold_data_tonic_clonic.json",
        "Production-level (tonic-clonic seizures)": run_dir / f"{model}_production_threshold_data_tonic_clonic.json",
    }

    selected_rows: Dict[str, Optional[Tuple[float, float, float, float, int, int, int, int]]] = {}
    for label, path in outputs.items():
        rows = _load_threshold_rows(path)
        selected_rows[label] = _summarize_threshold_rows(label, rows, selected_threshold)

    event_sel = selected_rows.get("Event-level (all seizures)")
    prod_sel = selected_rows.get("Production-level (all seizures)")
    if event_sel is not None and prod_sel is not None:
        print("\nEvent vs Production comparison at selected threshold")
        print(
            f"  Event:      TPR={event_sel[1]:.4f}  FPR={event_sel[2]:.4f}  "
            f"TP={event_sel[4]} FP={event_sel[5]} TN={event_sel[6]} FN={event_sel[7]}"
        )
        print(
            f"  Production: TPR={prod_sel[1]:.4f}  FPR={prod_sel[2]:.4f}  "
            f"TP={prod_sel[4]} FP={prod_sel[5]} TN={prod_sel[6]} FN={prod_sel[7]}"
        )
        print(
            f"  Delta (production - event): dTPR={prod_sel[1] - event_sel[1]:+.4f}, "
            f"dFPR={prod_sel[2] - event_sel[2]:+.4f}"
        )


def training_summary(run_dir: Path) -> None:
    print("\n=== Training History Summary ===")
    history_path = run_dir / "training_history.json"
    if not history_path.exists():
        print(f"Missing file: {history_path}")
        return

    with history_path.open("r", encoding="utf-8") as f:
        h = json.load(f)

    far = h.get("far")
    sensitivity = h.get("sensitivity")
    val_loss = h.get("val_loss")

    if not far or not sensitivity or not val_loss:
        print("Missing one or more expected series: far, sensitivity, val_loss")
        return

    n = len(far)
    best_far_idx = min(range(n), key=lambda i: far[i])
    best_vl_idx = min(range(n), key=lambda i: val_loss[i])
    best_y_idx = max(range(n), key=lambda i: sensitivity[i] - far[i])
    last = n - 1

    print(f"Epochs recorded: {n}")
    print(
        f"Best FAR epoch: {best_far_idx + 1}  FAR={far[best_far_idx]:.6f}  "
        f"Sens={sensitivity[best_far_idx]:.6f}  ValLoss={val_loss[best_far_idx]:.6f}"
    )
    print(
        f"Best ValLoss epoch: {best_vl_idx + 1}  ValLoss={val_loss[best_vl_idx]:.6f}  "
        f"FAR={far[best_vl_idx]:.6f}  Sens={sensitivity[best_vl_idx]:.6f}"
    )
    print(
        f"Best Youden epoch: {best_y_idx + 1}  Youden={sensitivity[best_y_idx] - far[best_y_idx]:.6f}  "
        f"FAR={far[best_y_idx]:.6f}  Sens={sensitivity[best_y_idx]:.6f}"
    )
    print(
        f"Last epoch: {last + 1}  FAR={far[last]:.6f}  "
        f"Sens={sensitivity[last]:.6f}  ValLoss={val_loss[last]:.6f}"
    )


def main() -> None:
    args = parse_args()

    run_dir = Path(args.output_root) / args.model / str(args.run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print("Post-hoc run analysis")
    print(f"Run directory: {run_dir}")

    split_summary(run_dir)
    threshold_analysis(run_dir, args.model, args.threshold)
    training_summary(run_dir)


if __name__ == "__main__":
    main()
