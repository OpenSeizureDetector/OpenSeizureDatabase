#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import importlib
#from tkinter import Y
import pandas as pd
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.osdAlgTools
import libosd.configUtils

try:
    from user_tools.nnTraining2 import augmentData
except ImportError:
    import augmentData

from sklearn.metrics import classification_report
from sklearn import metrics
import json
from datetime import datetime, timedelta

import nnTrainer


# NDA events are ~3 minutes long; estimate false alarms per day for real-world FAR
NDA_EVENT_DURATION_MIN = 3.0
NDA_EVENTS_PER_DAY = 24 * 60 / NDA_EVENT_DURATION_MIN  # 480


def _is_nda_series(series):
    """Return boolean mask where series value indicates NDA (case-insensitive)."""
    return series.astype(str).str.strip().str.lower() == 'nda'


def _fa_per_day(far):
    """Estimate false alarms per 24h assuming each NDA event ≈ NDA_EVENT_DURATION_MIN minutes."""
    try:
        return float(far) * NDA_EVENTS_PER_DAY
    except Exception:
        return 0.0


def get_test_prefill_mode(configObj):
    """Return the test-time acceleration buffer pre-fill mode, or None if disabled.

    Reads modelConfig.testBufferPrefill:
      'repeat' (default) - fill the rolling buffer by tiling the first real
                           datapoint of each buffer segment (event start, or
                           restart after a data gap), i.e. assume the device
                           was doing what it was doing at the segment start.
                           Deterministic.
      'noise'            - fill with Gaussian noise matched to the first real
                           datapoint's mean/SD. Seeded per event when the
                           top-level randomSeed config is set, random otherwise.
      'stationary'       - fill the rolling buffer with stationary (1 g) data
                           before the first datapoint of every segment, so the
                           first datapoint is scored instead of being dropped
                           while the buffer warms up. Legacy behaviour; note the
                           perfectly flat fill is out-of-distribution and tends
                           to inflate seizure probabilities at segment starts.
      'none' / 'off' / false - old behaviour (rows dropped until buffer full).

    Training is unaffected - nnTrainer.df2trainingData never calls prefillAccBuf().

    Warm-up datapoints (whose model-input window still contains pre-fill) are
    flagged (df['is_warm']) and excluded from alarm decisions - see
    _event_positive_from_probs(warm_mask=...) - but are still scored and plotted.
    """
    modelConfig = configObj.get('modelConfig') if isinstance(configObj, dict) else None
    mode = libosd.configUtils.getConfigParam("testBufferPrefill", modelConfig)
    if mode is None:
        return 'repeat'
    mode = str(mode).strip().lower()
    if mode in ('', 'none', 'off', 'false', 'no', 'disabled'):
        return None
    if mode not in ('repeat', 'noise', 'stationary', 'static'):
        print("nnTester.get_test_prefill_mode(): Warning - unknown testBufferPrefill value %r "
              "- disabling buffer pre-fill" % (mode,))
        return None
    return mode


# A dataTime jump within one event larger than this marks a missing-data span
# (left as a discontinuity by flattenData - no synthetic filler rows). The
# rolling buffer restarts there so no model window spans the gap. Must match
# flattenData's gap definition: end-time delta > 7000 ms (5 s datapoints with
# GAP_TOLERANCE_MS=2000). Normal 5 s spacing reads 4-6 s at 1 s time resolution;
# a single missing datapoint reads 9-11 s, so 7 s separates cleanly.
GAP_SEGMENT_SECONDS = 7.0


def _warmup_datapoints_for_model(nnModel, samples_per_datapoint=125):
    """Number of leading datapoints of a buffer segment whose model-input window
    still contains pre-fill (i.e. is not fully real data). Delegates to the
    model's get_warmup_datapoints(); 0 for models without a rolling buffer."""
    try:
        get_warm = getattr(nnModel, 'get_warmup_datapoints', None)
        if callable(get_warm):
            return int(get_warm(samples_per_datapoint=samples_per_datapoint))
    except Exception:
        pass
    try:
        nBuf = int(nnModel.getAccBufSize())
    except Exception:
        return 0
    if nBuf <= 0 or samples_per_datapoint <= 0:
        return 0
    import math
    return max(0, int(math.ceil(nBuf / float(samples_per_datapoint))) - 1)


def _segment_rng(base_seed, event_id):
    """Deterministic per-segment RNG for 'noise' pre-fill.

    Returns np.random.Generator seeded from (base_seed, event_id), or a
    non-deterministic generator when base_seed is None (consistent with the
    centralised seeding philosophy: null seed -> random sampling).
    """
    import numpy as _np
    if base_seed is None:
        return _np.random.default_rng()
    try:
        eid = int(str(event_id))
    except (TypeError, ValueError):
        eid = abs(hash(str(event_id))) % (2 ** 31)
    try:
        return _np.random.default_rng((int(base_seed) * 1000003 + eid) % (2 ** 32))
    except Exception:
        return _np.random.default_rng()


def _split_nonwarm_segments(probabilities, warm_mask):
    """Split a per-event probability trace into contiguous non-warm runs.

    Warm-up datapoints start a new segment (they sit at buffer-segment starts),
    so alarm runs must never bridge across them. Returns a list of float arrays
    (possibly empty if every datapoint is warm).
    """
    probs = np.asarray(probabilities, dtype=float)
    try:
        warm = np.asarray(warm_mask, dtype=bool)
    except Exception:
        warm = None
    if warm is None or warm.size != probs.size:
        return [probs] if probs.size else []
    segments, current = [], []
    for p, w in zip(probs.tolist(), warm.tolist()):
        if w:
            if current:
                segments.append(np.array(current, dtype=float))
                current = []
        else:
            current.append(p)
    if current:
        segments.append(np.array(current, dtype=float))
    return segments


def fpr_score(y, y_pred, pos_label=1, neg_label=0):
    """Calculate TPR and FPR from predictions."""
    cm = sklearn.metrics.confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = 1 - tnr
    return (tpr, fpr)


def _three_consecutive_predictions(probabilities, threshold, consecutive_required=3):
    """Return datapoint alarms where alarm starts at Nth consecutive positive sample."""
    probs = np.asarray(probabilities, dtype=float)
    out = np.zeros(len(probs), dtype=int)
    run_len = 0
    for i, p in enumerate(probs):
        if np.isnan(p):
            run_len = 0
            continue
        if p >= threshold:
            run_len += 1
            if run_len >= consecutive_required:
                out[i] = 1
        else:
            run_len = 0
    return out


def _event_positive_from_probs(probabilities, threshold, mode='event', consecutive_required=3,
                               warm_mask=None):
    """Classify an event from datapoint probabilities using event or production logic.

    warm_mask (optional bool array, one per datapoint): warm-up datapoints whose
    model-input window still contains buffer pre-fill are excluded from the
    decision. For mode='production', alarm runs must additionally not bridge
    across masked regions (segments are evaluated independently and OR-ed).
    Events with no non-warm datapoints return 0 here; callers needing to
    distinguish "negative" from "unevaluable" should check the mask first
    (see _masked_event_metrics, which uses a -1 sentinel).
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.size == 0:
        return 0

    if warm_mask is not None:
        try:
            warm = np.asarray(warm_mask, dtype=bool)
        except Exception:
            warm = None
        if warm is None or warm.size != probs.size:
            warm = np.zeros(probs.shape, dtype=bool)
        if mode == 'event':
            valid_probs = probs[~warm]
            valid_probs = valid_probs[~np.isnan(valid_probs)]
            if valid_probs.size == 0:
                return 0
            return int(np.any(valid_probs >= threshold))
        if mode == 'production':
            for seg in _split_nonwarm_segments(probs, warm):
                seg_pred = _three_consecutive_predictions(
                    seg, threshold, consecutive_required=consecutive_required)
                if np.any(seg_pred == 1):
                    return 1
            return 0
        raise ValueError(f"Unknown mode: {mode}")

    if mode == 'event':
        valid_probs = probs[~np.isnan(probs)]
        if valid_probs.size == 0:
            return 0
        return int(np.any(valid_probs >= threshold))

    if mode == 'production':
        dp_pred = _three_consecutive_predictions(probs, threshold, consecutive_required=consecutive_required)
        return int(np.any(dp_pred == 1))

    raise ValueError(f"Unknown mode: {mode}")


def _threshold_metrics_from_event_probs(event_probs_list, true_labels, threshold_list,
                                        mode='event', positive_mask=None,
                                        negative_mask=None,
                                        consecutive_required=3, warm_masks=None):
    """Compute threshold TPR/FPR curves from per-event probability sequences.

    warm_masks (optional, one bool array per event): when given, masked curves
    are computed alongside the standard ones by excluding warm-up datapoints
    from each event's decision (production runs cannot bridge masked regions).
    Events with no non-warm datapoint are excluded from the masked denominators
    and counted in 'n_excluded_warm_only'. Masked series are stored under the
    'tpr_masked'/'fpr_masked'/'tp_masked'/... keys.
    """
    y_true = np.asarray(true_labels).astype(int)
    if positive_mask is None:
        pos_mask = (y_true == 1)
    else:
        pos_mask = np.asarray(positive_mask).astype(bool)
    if negative_mask is None:
        neg_mask = (y_true == 0)
    else:
        neg_mask = np.asarray(negative_mask).astype(bool)

    out = {
        'thresholds': [], 'tpr': [], 'fpr': [],
        'tp': [], 'fp': [], 'tn': [], 'fn': [],
        'n_positive': int(pos_mask.sum()),
        'n_negative': int(neg_mask.sum()),
        'mode': mode,
    }

    # Warm-up masking: per-event validity (at least one non-warm datapoint).
    # Misaligned/missing masks fall back to unmasked handling for that event.
    if warm_masks is None:
        warm_list = [None] * len(event_probs_list)
    else:
        warm_list = list(warm_masks) + [None] * max(0, len(event_probs_list) - len(warm_masks))
    masked_valid = []
    for probs, wm in zip(event_probs_list, warm_list):
        p = np.asarray(probs, dtype=float) if probs is not None else np.array([])
        try:
            w = np.asarray(wm, dtype=bool) if wm is not None else None
        except Exception:
            w = None
        if w is None or w.size != p.size:
            masked_valid.append(True if p.size > 0 else False)
        else:
            masked_valid.append(bool((~w).any()))
    masked_valid = np.array(masked_valid, dtype=bool)
    if warm_masks is not None:
        out['n_excluded_warm_only'] = int((~masked_valid).sum())
        out['tpr_masked'] = []
        out['fpr_masked'] = []
        out['tp_masked'] = []
        out['fp_masked'] = []
        out['tn_masked'] = []
        out['fn_masked'] = []

    for th in threshold_list:
        preds = np.array([
            _event_positive_from_probs(probs, th, mode=mode, consecutive_required=consecutive_required)
            for probs in event_probs_list
        ], dtype=int)

        tp = int(((preds == 1) & pos_mask).sum())
        fn = int(((preds == 0) & pos_mask).sum())
        fp = int(((preds == 1) & neg_mask).sum())
        tn = int(((preds == 0) & neg_mask).sum())

        tpr = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        out['thresholds'].append(float(th))
        out['tpr'].append(float(tpr))
        out['fpr'].append(float(fpr))
        out['tp'].append(tp)
        out['fp'].append(fp)
        out['tn'].append(tn)
        out['fn'].append(fn)

        if warm_masks is not None:
            preds_m = np.array([
                _event_positive_from_probs(probs, th, mode=mode,
                                           consecutive_required=consecutive_required,
                                           warm_mask=wm)
                for probs, wm in zip(event_probs_list, warm_list)
            ], dtype=int)
            pos_m = pos_mask & masked_valid
            neg_m = neg_mask & masked_valid
            tp_m = int(((preds_m == 1) & pos_m).sum())
            fn_m = int(((preds_m == 0) & pos_m).sum())
            fp_m = int(((preds_m == 1) & neg_m).sum())
            tn_m = int(((preds_m == 0) & neg_m).sum())
            out['tp_masked'].append(tp_m)
            out['fn_masked'].append(fn_m)
            out['fp_masked'].append(fp_m)
            out['tn_masked'].append(tn_m)
            out['tpr_masked'].append(float(tp_m / (tp_m + fn_m)) if (tp_m + fn_m) > 0 else 0.0)
            out['fpr_masked'].append(float(fp_m / (fp_m + tn_m)) if (fp_m + tn_m) > 0 else 0.0)

    return out


def _masked_event_metrics(true_labels, masked_preds):
    """Operating-point metrics over warm-up-masked per-event predictions.

    masked_preds uses -1 for events with no non-warm datapoint (unevaluable);
    those events are excluded from the rates and counted in 'n_excluded'.
    Returns a dict with tp/fp/tn/fn/tpr/fpr/n_excluded/n_events.
    """
    y_true = np.asarray(true_labels).astype(int)
    preds = np.asarray(masked_preds).astype(int)
    valid = preds != -1
    n_excluded = int((~valid).sum())
    if valid.sum() == 0:
        return {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'tpr': 0.0, 'fpr': 0.0,
                'n_excluded': n_excluded, 'n_events': int(len(y_true))}
    tp = int(((preds == 1) & (y_true == 1) & valid).sum())
    fn = int(((preds == 0) & (y_true == 1) & valid).sum())
    fp = int(((preds == 1) & (y_true == 0) & valid).sum())
    tn = int(((preds == 0) & (y_true == 0) & valid).sum())
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'tpr': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'n_excluded': n_excluded, 'n_events': int(len(y_true))}


def _prod_masked_preds(event_stats_df, threshold, consecutive_required=3):
    """Per-event production-rule predictions with warm-up masking.

    Uses each row's 'event_probs_list' with its 'event_warm_list' (alarm runs
    cannot bridge masked regions). Returns an int array with -1 for events
    that have no non-warm datapoint (unevaluable - excluded from masked rates).
    """
    out = []
    for _, row in event_stats_df.iterrows():
        probs = row.get('event_probs_list', [])
        warm = row.get('event_warm_list', None)
        try:
            p = np.asarray(probs, dtype=float)
        except Exception:
            p = np.array([])
        try:
            w = np.asarray(warm, dtype=bool) if warm is not None else None
        except Exception:
            w = None
        if p.size == 0 or w is None or w.size != p.size or not bool((~w).any()):
            out.append(-1)
            continue
        out.append(int(_event_positive_from_probs(
            p, threshold, mode='production',
            consecutive_required=consecutive_required, warm_mask=w)))
    return np.array(out, dtype=int)


def _plot_threshold_analysis(threshold_data, out_path, title_prefix, level_label):
    """Create TPR/FPR-vs-threshold and ROC-style plots for a threshold sweep."""
    thresholds = threshold_data['thresholds']
    tpr_list = threshold_data['tpr']
    fpr_list = threshold_data['fpr']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(thresholds, tpr_list, 'o-', color='green', linewidth=2, markersize=8, label='TPR')
    axes[0].plot(thresholds, fpr_list, 's-', color='red', linewidth=2, markersize=8, label='FPR')
    axes[0].set_xlabel('Threshold', fontsize=12)
    axes[0].set_ylabel('Rate', fontsize=12)
    axes[0].set_title(f'{title_prefix}: {level_label} TPR/FPR vs Threshold', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])

    for i, th in enumerate(thresholds):
        if th in [0.3, 0.5, 0.7]:
            axes[0].annotate(f'{tpr_list[i]:.2f}',
                             xy=(th, tpr_list[i]),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=9,
                             color='green')
            axes[0].annotate(f'{fpr_list[i]:.2f}',
                             xy=(th, fpr_list[i]),
                             xytext=(5, -15),
                             textcoords='offset points',
                             fontsize=9,
                             color='red')

    sorted_indices = np.argsort(fpr_list)
    sorted_fpr = [fpr_list[i] for i in sorted_indices]
    sorted_tpr = [tpr_list[i] for i in sorted_indices]
    sorted_th = [thresholds[i] for i in sorted_indices]

    axes[1].plot(sorted_fpr, sorted_tpr, 'o-', color='blue', linewidth=2, markersize=8)
    axes[1].plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Random Classifier')
    axes[1].set_xlabel('False Positive Rate (FPR)', fontsize=12)
    axes[1].set_ylabel('True Positive Rate (TPR)', fontsize=12)
    axes[1].set_title(f'{title_prefix}: {level_label} ROC Curve', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])

    for fpr_val, tpr_val, th_val in zip(sorted_fpr, sorted_tpr, sorted_th):
        if th_val in [0.3, 0.5, 0.7]:
            axes[1].annotate(f'th={th_val}',
                             xy=(fpr_val, tpr_val),
                             xytext=(10, -10),
                             textcoords='offset points',
                             fontsize=9,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                             arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_event_vs_production_thresholds(event_data, production_data, out_path, title_prefix,
                                         nda_event_data=None, nda_production_data=None):
    """Overlay event vs production threshold curves for quick visual comparison.

    If nda_* data are provided, overlay NDA-only FPR curves for comparison and
    add a secondary axis estimating false alarms per day (NDA events ≈3 min,
    i.e. 480 NDA events per 24h).
    """
    thresholds = event_data['thresholds']
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(thresholds, event_data['tpr'], 'o-', color='green', linewidth=2, label='Event TPR')
    ax.plot(thresholds, event_data['fpr'], 'o--', color='green', linewidth=2, label='Event FPR (all non-seizure)')
    ax.plot(thresholds, production_data['tpr'], 's-', color='blue', linewidth=2, label='Production TPR (3-consecutive)')
    ax.plot(thresholds, production_data['fpr'], 's--', color='blue', linewidth=2, label='Production FPR (all, 3-consecutive)')

    has_nda = nda_event_data is not None and nda_production_data is not None
    if has_nda:
        ax.plot(thresholds, nda_event_data['fpr'], '^-', color='orange', linewidth=2, markersize=7, label='Event FPR (NDA only)')
        ax.plot(thresholds, nda_production_data['fpr'], 'v--', color='purple', linewidth=2, markersize=7, label='Production FPR (NDA only, 3-consecutive)')

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    tc_suffix = " (tonic-clonic)" if "tonic_clonic" in out_path else ""
    nda_suffix = " + NDA" if has_nda else ""
    ax.set_title(f'{title_prefix}: Event-Level vs Production-Level Threshold Curves{tc_suffix}{nda_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    if has_nda:
        # Secondary axis: FA/day = FAR_nda * 480 (3-min NDA events)
        ax2 = ax.twinx()
        ax2.set_ylabel('NDA FA/day (≈3 min/event, 480/day)', color='orange', fontsize=11)
        ax2.set_ylim(0, NDA_EVENTS_PER_DAY * 1.05)
        ax2.tick_params(axis='y', labelcolor='orange')
        # Lightly shade NDA curves' FA/day equivalence for quick reading
        # No extra line needed; axis conversion is linear.

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_nda_fa_threshold(nda_event_data, nda_production_data, out_path, title_prefix):
    """Dedicated NDA FAR vs threshold plot with FA/day secondary axis."""
    thresholds = nda_event_data['thresholds']
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(thresholds, nda_event_data['fpr'], '^-', color='orange', linewidth=2, markersize=8, label='Event FPR (NDA only)')
    ax.plot(thresholds, nda_production_data['fpr'], 'v--', color='purple', linewidth=2, markersize=8, label='Production FPR (NDA only, 3-consecutive)')
    # Also show TPR for reference (same TPR as all-seizure curves, carried in nda data)
    ax.plot(thresholds, nda_event_data['tpr'], 'o-', color='green', linewidth=2, markersize=6, label='TPR (seizures)')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title(f'{title_prefix}: NDA FAR vs Threshold (FA/day ≈ FAR×480)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    # FA/day secondary axis
    ax2 = ax.twinx()
    ax2.set_ylabel('False Alarms per Day (NDA, 3 min/event)', color='orange', fontsize=11)
    ax2.set_ylim(0, NDA_EVENTS_PER_DAY * 1.05)
    ax2.tick_params(axis='y', labelcolor='orange')
    # Add FA/day markers at standard thresholds
    for th in [0.3, 0.5, 0.7]:
        if th in thresholds:
            idx = thresholds.index(th)
            fa_day_event = _fa_per_day(nda_event_data['fpr'][idx])
            fa_day_prod = _fa_per_day(nda_production_data['fpr'][idx])
            ax2.annotate(f'{fa_day_event:.1f}', xy=(th, fa_day_event), xytext=(5, 5),
                         textcoords='offset points', fontsize=8, color='orange')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _parse_datetime_safe(value):
    """Robustly parse event/datapoint timestamps into naive datetime, or None.

    Handles OSDB formats seen in allData.json and flattened CSVs, e.g.
    '2022-02-17 06:35:30' and ISO-8601 variants. Returns None on failure.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo is not None else value
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        try:
            value = value.to_pydatetime()
            return value.replace(tzinfo=None) if value.tzinfo is not None else value
        except Exception:
            return None
    if isinstance(value, float) and np.isnan(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return None
    # Fast path: common OSDB formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f",
                "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    try:
        ts = pd.to_datetime(s, errors='coerce', utc=False)
        if pd.isna(ts):
            return None
        ts = pd.Timestamp(ts)
        py = ts.to_pydatetime()
        return py.replace(tzinfo=None) if py.tzinfo is not None else py
    except Exception:
        return None


def _seizure_start_from_event(event_data_time, seizure_times):
    """Return seizure onset datetime = event dataTime + seizureTimes[0], or None.

    Follows flattenData.py semantics where seizureTimes are offsets in seconds
    relative to the event reference time. Handles seizureTimes stored either as
    a list [start, end] or as a JSON/list-formatted string (as found in
    allData.json, e.g. "[-95.0, 70.0]"). Returns None if onset cannot be
    determined (missing dataTime or seizureTimes).
    """
    base_dt = _parse_datetime_safe(event_data_time)
    if base_dt is None:
        return None
    try:
        if seizure_times is None:
            return None
        # allData.json commonly stores seizureTimes as a string like "[-95.0, 70.0]"
        if isinstance(seizure_times, str):
            try:
                seizure_times = json.loads(seizure_times)
            except (json.JSONDecodeError, ValueError):
                return None
        if not isinstance(seizure_times, (list, tuple)) or len(seizure_times) < 1:
            return None
        offset_s = float(seizure_times[0])
    except (TypeError, ValueError, IndexError):
        return None
    return base_dt + timedelta(seconds=offset_s)


def _first_crossing_latency(dp_times, dp_probs, seizure_start_dt, threshold):
    """Latency (seconds) from seizure start to first dp with prob >= threshold.

    dp_times and dp_probs must be in matching chronological order. NaN
    probabilities are ignored. Returns (latency_seconds or None, alarm_time or
    None). Latency may be negative if the model fires before annotated onset.
    """
    if seizure_start_dt is None or dp_times is None or dp_probs is None:
        return None, None
    try:
        probs = np.asarray(dp_probs, dtype=float)
        if probs.size == 0:
            return None, None
        for t, p in zip(dp_times, probs):
            try:
                if p is None or (isinstance(p, float) and np.isnan(p)):
                    continue
                if float(p) >= float(threshold):
                    if t is None:
                        return None, None
                    return (t - seizure_start_dt).total_seconds(), t
            except (TypeError, ValueError):
                continue
    except Exception:
        return None, None
    return None, None


def _compute_alarm_latency(event_stats_df, df, prediction_proba, event_details_map,
                           threshold_list, debug=False):
    """Compute per-event alarm latency for each threshold in threshold_list.

    Latency = (first datapoint dataTime with seizure probability >= threshold)
              minus (event dataTime + seizureTimes[0]).

    Args:
        event_stats_df: per-event dataframe with 'eventId', 'true_label', 'subType'.
        df: filtered datapoint dataframe with 'eventId' and 'dataTime' columns,
            whose row order matches prediction_proba rows.
        prediction_proba: (n_datapoints, n_classes) array; class 1 = seizure.
        event_details_map: dict str(eventId) -> dict with 'dataTime',
            'seizureTimes', 'userId', 'subType'.
        threshold_list: iterable of probability thresholds.

    Returns:
        (latency_data dict, per_event_df DataFrame).
        latency_data has 'thresholds', 'all' and 'tonic_clonic' entries each
        with mean/std/n_detected plus n_total/n_with_onset. Statistics are over
        detected events only; undetected or onset-unknown events are excluded
        from mean/std but reported via counts.
    """
    thresholds = [float(th) for th in threshold_list]
    p_seizure_all = np.asarray(prediction_proba[:, 1], dtype=float)

    # Tonic-clonic positives (same definition as threshold analysis)
    try:
        tc_mask_all = (
            (event_stats_df['true_label'].values == 1) &
            event_stats_df['subType'].astype(str).str.contains('tonic-clonic', case=False, na=False).values
        )
    except Exception:
        tc_mask_all = np.zeros(len(event_stats_df), dtype=bool)

    per_event_rows = []
    # Collect latencies per threshold for summary stats
    lat_all = {th: [] for th in thresholds}
    lat_tc = {th: [] for th in thresholds}
    n_with_onset_all = 0
    n_with_onset_tc = 0

    for idx, ev_row in event_stats_df.iterrows():
        event_id = ev_row['eventId']
        true_label = int(ev_row.get('true_label', 0))
        is_tc = bool(tc_mask_all[event_stats_df.index.get_loc(idx)] if idx in event_stats_df.index else False)
        meta = event_details_map.get(str(event_id), {})
        seizure_start = _seizure_start_from_event(meta.get('dataTime'), meta.get('seizureTimes'))
        has_onset = seizure_start is not None

        # Gather this event's datapoints in chronological order
        try:
            group = df[df['eventId'] == event_id]
        except Exception:
            group = df.iloc[0:0]
        if len(group) > 0:
            try:
                row_pos = group.index.to_numpy(dtype=int)
                # Guard against stale indices (e.g. after filtering)
                row_pos = row_pos[(row_pos >= 0) & (row_pos < len(p_seizure_all))]
                probs = p_seizure_all[row_pos]
                times_raw = group.loc[group.index.isin(row_pos)].copy() if len(row_pos) != len(group) else group
                # Align probs with times_raw order before sorting
                if len(row_pos) == len(times_raw):
                    order_probs = probs
                    order_idx = row_pos
                else:
                    order_idx = times_raw.index.to_numpy(dtype=int)
                    order_idx = order_idx[(order_idx >= 0) & (order_idx < len(p_seizure_all))]
                    order_probs = p_seizure_all[order_idx]
                # Warm-up masking: datapoints whose window still contained
                # buffer pre-fill cannot trigger the alarm (set to NaN, which
                # _first_crossing_latency ignores). df row order matches
                # prediction_proba rows (both follow kept-row order).
                try:
                    _warm_all = df['is_warm'].values.astype(bool) \
                        if 'is_warm' in df.columns else None
                    if _warm_all is not None and len(_warm_all) != len(p_seizure_all):
                        _warm_all = None
                except Exception:
                    _warm_all = None
                if _warm_all is not None:
                    try:
                        order_warm = _warm_all[order_idx]
                    except Exception:
                        order_warm = None
                else:
                    order_warm = None
                parsed = [_parse_datetime_safe(v) for v in times_raw['dataTime'].tolist()] \
                    if 'dataTime' in times_raw.columns else [None] * len(times_raw)
                # Sort by time, keeping None times last
                sort_idx = sorted(range(len(parsed)),
                                  key=lambda i: (parsed[i] is None, parsed[i]))
                dp_times = [parsed[i] for i in sort_idx]
                dp_probs = [float(order_probs[i]) if i < len(order_probs) else float('nan')
                            for i in sort_idx]
                if order_warm is not None and len(order_warm) == len(order_probs):
                    dp_warm = [bool(order_warm[i]) if i < len(order_warm) else False
                               for i in sort_idx]
                    dp_probs = [float('nan') if w else p
                                for p, w in zip(dp_probs, dp_warm)]
            except Exception:
                dp_times, dp_probs = [], []
        else:
            dp_times, dp_probs = [], []

        row = {
            'EventID': event_id,
            'UserID': meta.get('userId', ev_row.get('userId', 'N/A')),
            'SubType': meta.get('subType', ev_row.get('subType', '')),
            'TrueLabel': true_label,
            'SeizureStart': seizure_start.isoformat(sep=' ') if seizure_start is not None else '',
        }
        if true_label == 1 and has_onset:
            n_with_onset_all += 1
            if is_tc:
                n_with_onset_tc += 1

        for th in thresholds:
            col = f'latency_th_{th:.1f}'
            if true_label != 1 or not has_onset or len(dp_times) == 0:
                row[col] = ''
                continue
            lat, _ = _first_crossing_latency(dp_times, dp_probs, seizure_start, th)
            if lat is None:
                row[col] = ''
            else:
                row[col] = float(lat)
                lat_all[th].append(float(lat))
                if is_tc:
                    lat_tc[th].append(float(lat))
        per_event_rows.append(row)

    def _summ(vals):
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return None, None, 0
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        return mean, std, int(arr.size)

    n_total_all = int((event_stats_df['true_label'].values == 1).sum())
    n_total_tc = int(tc_mask_all.sum())
    latency_data = {
        'thresholds': thresholds,
        'unit': 'seconds',
        'definition': ('latency = first datapoint dataTime with seizure probability '
                       '>= threshold minus seizure start (event dataTime + seizureTimes[0]); '
                       'warm-up datapoints (window still containing buffer pre-fill) '
                       'cannot trigger the alarm; '
                       'negative = alarm before annotated onset; '
                       'statistics over detected events with known onset only'),
        'all': {'mean': [], 'std': [], 'n_detected': [],
                'n_total': n_total_all, 'n_with_onset': int(n_with_onset_all)},
        'tonic_clonic': {'mean': [], 'std': [], 'n_detected': [],
                         'n_total': n_total_tc, 'n_with_onset': int(n_with_onset_tc)},
    }
    for th in thresholds:
        m, s, n = _summ(lat_all[th])
        latency_data['all']['mean'].append(m)
        latency_data['all']['std'].append(s)
        latency_data['all']['n_detected'].append(n)
        m, s, n = _summ(lat_tc[th])
        latency_data['tonic_clonic']['mean'].append(m)
        latency_data['tonic_clonic']['std'].append(s)
        latency_data['tonic_clonic']['n_detected'].append(n)

    per_event_df = pd.DataFrame(per_event_rows)
    if debug:
        print(f"_compute_alarm_latency: {n_total_all} seizures ({n_total_tc} TC), "
              f"{n_with_onset_all} with known onset")
    return latency_data, per_event_df


def _plot_latency_vs_threshold(latency_data, out_path, title_prefix):
    """Plot mean +/- std alarm latency vs threshold for all and TC seizures."""
    thresholds = latency_data.get('thresholds', [])
    if len(thresholds) == 0:
        return
    all_d = latency_data.get('all', {})
    tc_d = latency_data.get('tonic_clonic', {})

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.asarray(thresholds, dtype=float)

    def _series(d):
        m = np.asarray([(np.nan if v is None else v) for v in d.get('mean', [])], dtype=float)
        s = np.asarray([(np.nan if v is None else v) for v in d.get('std', [])], dtype=float)
        n = d.get('n_detected', [])
        return m, s, n

    m_all, s_all, n_all = _series(all_d)
    m_tc, s_tc, n_tc = _series(tc_d)

    ax.errorbar(x, m_all, yerr=s_all, fmt='o-', color='green', linewidth=2,
                markersize=7, capsize=4, label='All seizures (mean ± std)')
    ax.errorbar(x, m_tc, yerr=s_tc, fmt='s--', color='blue', linewidth=2,
                markersize=7, capsize=4, label='Tonic-clonic only (mean ± std)')
    ax.axhline(0.0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Seizure probability threshold', fontsize=12)
    ax.set_ylabel('Alarm latency (s, vs seizure start)', fontsize=12)
    ax.set_title(f'{title_prefix}: Alarm Latency vs Threshold\n'
                 '(first crossing minus event dataTime + seizureTimes[0])',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim([0, 1])

    # Annotate detection counts at standard thresholds
    for th_mark in [0.3, 0.5, 0.7]:
        for xv, mv, nv, color, dy in (
                *[(float(th), float(m), int(n), 'green', 6)
                  for th, m, n in zip(thresholds, m_all.tolist(), list(n_all) + [0] * len(thresholds))
                  if abs(float(th) - th_mark) < 1e-9 and not np.isnan(m)],
                *[(float(th), float(m), int(n), 'blue', -14)
                  for th, m, n in zip(thresholds, m_tc.tolist(), list(n_tc) + [0] * len(thresholds))
                  if abs(float(th) - th_mark) < 1e-9 and not np.isnan(m)]):
            ax.annotate(f'{mv:.1f}s (n={nv})', xy=(xv, mv), xytext=(5, dy),
                        textcoords='offset points', fontsize=8, color=color)

    n_all_tot = all_d.get('n_total', 0)
    n_tc_tot = tc_d.get('n_total', 0)
    ax.text(0.02, 0.02,
            f"Seizures: {n_all_tot} all / {n_tc_tot} TC; stats over detected events with known onset.\n"
            f"Negative latency = alarm before annotated seizure start.",
            transform=ax.transAxes, va='bottom', ha='left', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _format_subtype_weighting_summary(configObj):
    """Return a human-readable summary of subtype-weighted training settings."""
    model_cfg = configObj.get('modelConfig', {}) if isinstance(configObj, dict) else {}
    use_subtype_weighting = bool(model_cfg.get('useSubtypeWeighting', False))
    subtype_weights = model_cfg.get('subtypeWeights', {})
    if not isinstance(subtype_weights, dict):
        subtype_weights = {}

    lines = [
        "TRAINING SAMPLING CONFIGURATION",
        "-" * 70,
        f"Subtype weighting enabled: {use_subtype_weighting}",
    ]

    if use_subtype_weighting and subtype_weights:
        lines.append("Subtype weights:")
        for key in sorted(subtype_weights.keys()):
            lines.append(f"  {key}: {subtype_weights[key]}")
    else:
        lines.append("Subtype weights: not active")

    lines.append("")
    return "\n".join(lines)


def _extract_prob_trace_from_event_row(row):
    """Extract ordered dp0..dpN probabilities from an event-results row."""
    dp_columns = [col for col in row.index if isinstance(col, str) and col.startswith('dp')]
    if not dp_columns:
        return np.array([], dtype=float)

    ordered_cols = sorted(dp_columns, key=lambda c: int(c[2:]) if c[2:].isdigit() else 10**9)
    probs = pd.to_numeric(row[ordered_cols], errors='coerce').to_numpy(dtype=float)
    return probs[~np.isnan(probs)]


def _longest_consecutive_above_threshold(probabilities, threshold=0.5):
    """Return longest run of consecutive values >= threshold."""
    longest = 0
    current = 0
    for prob in probabilities:
        if prob >= threshold:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)


def _export_interesting_events(event_results_df, out_csv_path, prod_threshold=0.5, top_k_per_category=40):
    """Create ranked lists of interesting events for manual review."""
    rows = []
    for _, row in event_results_df.iterrows():
        probs = _extract_prob_trace_from_event_row(row)
        if probs.size == 0:
            continue

        actual = int(row.get('ActualLabel', 0))
        pred = int(row.get('ModelPrediction', 0))

        max_prob = float(np.max(probs))
        mean_prob = float(np.mean(probs))
        std_prob = float(np.std(probs))
        p95_prob = float(np.quantile(probs, 0.95))
        pct_ge_03 = float(np.mean(probs >= 0.3))
        pct_ge_05 = float(np.mean(probs >= prod_threshold))
        longest_ge_05 = _longest_consecutive_above_threshold(probs, threshold=prod_threshold)

        if actual == 1 and pred == 0:
            category = 'FN_near_miss'
            review_reason = 'False negative with strongest seizure evidence among missed events'
            sort_key = (-max_prob, -pct_ge_03, -mean_prob)
        elif actual == 0 and pred == 1:
            category = 'FP_sustained'
            review_reason = 'False positive with sustained high seizure probability'
            sort_key = (-longest_ge_05, -pct_ge_05, -mean_prob)
        elif actual == 1 and pred == 1:
            category = 'TP_fragile'
            review_reason = 'True positive with weak confidence that may regress after tuning'
            sort_key = (pct_ge_05, longest_ge_05, max_prob)
        else:
            category = 'TN_noisy'
            review_reason = 'True negative with noisy probability trace worth checking for confounders'
            sort_key = (-std_prob, -max_prob, -pct_ge_03)

        rows.append({
            'EventID': row.get('EventID'),
            'UserID': row.get('UserID', ''),
            'Type': row.get('Type', ''),
            'SubType': row.get('SubType', ''),
            'ActualLabel': actual,
            'ModelPrediction': pred,
            'MaxSeizureProbability': float(row.get('MaxSeizureProbability', max_prob)),
            'max_prob': max_prob,
            'mean_prob': mean_prob,
            'std_prob': std_prob,
            'p95_prob': p95_prob,
            'pct_ge_03': pct_ge_03,
            'pct_ge_05': pct_ge_05,
            'longest_ge_05': longest_ge_05,
            'n_datapoints': int(probs.size),
            'category': category,
            'review_reason': review_reason,
            'Description': row.get('Description', ''),
            '_sort_key': sort_key,
        })

    if not rows:
        return pd.DataFrame()

    scored_df = pd.DataFrame(rows)
    selected_frames = []
    for category in ['FN_near_miss', 'FP_sustained', 'TP_fragile', 'TN_noisy']:
        cat_df = scored_df[scored_df['category'] == category].copy()
        if len(cat_df) == 0:
            continue
        cat_df = cat_df.sort_values('_sort_key').head(top_k_per_category).copy()
        cat_df.insert(0, 'rank_within_category', range(1, len(cat_df) + 1))
        selected_frames.append(cat_df)

    out_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    if len(out_df) > 0:
        out_df = out_df.drop(columns=['_sort_key'])
        out_df.to_csv(out_csv_path, index=False)

    return out_df


def get_model_extension(framework):
    """Get the appropriate file extension for the framework."""
    if framework == 'pytorch':
        return '.pt'
    else:
        return '.keras'


def load_ptl_model_for_testing(modelFnamePath):
    """Load a PyTorch Lite (.ptl) model for testing.
    
    Args:
        modelFnamePath: Path to the .ptl model file
    
    Returns:
        Loaded PTL model ready for inference
    """
    import torch
    
    # Load the mobile-optimized model using jit.load
    # PTL models are TorchScript models optimized for mobile
    model = torch.jit.load(modelFnamePath, map_location=torch.device('cpu'))
    model.eval()
    return model


def load_pte_model_for_testing(modelFnamePath):
    """Load an ExecuTorch (.pte) model for testing.
    
    Args:
        modelFnamePath: Path to the .pte model file
    
    Returns:
        Loaded PTE Method ready for inference
    """
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
        from pathlib import Path
        import os
        
        print(f"Loading ExecuTorch .pte model from {modelFnamePath}")
        
        # Check file exists and show size/timestamp for debugging
        if os.path.exists(modelFnamePath):
            file_stat = os.stat(modelFnamePath)
            print(f"  File size: {file_stat.st_size / (1024*1024):.2f} MB")
            import datetime
            mod_time = datetime.datetime.fromtimestamp(file_stat.st_mtime)
            print(f"  Last modified: {mod_time}")
        
        # Try the portable_lib approach which has better operator support
        print(f"Loading with portable_lib extension...")
        method = _load_for_executorch(str(modelFnamePath))
        
        print(f"Successfully loaded ExecuTorch model using portable_lib")
        return method
        
    except ImportError as e:
        # Fall back to Runtime API if portable_lib not available
        try:
            from executorch.runtime import Runtime, Verification
            from pathlib import Path
            
            print(f"portable_lib not available, trying Runtime API...")
            
            # Get the Runtime singleton
            et_runtime = Runtime.get()
            
            # Load the program from the .pte file
            program = et_runtime.load_program(
                Path(modelFnamePath),
                verification=Verification.Minimal
            )
            
            # Load the forward method
            print(f"Program loaded. Available methods: {program.method_names}")
            method = program.load_method("forward")
            
            print(f"Successfully loaded ExecuTorch model using Runtime API")
            return method
            
        except Exception as e2:
            print(f"Warning: Both loading methods failed", file=sys.stderr)
            print(f"  portable_lib error: {e}", file=sys.stderr)
            print(f"  Runtime API error: {e2}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Warning: Could not load ExecuTorch (.pte) model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def load_model_for_testing(modelFnamePath, nnModel, framework='tensorflow'):
    """Load a trained model for testing (framework-agnostic).
    
    Args:
        modelFnamePath: Path to the model file
        nnModel: Model instance (needed for PyTorch architecture)
        framework: 'tensorflow' or 'pytorch'
    
    Returns:
        Loaded model ready for inference, or None if loading fails for .pte models
    """
    if framework == 'tensorflow':
        from tensorflow import keras
        model = keras.models.load_model(modelFnamePath)
        return model
    elif framework == 'pytorch':
        import torch
        
        # Check if this is a .pte (ExecuTorch) model
        if modelFnamePath.endswith('.pte'):
            pte_model = load_pte_model_for_testing(modelFnamePath)
            if pte_model is None:
                return None
            return pte_model
        
        # Check if this is a .ptl (PyTorch Lite) model
        if modelFnamePath.endswith('.ptl'):
            return load_ptl_model_for_testing(modelFnamePath)
        
        # For PyTorch, we need to recreate the model first, then load weights
        # Use weights_only=False since we trust our own checkpoint files
        checkpoint = torch.load(modelFnamePath, map_location=nnModel.device, weights_only=False)
        
        # Get model configuration from checkpoint
        if 'config' in checkpoint and 'modelConfig' in checkpoint['config']:
            nLayers = checkpoint['config']['modelConfig'].get('nLayers', 14)
        else:
            nLayers = 14
        
        # Infer input shape and num_classes from the checkpoint state_dict
        model_state = checkpoint['model_state_dict']
        num_classes = None
        input_shape = None
        
        # Find num_classes from final layer
        for key in sorted(model_state.keys(), reverse=True):
            if 'fc' in key and 'weight' in key:
                num_classes = model_state[key].shape[0]
                break
        
        # Find input shape from first conv layer  
        for key in sorted(model_state.keys()):
            if 'conv' in key and 'weight' in key:
                # Shape is (out_channels, in_channels, kernel_size)
                # For 1D conv: (out, in, kernel)
                in_channels = model_state[key].shape[1]
                # We'll use a placeholder sequence length, actual shape will come from data
                input_shape = (750, in_channels)  # 750 is typical sequence length
                break
        
        # Create model architecture if not already created
        if nnModel.model is None:
            nnModel.makeModel(input_shape=input_shape, num_classes=num_classes, nLayers=nLayers)
        
        nnModel.model.load_state_dict(checkpoint['model_state_dict'])
        nnModel.model.eval()
        return nnModel.model
    else:
        raise ValueError(f"Unknown framework: {framework}")


def evaluate_model(model, xTest, yTest, framework='tensorflow', batch_size=512, is_ptl=False):
    """Evaluate model and return loss and accuracy (framework-agnostic).
    
    Args:
        model: Trained model
        xTest: Test data
        yTest: Test labels
        framework: 'tensorflow' or 'pytorch'
        batch_size: Batch size for PyTorch evaluation to avoid OOM
        is_ptl: Whether the model is a PyTorch Lite (.ptl) model
    
    Returns:
        tuple: (test_loss, test_acc)
    """
    if framework == 'tensorflow':
        test_loss, test_acc = model.evaluate(xTest, yTest, verbose=0)
        return test_loss, test_acc
    elif framework == 'pytorch':
        import torch
        import torch.nn as nn
        
        criterion = nn.CrossEntropyLoss()
        
        if is_ptl:
            # PyTorch Lite model evaluation
            # PTL models expect shape (batch, channels, length) but xTest has (batch, length, channels)
            # Process in batches for speed
            # PTL models use mobile-optimized prepacked operators that only support CPU
            device = torch.device('cpu')
            model = model.to(device)
            
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_samples = len(xTest)
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            print(f"Evaluating PTL model on CPU: {n_samples} samples in {n_batches} batches...")
            
            for batch_idx, i in enumerate(range(0, n_samples, batch_size)):
                batch_end = min(i + batch_size, n_samples)
                xTest_batch = xTest[i:batch_end]
                yTest_batch = yTest[i:batch_end]
                
                # Convert to tensors if needed
                if not isinstance(xTest_batch, torch.Tensor):
                    xTest_tensor = torch.from_numpy(xTest_batch).float().to(device)
                    yTest_tensor = torch.from_numpy(yTest_batch).long().to(device)
                else:
                    xTest_tensor = xTest_batch.to(device)
                    yTest_tensor = yTest_batch.to(device)
                
                # Transpose to match PTL model's expected input: (batch, channels, length)
                # xTest has shape (batch, length, channels), need (batch, channels, length)
                if len(xTest_tensor.shape) == 3:
                    xTest_tensor = xTest_tensor.permute(0, 2, 1)
                
                outputs = model(xTest_tensor)
                loss = criterion(outputs, yTest_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
                batch_samples = yTest_tensor.size(0)
                total_loss += loss.item() * batch_samples
                total_correct += (predicted == yTest_tensor).sum().item()
                total_samples += batch_samples
                
                # Progress update every 10% or every 10 batches, whichever is more frequent
                progress_interval = max(1, min(10, n_batches // 10))
                if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == n_batches:
                    progress_pct = 100 * (batch_idx + 1) / n_batches
                    print(f"  PTL evaluation progress: {batch_idx + 1}/{n_batches} batches ({progress_pct:.1f}%)")
                
                # Clean up memory
                del xTest_tensor, yTest_tensor, outputs, predicted, loss
            
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            
            return avg_loss, accuracy
        else:
            # Regular PyTorch model evaluation
            model.eval()
            device = next(model.parameters()).device
            
            # Process in batches to avoid OOM
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_samples = len(xTest)
            
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    xTest_batch = xTest[i:batch_end]
                    yTest_batch = yTest[i:batch_end]
                    
                    # Convert to tensors if needed
                    if not isinstance(xTest_batch, torch.Tensor):
                        xTest_tensor = torch.from_numpy(xTest_batch).float().to(device)
                        yTest_tensor = torch.from_numpy(yTest_batch).long().to(device)
                    else:
                        xTest_tensor = xTest_batch.to(device)
                        yTest_tensor = yTest_batch.to(device)
                
                outputs = model(xTest_tensor)
                loss = criterion(outputs, yTest_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
                batch_samples = yTest_tensor.size(0)
                total_loss += loss.item() * batch_samples
                total_correct += (predicted == yTest_tensor).sum().item()
                total_samples += batch_samples
                
                # Clean up GPU memory after each batch
                del xTest_tensor, yTest_tensor, outputs, predicted, loss
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    else:
        raise ValueError(f"Unknown framework: {framework}")


def predict_model(model, xTest, framework='tensorflow', batch_size=512, is_ptl=False, is_pte=False, test_percent=100.0):
    """Get prediction probabilities from model (framework-agnostic).
    
    Args:
        model: Trained model
        xTest: Test data
        framework: 'tensorflow' or 'pytorch'
        batch_size: Batch size for PyTorch inference to avoid OOM
        is_ptl: Whether the model is a PyTorch Lite (.ptl) model
        is_pte: Whether the model is an ExecuTorch (.pte) model
    
    Returns:
        numpy array of prediction probabilities, shape (n_samples, n_classes)
    """
    if framework == 'tensorflow':
        return model.predict(xTest, verbose=0)
    elif framework == 'pytorch':
        import torch
        import numpy as np
        
        if is_pte:
            # ExecuTorch model - the model is actually a Method object
            # PTE models have static shapes and were exported with batch_size=1
            # Must process one sample at a time
            import torch
            import time
            device = torch.device('cpu')
            
            all_probs = []
            n_samples = len(xTest)
            
            print(f"Generating PTE model predictions on CPU: {n_samples} samples (batch_size=1 for static shape)...")
            print("Note: ExecuTorch Python runtime is single-threaded and slow. For production, use C++ runtime on target device.")
            
            start_time = time.time()
            last_update_time = start_time
            
            # Process one sample at a time due to static shape requirement
            for idx in range(n_samples):
                xTest_single = xTest[idx:idx+1]  # Keep batch dimension
                
                # Convert to tensor if needed
                if not isinstance(xTest_single, torch.Tensor):
                    xTest_tensor = torch.from_numpy(xTest_single).float().to(device)
                else:
                    xTest_tensor = xTest_single.to(device)
                
                # Transpose to match PTE model's expected input: (batch, channels, length)
                # xTest has shape (batch, length, channels), need (batch, channels, length)
                if len(xTest_tensor.shape) == 3:
                    xTest_tensor = xTest_tensor.permute(0, 2, 1)
                
                # Run inference - handle both portable_lib and Runtime API
                # portable_lib.ExecuTorchModule is callable directly
                # Runtime.Method uses .execute()
                if hasattr(model, 'execute'):
                    # Runtime API Method
                    outputs = model.execute((xTest_tensor,))
                else:
                    # portable_lib ExecuTorchModule - callable directly
                    outputs = model(xTest_tensor)
                
                # outputs is a list/tuple of tensors or a single tensor, get the first one
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                
                # Convert output to tensor if needed
                if not isinstance(outputs, torch.Tensor):
                    outputs = torch.from_numpy(outputs).float()
                
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy() if isinstance(probs, torch.Tensor) else probs)
                
                # Progress update every 100 samples
                current_time = time.time()
                if (idx + 1) % 100 == 0 or (idx + 1) == n_samples:
                    elapsed = current_time - start_time
                    samples_per_sec = (idx + 1) / elapsed if elapsed > 0 else 0
                    remaining_samples = n_samples - (idx + 1)
                    eta_seconds = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    progress_pct = 100 * (idx + 1) / n_samples
                    if eta_minutes > 1:
                        print(f"  Progress: {idx + 1}/{n_samples} ({progress_pct:.1f}%) | {samples_per_sec:.1f} samples/sec | ETA: {eta_minutes:.1f} min")
                    else:
                        print(f"  Progress: {idx + 1}/{n_samples} ({progress_pct:.1f}%) | {samples_per_sec:.1f} samples/sec | ETA: {eta_seconds:.0f} sec")
                
                del xTest_tensor, outputs, probs
            
            total_time = time.time() - start_time
            print(f"PTE inference completed in {total_time:.1f} seconds ({n_samples/total_time:.1f} samples/sec)")
            
            result_probs = np.vstack(all_probs)
            return result_probs
        
        elif is_ptl:
            # PyTorch Lite model - use batching for speed
            # PTL models expect shape (batch, channels, length) but xTest has (batch, length, channels)
            # PTL models use mobile-optimized prepacked operators that only support CPU
            device = torch.device('cpu')
            model = model.to(device)
            
            all_probs = []
            n_samples = len(xTest)
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            print(f"Generating PTL model predictions on CPU: {n_samples} samples in {n_batches} batches...")
            
            for batch_idx, i in enumerate(range(0, n_samples, batch_size)):
                batch_end = min(i + batch_size, n_samples)
                xTest_batch = xTest[i:batch_end]
                
                # Convert to tensor if needed
                if not isinstance(xTest_batch, torch.Tensor):
                    xTest_tensor = torch.from_numpy(xTest_batch).float().to(device)
                else:
                    xTest_tensor = xTest_batch.to(device)
                
                # Transpose to match PTL model's expected input: (batch, channels, length)
                # xTest has shape (batch, length, channels), need (batch, channels, length)
                if len(xTest_tensor.shape) == 3:
                    xTest_tensor = xTest_tensor.permute(0, 2, 1)
                
                outputs = model(xTest_tensor)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.numpy())
                
                # Progress update every 10% or every 10 batches, whichever is more frequent
                progress_interval = max(1, min(10, n_batches // 10))
                if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == n_batches:
                    progress_pct = 100 * (batch_idx + 1) / n_batches
                    print(f"  PTL prediction progress: {batch_idx + 1}/{n_batches} batches ({progress_pct:.1f}%)")
                
                del xTest_tensor, outputs, probs
            
            return np.vstack(all_probs)
        else:
            # Regular PyTorch model
            model.eval()
            device = next(model.parameters()).device
            
            # Process in batches to avoid OOM
            all_probs = []
            n_samples = len(xTest)
            
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    xTest_batch = xTest[i:batch_end]
                    
                    # Convert to tensor if needed
                    if not isinstance(xTest_batch, torch.Tensor):
                        xTest_tensor = torch.from_numpy(xTest_batch).float().to(device)
                    else:
                        xTest_tensor = xTest_batch.to(device)
                    
                    outputs = model(xTest_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy())
                    
                    # Clean up GPU memory after each batch
                    del xTest_tensor, outputs, probs
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return np.vstack(all_probs)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def testModel(configObj, dataDir='.', balanced=True, debug=False, testDataCsv=None, test_ptl=False, test_pte=False, test_percent=100.0, outputDir=None, titlePrefix=None):
    TAG = "nnTester.testModel()"
    print("____%s____" % (TAG))
    
    # Detect framework
    framework = nnTrainer.get_framework_from_config(configObj)
    print(f"{TAG}: Using framework: {framework}")
    
    modelFnameRoot = libosd.configUtils.getConfigParam("modelFname", configObj['modelConfig'])
    nnModelClassName = libosd.configUtils.getConfigParam("modelClass", configObj['modelConfig'])
    
    # If outputDir not specified, use dataDir for outputs
    if outputDir is None:
        outputDir = dataDir
    else:
        print(f"{TAG}: Using separate output directory: {outputDir}")
        os.makedirs(outputDir, exist_ok=True)
    
    # If titlePrefix not specified, use modelFnameRoot
    if titlePrefix is None:
        titlePrefix = modelFnameRoot
    else:
        print(f"{TAG}: Using title prefix: {titlePrefix}")
    
    # If testDataCsv is explicitly provided, use it directly
    if testDataCsv is not None:
        testDataPath = testDataCsv if os.path.isabs(testDataCsv) else os.path.join(dataDir, testDataCsv)
        if not os.path.exists(testDataPath):
            raise ValueError(f"{TAG}: Specified test data file not found: {testDataPath}")
        testDataFname = os.path.basename(testDataPath)
        print(f"{TAG}: Using specified test data CSV: {testDataPath}")
    else:
        # Resolve test data file path, preferring feature CSVs
        if (balanced):
            testDataFname = libosd.configUtils.getConfigParam("testBalancedFileCsv", configObj['dataFileNames'])
        else:   
            testDataFname = libosd.configUtils.getConfigParam("testFeaturesHistoryFileCsv", configObj['dataFileNames'])
        
        if testDataFname is None:
            raise ValueError(f"{TAG}: Test data filename is None. Check config for testBalancedFileCsv or testFeaturesHistoryFileCsv")

        # Build initial path
        testDataPath = os.path.join(dataDir, testDataFname) if isinstance(testDataFname, str) else None
        
        # Prefer feature CSV if it exists
        try:
            testFeaturesName = configObj['dataFileNames'].get('testFeaturesFileCsv')
            if isinstance(testFeaturesName, str):
                candidate = os.path.join(dataDir, testFeaturesName)
                if os.path.exists(candidate):
                    print(f"{TAG}: Using test features CSV {candidate}")
                    testDataPath = candidate
                    testDataFname = testFeaturesName
        except Exception:
            pass
        
        if testDataPath is None or not os.path.exists(testDataPath):
            raise ValueError(f"{TAG}: Test data file not found: {testDataPath}")

    inputDims = libosd.configUtils.getConfigParam("dims", configObj['modelConfig'])
    if (inputDims is None): inputDims = 1

    modelExt = get_model_extension(framework)
    modelFname = f"{modelFnameRoot}{modelExt}"
    
    # Parse model class name properly
    parts = nnModelClassName.split('.')
    if len(parts) < 2:
        raise ValueError("modelClass must be a module path and class name, e.g. 'mod.submod.ClassName'")
    nnModuleId = '.'.join(parts[:-1])
    nnClassId = parts[-1]

    print("%s: Importing nn Module %s" % (TAG, nnModuleId))
    nnModule = importlib.import_module(nnModuleId)
    # Instantiate the model class with modelConfig
    if configObj.get('modelConfig') is None:
        raise ValueError(f"{TAG}: configObj['modelConfig'] is None")
    nnModel = getattr(nnModule, nnClassId)(configObj['modelConfig'])

    # Test-time buffer pre-fill: score the first datapoint of each event instead
    # of dropping rows until the rolling buffer has filled (training unchanged).
    prefillMode = get_test_prefill_mode(configObj)
    nAccBuf = nnModel.getAccBufSize()
    prefillEnabled = prefillMode is not None and nAccBuf > 0
    if prefillMode is None:
        print(f"{TAG}: Buffer pre-fill disabled (testBufferPrefill=none) - rows dropped until buffer full")
    elif prefillEnabled:
        _prefill_desc = {
            'repeat': "tiling each buffer segment's first real datapoint",
            'noise': "Gaussian noise matched to each segment's first datapoint",
            'stationary': (f"{nAccBuf} samples of "
                           f"{getattr(nnModel, 'STATIONARY_ACC_MILLIG', '?')} milli-g"),
            'static': (f"{nAccBuf} samples of "
                       f"{getattr(nnModel, 'STATIONARY_ACC_MILLIG', '?')} milli-g"),
        }.get(prefillMode, prefillMode)
        print(f"{TAG}: Buffer pre-fill '{prefillMode}' ({_prefill_desc}) at each event start "
              f"and after data gaps")
    else:
        print(f"{TAG}: Buffer pre-fill '{prefillMode}' requested but model {nnModelClassName} "
              f"has no rolling acceleration buffer")

    # Load the test data from file
    print("%s: Loading Test Data from File %s" % (TAG, testDataFname))
    df_original = augmentData.loadCsv(testDataPath, debug=debug)
    print("%s: Loaded %d datapoints" % (TAG, len(df_original)))

    # Process data and track which rows are kept
    # df2trainingData may skip some rows where dp2vector returns None
    print("%s: Re-formatting data for testing" % (TAG))
    xTest_list, yTest_list, kept_indices = [], [], []
    
    # We need to replicate df2trainingData logic but track indices
    cols = list(df_original.columns)

    def _collect_axis_cols(prefix):
        with_suffix = [
            c for c in cols
            if isinstance(c, str) and c.startswith(prefix) and c.endswith('_t-0') and c[len(prefix):-4].isdigit()
        ]
        if with_suffix:
            return sorted(with_suffix, key=lambda c: int(c[len(prefix):-4]))
        no_suffix = [
            c for c in cols
            if isinstance(c, str) and c.startswith(prefix) and len(c) == 4 and c[1:].isdigit()
        ]
        return sorted(no_suffix, key=lambda c: int(c[1:]))

    accel_input_mode = str(getattr(nnModel, 'accel_input_mode', 'magnitude')).lower()
    use_xyz = accel_input_mode == 'xyz'

    m_cols = _collect_axis_cols('M')
    x_cols = _collect_axis_cols('X')
    y_cols = _collect_axis_cols('Y')
    z_cols = _collect_axis_cols('Z')

    if use_xyz:
        if len(x_cols) == 0 or len(y_cols) == 0 or len(z_cols) == 0:
            raise ValueError("XYZ mode requested but X/Y/Z columns not found in dataframe")
        if not (len(x_cols) == len(y_cols) == len(z_cols)):
            raise ValueError("X/Y/Z column counts do not match")
        x_idx = [cols.index(c) for c in x_cols]
        y_idx = [cols.index(c) for c in y_cols]
        z_idx = [cols.index(c) for c in z_cols]
        xStartCol, xEndCol = min(x_idx), max(x_idx) + 1
        yStartCol, yEndCol = min(y_idx), max(y_idx) + 1
        zStartCol, zEndCol = min(z_idx), max(z_idx) + 1
        accStartCol = accEndCol = None
    else:
        if len(m_cols) == 0:
            raise ValueError("No magnitude (Mxxx_t-0 or Mxxx) columns found in dataframe")
        m_indices = [cols.index(c) for c in m_cols]
        accStartCol = min(m_indices)
        accEndCol = max(m_indices) + 1
        xStartCol = xEndCol = yStartCol = yEndCol = zStartCol = zEndCol = None
    
    try:
        hrCol = df_original.columns.get_loc('hr')
    except:
        hrCol = None
    typeCol = df_original.columns.get_loc('type')
    eventIdCol = df_original.columns.get_loc('eventId')
    try:
        dataTimeCol = df_original.columns.get_loc('dataTime')
    except:
        dataTimeCol = None

    # Warm-up length: leading datapoints of each buffer segment whose model
    # window still contains pre-fill. Flagged in df['is_warm'] and excluded
    # from alarm decisions (but still scored and plotted).
    samples_per_dp = len(x_cols) if use_xyz else len(m_cols)
    warm_len = _warmup_datapoints_for_model(nnModel, samples_per_datapoint=samples_per_dp)
    if warm_len > 0:
        print(f"{TAG}: Warm-up masking: first {warm_len} datapoint(s) of each buffer "
              f"segment flagged as warm (excluded from alarm decisions)")
    # Base seed for deterministic 'noise' pre-fill (None -> non-deterministic,
    # mirroring the centralised seeding philosophy).
    try:
        _raw_seed = configObj.get('randomSeed', None) if isinstance(configObj, dict) else None
        noiseBaseSeed = None if _raw_seed is None else int(_raw_seed)
    except Exception:
        noiseBaseSeed = None

    lastEventId = None
    lastTime = None
    segPos = 0
    n_gap_resets = 0
    kept_warm = []
    prefillFailed = False
    for idx in range(len(df_original)):
        rowArr = df_original.iloc[idx]

        # A new buffer segment starts at each event boundary and at each
        # dataTime gap (missing-data span left as a discontinuity by
        # flattenData - no model window may span it).
        eventId = rowArr.iloc[eventIdCol]
        curTime = None
        if dataTimeCol is not None:
            try:
                curTime = _parse_datetime_safe(rowArr.iloc[dataTimeCol])
            except Exception:
                curTime = None
        newSegment = (eventId != lastEventId)
        if not newSegment and curTime is not None and lastTime is not None:
            try:
                if (curTime - lastTime).total_seconds() > GAP_SEGMENT_SECONDS:
                    newSegment = True
                    n_gap_resets += 1
            except Exception:
                pass

        dpDict = {}
        if use_xyz:
            xArr = rowArr.iloc[xStartCol:xEndCol].values.astype(float).tolist()
            yArr = rowArr.iloc[yStartCol:yEndCol].values.astype(float).tolist()
            zArr = rowArr.iloc[zStartCol:zEndCol].values.astype(float).tolist()
            raw3d = []
            for xv, yv, zv in zip(xArr, yArr, zArr):
                raw3d.extend([xv, yv, zv])
            dpDict['rawData3D'] = raw3d
        else:
            accArr = rowArr.iloc[accStartCol:accEndCol].values.astype(float).tolist()
            dpDict['rawData'] = accArr

        if newSegment:
            nnModel.resetAccBuf()
            segPos = 0
            if prefillEnabled:
                # Reference-based pre-fill: tile the segment's first real
                # datapoint ('repeat') or matched noise ('noise') instead of a
                # flat line, so warm-up windows resemble real sensor data.
                ref = dpDict.get('rawData3D') if use_xyz else dpDict.get('rawData')
                rng = _segment_rng(noiseBaseSeed, eventId) if prefillMode == 'noise' else None
                try:
                    filled = nnModel.prefillAccBuf(prefillMode, ref=ref, rng=rng)
                except TypeError:
                    # Model with legacy prefillAccBuf(mode) signature.
                    filled = nnModel.prefillAccBuf(prefillMode)
                if not filled and not prefillFailed:
                    print(f"{TAG}: Warning - buffer pre-fill mode '{prefillMode}' failed on "
                          f"model {nnModelClassName}")
                    prefillFailed = True
            lastEventId = eventId
        lastTime = curTime

        if hrCol is not None:
            try:
                dpDict['hr'] = int(rowArr.iloc[hrCol])
            except:
                dpDict['hr'] = None
        else:
            dpDict['hr'] = None

        dpInputData = nnModel.dp2vector(dpDict, normalise=False)
        if dpInputData is not None:
            xTest_list.append(dpInputData)
            yTest_list.append(rowArr.iloc[typeCol])
            kept_indices.append(idx)
            kept_warm.append(segPos < warm_len)
        segPos += 1
    if n_gap_resets > 0:
        print(f"{TAG}: Restarted rolling buffer at {n_gap_resets} dataTime gap(s) "
              f"(missing-data spans; post-gap warm-up flagged, not dropped)")
    
    # Filter dataframe to only rows that were kept (for model predictions)
    original_df_len = len(df_original)
    original_events = df_original['eventId'].nunique()
    
    # Debug: check OSD alarms before filtering
    if debug:
        print(f"{TAG}: Before filtering - OSD alarms (>=2): {(df_original['osdAlarmState'] >= 2).sum()}/{len(df_original)} datapoints")
        orig_osd_pred = (df_original['osdAlarmState'] >= 2).astype(int)
        for eid, grp in df_original.groupby('eventId'):
            evt_true = grp['type'].iloc[0]
            evt_osd_any = (grp['osdAlarmState'] >= 2).any()
            n_alarms = (grp['osdAlarmState'] >= 2).sum()
            if evt_true == 1:  # Only print seizure events
                print(f"{TAG}: Event {eid} (seizure): OSD alarms in {n_alarms}/{len(grp)} datapoints, any={evt_osd_any}")
    
    df = df_original.iloc[kept_indices].reset_index(drop=True)
    print(f"%s: Kept {len(kept_indices)} of {original_df_len} rows after filtering ({original_df_len - len(kept_indices)} removed)" % TAG)
    # Warm-up flags aligned with kept rows: True where the model-input window
    # still contained buffer pre-fill (segment starts and post-gap restarts).
    # Used to exclude warm-up datapoints from alarm decisions (they are still
    # scored, plotted and counted in datapoint-level metrics).
    if len(kept_warm) == len(df):
        df['is_warm'] = np.asarray(kept_warm, dtype=bool)
    else:
        df['is_warm'] = False
        print(f"{TAG}: Warning - warm-up flags unavailable "
              f"({len(kept_warm)} flags for {len(df)} rows); masking disabled")
    
    # Debug: check OSD alarms after filtering
    if debug:
        print(f"{TAG}: After filtering - OSD alarms (>=2): {(df['osdAlarmState'] >= 2).sum()}/{len(df)} datapoints")
        for eid, grp in df.groupby('eventId'):
            evt_true = grp['type'].iloc[0]
            evt_osd_any = (grp['osdAlarmState'] >= 2).any()
            n_alarms = (grp['osdAlarmState'] >= 2).sum()
            if evt_true == 1:  # Only print seizure events
                print(f"{TAG}: Event {eid} (seizure): OSD alarms in {n_alarms}/{len(grp)} datapoints after filter, any={evt_osd_any}")

    # Optional event-level subsampling for all model types
    if test_percent < 100.0:
        unique_events = df['eventId'].unique()
        n_events = len(unique_events)
        n_keep = max(1, int(n_events * test_percent / 100.0))
        rng = np.random.default_rng(42)
        selected_events = rng.choice(unique_events, size=n_keep, replace=False)
        event_mask_series = df['eventId'].isin(selected_events)
        event_mask = event_mask_series.tolist()
        if debug:
            print(f"{TAG}: Event-level sampling at {test_percent}% -> keeping {n_keep}/{n_events} events")
        df = df[event_mask_series].reset_index(drop=True)
        df_original = df_original[df_original['eventId'].isin(selected_events)].reset_index(drop=True)
        # Filter xTest/yTest lists in sync with sampled events
        xTest_list = [x for x, keep in zip(xTest_list, event_mask) if keep]
        yTest_list = [y for y, keep in zip(yTest_list, event_mask) if keep]
    else:
        selected_events = None

    # Count seizure and non-seizure events and validate
    n_seizure_events = (df['type'] == 1).sum()
    n_normal_events = (df['type'] == 0).sum()
    unique_seizure_events = df[df['type'] == 1]['eventId'].nunique()
    unique_normal_events = df[df['type'] == 0]['eventId'].nunique()
    
    print(f"{TAG}: Seizure datapoints: {n_seizure_events}, Non-seizure datapoints: {n_normal_events}")
    print(f"{TAG}: Seizure events: {unique_seizure_events}, Non-seizure events: {unique_normal_events}")
    
    if unique_seizure_events == 0:
        raise ValueError(f"{TAG}: ERROR - No seizure events in selected dataset. Cannot compute metrics that require seizure events.")

    print("%s: Converting to np arrays" % (TAG))
    xTest = np.array(xTest_list)
    yTest = np.array(yTest_list)

    print("%s: re-shaping array for testing" % (TAG))
    if xTest.ndim == 2:
        xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
    elif xTest.ndim == 3:
        # Keep channel-last tensors as-is, e.g. (batch, 750, 3) for XYZ mode.
        pass
    elif xTest.ndim == 4 and inputDims == 2:
        xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], xTest.shape[2], 1))
    else:
        print(f"ERROR - unsupported xTest shape {xTest.shape} for inputDims={inputDims}")
        exit(-1)

    # Load the model once
    modelFnamePath = os.path.join(dataDir, modelFname)
    
    # For PyTorch models, check if .pt, .ptl, and/or .pte exist
    models_to_test = [('pt', modelFnamePath, False, False)]  # List of (label, path, is_ptl, is_pte)
    
    if framework == 'pytorch' and modelFnamePath.endswith('.pt'):
        # Check for .ptl model
        if test_ptl:
            ptl_model_path = modelFnamePath.replace('.pt', '.ptl')
            if os.path.exists(ptl_model_path):
                print(f"{TAG}: Found .ptl model - will test both .pt and .ptl for comparison")
                models_to_test.append(('ptl', ptl_model_path, True, False))
            elif os.path.exists(modelFnamePath):
                # .pt exists but .ptl doesn't - generate .ptl for testing
                print(f"{TAG}: .ptl model not found, generating from .pt model...")
                try:
                    try:
                        from user_tools.nnTraining2.convertPt2Ptl import convert_pt_to_ptl
                    except ImportError:
                        from convertPt2Ptl import convert_pt_to_ptl
                    
                    input_shape = (1, 1, 750)  # Standard batch, channels, sequence length
                    success = convert_pt_to_ptl(
                        input_path=modelFnamePath,
                        output_path=ptl_model_path,
                        input_shape=input_shape,
                        num_classes=2,
                        verbose=True
                    )
                    
                    if success:
                        print(f"{TAG}: Successfully generated {ptl_model_path}")
                        print(f"{TAG}: Will test both .pt and .ptl models for comparison")
                        models_to_test.append(('ptl', ptl_model_path, True, False))
                    else:
                        print(f"{TAG}: Warning - Failed to generate .ptl model")
                except Exception as e:
                    print(f"{TAG}: Warning - Could not generate .ptl model: {e}")
        
        # Check for .pte model
        if test_pte:
            pte_model_path = modelFnamePath.replace('.pt', '.pte')
            if os.path.exists(pte_model_path):
                print(f"{TAG}: Found .pte model - will test for comparison")
                models_to_test.append(('pte', pte_model_path, False, True))
            elif os.path.exists(modelFnamePath):
                # .pt exists but .pte doesn't - generate .pte for testing
                print(f"{TAG}: .pte model not found, generating from .pt model...")
                try:
                    try:
                        from user_tools.nnTraining2.convertPt2Pte import convert_pt_to_pte
                    except ImportError:
                        from convertPt2Pte import convert_pt_to_pte
                    
                    input_shape = (1, 1, 750)  # Standard batch, channels, sequence length
                    success = convert_pt_to_pte(
                        input_path=modelFnamePath,
                        output_path=pte_model_path,
                        input_shape=input_shape,
                        num_classes=2,
                        verbose=True
                    )
                    
                    if success:
                        print(f"{TAG}: Successfully generated {pte_model_path}")
                        print(f"{TAG}: Will test .pte model for comparison")
                        models_to_test.append(('pte', pte_model_path, False, True))
                    else:
                        print(f"{TAG}: Warning - Failed to generate .pte model")
                except Exception as e:
                    print(f"{TAG}: Warning - Could not generate .pte model: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Store results for all models tested
    all_model_results = {}
    
    for model_label, model_path, is_ptl, is_pte in models_to_test:
        print(f"\n{'='*70}")
        print(f"{TAG}: Testing {model_label.upper()} model: {os.path.basename(model_path)}")
        print(f"{'='*70}\n")
        
        model = load_model_for_testing(model_path, nnModel, framework)
        
        # Skip or exit based on model type and load result
        if model is None:
            if is_pte:
                # PTE models are explicitly requested - don't silently skip
                print(f"\nERROR: {TAG}: Failed to load explicitly requested PTE model!")
                print(f"  PTE file: {model_path}")
                print(f"  This usually indicates a mismatch between:")
                print(f"    1. The ExecuTorch version used to export the .pte file")
                print(f"    2. The ExecuTorch version installed in your venv")
                print(f"")
                print(f"  Solutions:")
                print(f"    a) If you built ExecuTorch from source, rebuild and install in venv:")
                print(f"       cd <executorch_source>")
                print(f"       pip install -e .")
                print(f"    b) Or, re-export the .pte file with current executorch:")
                print(f"       Delete {model_path}")
                print(f"       Re-run with --testPte to regenerate")
                print(f"    c) Or, skip PTE testing (remove --testPte flag)")
                print(f"")
                sys.exit(1)
            else:
                # PTL or PT model failed to load - also exit
                print(f"\nERROR: {TAG}: Failed to load {model_label.upper()} model: {model_path}")
                sys.exit(1)

        # Get prediction probabilities
        print("%s: Calculating Seizure probabilities from test data for %s model" % (TAG, model_label.upper()))
        prediction_proba = predict_model(model, xTest, framework, is_ptl=is_ptl, is_pte=is_pte, test_percent=test_percent)
        if (debug): print("prediction_proba=",prediction_proba)

        # Prediction classes
        prediction = np.argmax(prediction_proba, axis=1)
        if (debug): print("prediction=", prediction)
        
        # Calculate metrics from predictions (more efficient than separate evaluation pass)
        print("Testing using %d seizure datapoints and %d false alarm datapoints"
            % (np.count_nonzero(yTest == 1),
            np.count_nonzero(yTest == 0)))
        
        # Calculate accuracy
        test_acc = np.mean(prediction == yTest)
        
        # Calculate loss (cross-entropy)
        import torch.nn.functional as F
        import torch
        if framework == 'pytorch':
            # Use PyTorch's cross-entropy on the probabilities
            proba_tensor = torch.from_numpy(prediction_proba).float()
            yTest_tensor = torch.from_numpy(yTest).long()
            test_loss = F.cross_entropy(proba_tensor, yTest_tensor).item()
        else:
            # TensorFlow: manual cross-entropy calculation
            epsilon = 1e-7
            yTest_one_hot = np.eye(prediction_proba.shape[1])[yTest]
            test_loss = -np.mean(np.sum(yTest_one_hot * np.log(prediction_proba + epsilon), axis=1))
        
        print(f"{model_label.upper()} Model - Test accuracy: {test_acc:.6f}")
        print(f"{model_label.upper()} Model - Test loss: {test_loss:.6f}")

        # CALCULATE EVENT-LEVEL METRICS FOR THIS MODEL
        # Filter out NaN values (from percentage-based PTE testing)
        pSeizure = prediction_proba[:,1]
        valid_mask = ~np.isnan(pSeizure)
        
        if not valid_mask.all():
            pSeizure_for_calc = pSeizure[valid_mask]
            yTest_for_calc = yTest[valid_mask] if len(yTest.shape) == 1 else yTest[valid_mask]
            prediction_for_calc = prediction[valid_mask]
        else:
            pSeizure_for_calc = pSeizure
            yTest_for_calc = yTest
            prediction_for_calc = prediction
        
        # Calculate predictions at datapoint level for event analysis
        if len(yTest_for_calc.shape) > 1 and yTest_for_calc.shape[1] > 1:
            y_true = np.argmax(yTest_for_calc, axis=1)
        else:
            y_true = yTest_for_calc.flatten()
        y_pred = prediction_for_calc
        
        # Calculate OSD algorithm predictions from dataframe
        # Keep track of original indices for event-level analysis
        if not valid_mask.all():
            df_filtered = df.iloc[valid_mask].copy()
            df_filtered.reset_index(drop=False, inplace=True)  # Keep original index as column
            df_filtered.rename(columns={'index': 'original_index'}, inplace=True)
        else:
            df_filtered = df.copy()
            df_filtered['original_index'] = df_filtered.index
        
        df_filtered['pred'] = y_pred
        df_filtered['osd_pred'] = df_filtered['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)
        
        # For event-level OSD statistics, use the ORIGINAL dataframe (before filtering)
        df_original['osd_pred_orig'] = df_original['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)
        
        # Create mapping of original indices to valid prediction indices
        valid_indices = np.where(valid_mask)[0]  # Original indices with valid predictions
        
        # Add prediction indices to df_filtered for event-level analysis
        df_filtered['pred_index'] = -1  # Mark as untested by default
        for new_idx, orig_idx in enumerate(valid_indices):
            # Find rows in df_filtered that correspond to this original index
            mask_match = df_filtered['original_index'] == orig_idx
            df_filtered.loc[mask_match, 'pred_index'] = new_idx
        
        # Event-level statistics for this model
        event_stats = []
        for eventId, group_orig in df_original.groupby('eventId'):
            true_label = group_orig['type'].iloc[0]
            
            # For OSD event prediction, use the ORIGINAL unfiltered data
            osd_event_pred = 1 if (group_orig['osd_pred_orig'] == 1).any() else 0
            
            # For model predictions, find tested samples for this event
            group_model_all = df_filtered[df_filtered['eventId'] == eventId]
            group_model_tested = group_model_all[group_model_all['pred_index'] >= 0]  # Only tested samples
            
            if len(group_model_tested) > 0:
                # Use predictions from tested samples for this event
                tested_pred_indices = group_model_tested['pred_index'].astype(int).values
                tested_preds = prediction[valid_mask][tested_pred_indices]
                model_event_pred = 1 if (tested_preds == 1).any() else 0

                # Get probabilities for tested samples in this event
                seizure_probs = prediction_proba[valid_mask][tested_pred_indices, 1]
                max_prob = seizure_probs.max()
                event_probs_list = seizure_probs[:50].tolist()

                # Warm-up masking: datapoints whose window still contained
                # pre-fill are excluded from the masked decision (-1 when the
                # event has no non-warm datapoint at all).
                try:
                    warm_full = group_model_tested['is_warm'].values.astype(bool)
                    if warm_full.shape[0] != tested_preds.shape[0]:
                        warm_full = np.zeros(tested_preds.shape, dtype=bool)
                except Exception:
                    warm_full = np.zeros(tested_preds.shape, dtype=bool)
                n_warm_dps = int(warm_full.sum())
                event_warm_list = warm_full[:50].tolist()
                if int((~warm_full).sum()) > 0:
                    model_pred_masked = 1 if (tested_preds[~warm_full] == 1).any() else 0
                    max_prob_masked = float(seizure_probs[~warm_full].max())
                else:
                    model_pred_masked = -1
                    max_prob_masked = 0.0
            else:
                # No tested samples for this event
                model_event_pred = 0
                max_prob = 0.0
                event_probs_list = []
                n_warm_dps = 0
                event_warm_list = []
                model_pred_masked = -1
                max_prob_masked = 0.0

            event_stats.append({
                'eventId': eventId,
                'true_label': true_label,
                'model_pred': model_event_pred,
                'osd_pred': osd_event_pred,
                'max_seizure_prob': max_prob,
                'event_probs_list': event_probs_list,
                'n_warm_dps': n_warm_dps,
                'model_pred_masked': model_pred_masked,
                'max_seizure_prob_masked': max_prob_masked,
                'event_warm_list': event_warm_list
            })
        
        event_stats_df = pd.DataFrame(event_stats)
        
        # Calculate event-level metrics
        event_y_true = event_stats_df['true_label'].values
        event_y_pred_model = event_stats_df['model_pred'].values
        event_y_pred_osd = event_stats_df['osd_pred'].values
        
        event_tpr_model, event_fpr_model = fpr_score(event_y_true, event_y_pred_model)
        event_tpr_osd, event_fpr_osd = fpr_score(event_y_true, event_y_pred_osd)
        
        # Event-level confusion matrix
        event_cm = sklearn.metrics.confusion_matrix(event_y_true, event_y_pred_model, labels=[0, 1])
        event_tn, event_fp, event_fn, event_tp = event_cm.ravel()
        event_accuracy = sklearn.metrics.accuracy_score(event_y_true, event_y_pred_model)

        # Store results for this model including event-level metrics
        all_model_results[model_label] = {
            'model': model,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'prediction_proba': prediction_proba,
            'prediction': prediction,
            'is_ptl': is_ptl,
            'is_pte': is_pte,
            'event_stats_df': event_stats_df,
            'event_cm': event_cm,
            'event_tpr': event_tpr_model,
            'event_fpr': event_fpr_model,
            'event_tpr_osd': event_tpr_osd,
            'event_fpr_osd': event_fpr_osd,
            'event_accuracy': event_accuracy
        }
    
    # Generate outputs for each model variant
    for model_label in all_model_results.keys():
        # Create model-specific output filename
        if model_label == 'pt':
            modelFnameRoot_variant = f"{modelFnameRoot}_pt"
            titlePrefix_variant = f"{titlePrefix} (.pt)"
        elif model_label == 'ptl':
            modelFnameRoot_variant = f"{modelFnameRoot}_ptl"
            titlePrefix_variant = f"{titlePrefix} (.ptl)"
        elif model_label == 'pte':
            modelFnameRoot_variant = f"{modelFnameRoot}_pte"
            titlePrefix_variant = f"{titlePrefix} (.pte)"
        else:
            modelFnameRoot_variant = f"{modelFnameRoot}_{model_label}"
            titlePrefix_variant = f"{titlePrefix} (.{model_label})"
        
        # Get results for this model
        model_results = all_model_results[model_label]
        model = model_results['model']
        test_loss = model_results['test_loss']
        test_acc = model_results['test_acc']
        prediction_proba = model_results['prediction_proba']
        prediction = model_results['prediction']
        is_ptl = model_results['is_ptl']
        is_pte = model_results['is_pte']
        
        # Retrieve event-level metrics calculated in first loop
        event_stats_df = model_results['event_stats_df']
        event_cm = model_results['event_cm']
        event_tpr_model = model_results['event_tpr']
        event_fpr_model = model_results['event_fpr']
        event_tpr_osd = model_results['event_tpr_osd']
        event_fpr_osd = model_results['event_fpr_osd']
        event_accuracy = model_results['event_accuracy']
        
        # Extract event-level predictions for use in visualizations
        event_y_true = event_stats_df['true_label'].values
        event_y_pred_model = event_stats_df['model_pred'].values
        event_y_pred_osd = event_stats_df['osd_pred'].values

        pSeizure = prediction_proba[:,1]
        
        # Filter out NaN values (from percentage-based PTE testing)
        valid_mask = ~np.isnan(pSeizure)
        if not valid_mask.all():
            n_valid = valid_mask.sum()
            n_total = len(pSeizure)
            print(f"{TAG}: Using {n_valid}/{n_total} valid predictions (NaN filtered for {model_label.upper()} model)")
            pSeizure_filtered = pSeizure[valid_mask]
            yTest_filtered = yTest[valid_mask] if len(yTest.shape) == 1 else yTest[valid_mask]
            prediction_filtered = prediction[valid_mask]
        else:
            pSeizure_filtered = pSeizure
            yTest_filtered = yTest
            prediction_filtered = prediction
        
        seq = range(0, len(pSeizure_filtered))
        # Colour seizure data points red, and non-seizure data blue
        colours = ['red' if seizureVal==1 else 'blue' for seizureVal in yTest_filtered]

        # Calculate statistics at different thresholds
        thLst = []
        nTPLst = []
        nFPLst = []
        nTNLst = []
        nFNLst = []
        TPRLst = []
        FPRLst = []

        thresholdLst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for th in thresholdLst:
            nTP, nFP, nTN, nFN = calcTotals(yTest_filtered, pSeizure_filtered, th)
            thLst.append(th)
            nTPLst.append(nTP)
            nFPLst.append(nFP)
            nTNLst.append(nTN)
            nFNLst.append(nFN)
            tp_denom = (nTP + nFN)
            fp_denom = (nFP + nTN)
            TPRLst.append((nTP / tp_denom) if tp_denom > 0 else float('nan'))
            FPRLst.append((nFP / fp_denom) if fp_denom > 0 else float('nan'))

        # Datapoint-level threshold curve (explicitly labeled level)
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(thLst, TPRLst, 'o-', color='green', linewidth=2, markersize=6, label='Datapoint TPR')
        ax.plot(thLst, FPRLst, 's-', color='red', linewidth=2, markersize=6, label='Datapoint FPR')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.set_title(f"{titlePrefix_variant}: Datapoint-Level TPR/FPR vs Threshold", fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        datapoint_threshold_plot = os.path.join(outputDir, f"{modelFnameRoot_variant}_datapoint_threshold_analysis.png")
        fig.savefig(datapoint_threshold_plot, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Create probability scatter plot
        fig, ax = plt.subplots(3,1)
        ax[0].title.set_text("%s: Datapoint-Level Seizure Probabilities" % titlePrefix_variant)
        ax[0].set_ylabel('Probability')
        ax[0].set_xlabel('Datapoint')
        ax[0].scatter(seq, pSeizure_filtered, s=2.0, marker='x', c=colours)
        ax[1].plot(yTest_filtered)
        fname = os.path.join(outputDir, "%s_probabilities.png" % modelFnameRoot_variant)
        fig.savefig(fname)
        plt.close()

        # Calculate and save confusion matrix and detailed statistics
        # Pass pre-computed predictions to avoid re-running inference
        # Use filtered data to exclude NaN values from percentage-based PTE testing
        xTest_filtered = xTest[valid_mask] if not valid_mask.all() else xTest
        calcConfusionMatrix(configObj, modelFnameRoot_variant, xTest_filtered, yTest_filtered, dataDir=outputDir, 
                           balanced=balanced, debug=debug, titlePrefix=titlePrefix_variant,
                           prediction_proba=prediction_proba[valid_mask] if not valid_mask.all() else prediction_proba, 
                           prediction=prediction_filtered)

        # Calculate epoch-level statistics for this model
        # Use filtered data to handle NaN values from percentage-based PTE testing
        if len(yTest_filtered.shape) > 1 and yTest_filtered.shape[1] > 1:
            y_true = np.argmax(yTest_filtered, axis=1)
        else:
            y_true = yTest_filtered.flatten()
        y_pred = prediction_filtered

        # Epoch-level confusion matrix and metrics
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        tpr, fpr = fpr_score(y_true, y_pred)
        
        # Calculate OSD algorithm predictions from dataframe  
        # Need to filter df to match the valid predictions for datapoint-level comparison
        if not valid_mask.all():
            df_filtered = df.iloc[valid_mask].copy()
        else:
            df_filtered = df.copy()

        # Production-mode datapoint predictions: require 3 consecutive datapoints >= threshold
        prod_threshold = 0.5
        prod_pred = np.zeros(len(pSeizure_filtered), dtype=int)
        if len(pSeizure_filtered) == len(df_filtered):
            for _, idx in df_filtered.groupby('eventId', sort=False).groups.items():
                idx_arr = np.asarray(list(idx), dtype=int)
                prod_pred[idx_arr] = _three_consecutive_predictions(
                    pSeizure_filtered[idx_arr],
                    threshold=prod_threshold,
                    consecutive_required=3,
                )
        else:
            prod_pred = _three_consecutive_predictions(pSeizure_filtered, threshold=prod_threshold, consecutive_required=3)
        
        df_filtered['osd_pred'] = df_filtered['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)
        yPredOsd = df_filtered['osd_pred'].values
        yTestOsd = y_true
        
        tprOsd, fprOsd = fpr_score(yTestOsd, yPredOsd)
        cmOsd = sklearn.metrics.confusion_matrix(yTestOsd, yPredOsd, labels=[0, 1])
        tnOsd, fpOsd, fnOsd, tpOsd = cmOsd.ravel()
        accuracyOsd = sklearn.metrics.accuracy_score(yTestOsd, yPredOsd)

        # Production-mode datapoint metrics for model
        prod_tpr, prod_fpr = fpr_score(y_true, prod_pred)
        prod_cm = sklearn.metrics.confusion_matrix(y_true, prod_pred, labels=[0, 1])
        prod_tn, prod_fp, prod_fn, prod_tp = prod_cm.ravel()
        prod_accuracy = sklearn.metrics.accuracy_score(y_true, prod_pred)

        # Production-mode event metrics for model from event probability sequences
        prod_event_pred = event_stats_df['event_probs_list'].apply(
            lambda probs: _event_positive_from_probs(probs, prod_threshold, mode='production', consecutive_required=3)
        ).astype(int).values
        prod_event_tpr, prod_event_fpr = fpr_score(event_y_true, prod_event_pred)
        prod_event_cm = sklearn.metrics.confusion_matrix(event_y_true, prod_event_pred, labels=[0, 1])
        prod_event_tn, prod_event_fp, prod_event_fn, prod_event_tp = prod_event_cm.ravel()
        prod_event_accuracy = sklearn.metrics.accuracy_score(event_y_true, prod_event_pred)

        # Warm-up-masked event metrics: same rules, ignoring datapoints whose
        # window still contained buffer pre-fill (segment starts / post-gap).
        # Decisions use the first 50 datapoints per event (pre-existing limit).
        masked_event_m = _masked_event_metrics(
            event_y_true, event_stats_df['model_pred_masked'].values)
        prod_event_pred_masked = _prod_masked_preds(
            event_stats_df, prod_threshold, consecutive_required=3)
        masked_prod_m = _masked_event_metrics(event_y_true, prod_event_pred_masked)
        n_warm_dps_total = int(event_stats_df['n_warm_dps'].sum())
        
        # Plot event-level confusion matrix
        import seaborn as sns
        LABELS = ['Non-Seizure', 'Seizure']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Model confusion matrix
        sns.heatmap(event_cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                   linewidths=0.1, fmt="d", cmap='YlGnBu', ax=ax1)
        ax1.set_title(f"{titlePrefix_variant}\nEvent-Level Confusion Matrix (Model)", fontsize=12)
        ax1.set_ylabel('True label')
        ax1.set_xlabel('Predicted label')
        
        # OSD confusion matrix
        event_cm_osd = sklearn.metrics.confusion_matrix(event_y_true, event_y_pred_osd, labels=[0, 1])
        sns.heatmap(event_cm_osd, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                   linewidths=0.1, fmt="d", cmap='Oranges', ax=ax2)
        ax2.set_title(f"{titlePrefix_variant}\nEvent-Level Confusion Matrix (OSD)", fontsize=12)
        ax2.set_ylabel('True label')
        ax2.set_xlabel('Predicted label')
        
        plt.tight_layout()
        event_cm_fname = os.path.join(outputDir, f"{modelFnameRoot_variant}_event_confusion.png")
        plt.savefig(event_cm_fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"{TAG}: Saved event-level confusion matrix to {event_cm_fname}")
        
        # Save event-level statistics to CSV (standardised format)
        event_stats_csv = os.path.join(outputDir, f"{modelFnameRoot_variant}_event_results.csv")
        event_stats_df.to_csv(event_stats_csv, index=False)
        print(f"{TAG}: Saved event-level statistics to {event_stats_csv}")
        
        # Save comprehensive statistics summary for this variant
        stats_summary_file = os.path.join(outputDir, f"{modelFnameRoot_variant}_summary.txt")
        with open(stats_summary_file, 'w') as f:
            f.write(f"="*70 + "\n")
            f.write(f"{titlePrefix_variant} - Comprehensive Statistics Summary\n")
            f.write(f"="*70 + "\n\n")
            f.write(_format_subtype_weighting_summary(configObj))
            
            f.write("*** EVENT-LEVEL STATISTICS (MOST IMPORTANT) ***\n")
            f.write("="*70 + "\n")
            f.write(f"Total Events: {len(event_stats_df)}\n")
            f.write(f"True Seizure Events: {(event_y_true == 1).sum()}\n")
            f.write(f"True Non-Seizure Events: {(event_y_true == 0).sum()}\n\n")
            
            f.write(f"Model - Event-Level Metrics:\n")
            f.write(f"  Accuracy: {event_accuracy:.4f}\n")
            f.write(f"  Sensitivity/TPR: {event_tpr_model:.4f}\n")
            f.write(f"  FPR: {event_fpr_model:.4f}\n")
            f.write(f"  TP={event_tp}, FP={event_fp}, TN={event_tn}, FN={event_fn}\n")
            f.write(f"  Detected Seizures: {(event_y_pred_model == 1).sum()}/{(event_y_true == 1).sum()}\n\n")

            f.write(f"Model - Event-Level Metrics (Production Rule: 3 consecutive >= 0.5):\n")
            f.write(f"  Accuracy: {prod_event_accuracy:.4f}\n")
            f.write(f"  Sensitivity/TPR: {prod_event_tpr:.4f}\n")
            f.write(f"  FPR: {prod_event_fpr:.4f}\n")
            f.write(f"  TP={prod_event_tp}, FP={prod_event_fp}, TN={prod_event_tn}, FN={prod_event_fn}\n\n")

            f.write(f"Model - Event-Level Metrics, warm-up masked (same rules, ignoring\n")
            f.write(f"  datapoints whose window still contained buffer pre-fill):\n")
            f.write(f"  Warm datapoints: {n_warm_dps_total} "
                    f"(events with no non-warm datapoint excluded: {masked_event_m['n_excluded']})\n")
            f.write(f"  Event rule - Sensitivity/TPR: {masked_event_m['tpr']:.4f}\n")
            f.write(f"  Event rule - FPR: {masked_event_m['fpr']:.4f}\n")
            f.write(f"  Event rule - TP={masked_event_m['tp']}, FP={masked_event_m['fp']}, "
                    f"TN={masked_event_m['tn']}, FN={masked_event_m['fn']}\n")
            f.write(f"  Production rule - Sensitivity/TPR: {masked_prod_m['tpr']:.4f}\n")
            f.write(f"  Production rule - FPR: {masked_prod_m['fpr']:.4f}\n")
            f.write(f"  Production rule - TP={masked_prod_m['tp']}, FP={masked_prod_m['fp']}, "
                    f"TN={masked_prod_m['tn']}, FN={masked_prod_m['fn']}\n\n")
            
            f.write(f"OSD Algorithm - Event-Level Metrics:\n")
            f.write(f"  Sensitivity/TPR: {event_tpr_osd:.4f}\n")
            f.write(f"  FPR: {event_fpr_osd:.4f}\n")
            f.write(f"  Detected Seizures: {(event_y_pred_osd == 1).sum()}/{(event_y_true == 1).sum()}\n\n")
            
            f.write("\nDATAPOINT-LEVEL STATISTICS (for reference):\n")
            f.write("-"*70 + "\n")
            f.write(f"Model - Accuracy: {accuracy:.4f}\n")
            f.write(f"Model - Sensitivity/TPR: {tpr:.4f}\n")
            f.write(f"Model - FPR: {fpr:.4f}\n")
            f.write(f"Model - TP={tp}, FP={fp}, TN={tn}, FN={fn}\n\n")

            f.write(f"Model (Production Rule: 3 consecutive >= 0.5) - Accuracy: {prod_accuracy:.4f}\n")
            f.write(f"Model (Production Rule: 3 consecutive >= 0.5) - Sensitivity/TPR: {prod_tpr:.4f}\n")
            f.write(f"Model (Production Rule: 3 consecutive >= 0.5) - FPR: {prod_fpr:.4f}\n")
            f.write(f"Model (Production Rule: 3 consecutive >= 0.5) - TP={prod_tp}, FP={prod_fp}, TN={prod_tn}, FN={prod_fn}\n\n")
            
            f.write(f"OSD Algorithm - Accuracy: {accuracyOsd:.4f}\n")
            f.write(f"OSD Algorithm - Sensitivity/TPR: {tprOsd:.4f}\n")
            f.write(f"OSD Algorithm - FPR: {fprOsd:.4f}\n")
            f.write(f"OSD Algorithm - TP={tpOsd}, FP={fpOsd}, TN={tnOsd}, FN={fnOsd}\n")
            f.write("="*70 + "\n")
        
        print(f"{TAG}: Saved summary statistics to {stats_summary_file}")
        
        print(f"\n{TAG}: {model_label.upper()} Model Event-Level Statistics:")
        print(f"  Datapoint - Sensitivity (TPR): {tpr:.4f}, False Alarm Rate (FPR): {fpr:.4f}")
        print(f"  Datapoint - TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"  Datapoint Production (3-consecutive) - Sensitivity (TPR): {prod_tpr:.4f}, False Alarm Rate (FPR): {prod_fpr:.4f}")
        print(f"  Event - Sensitivity (TPR): {event_tpr_model:.4f}, False Alarm Rate (FPR): {event_fpr_model:.4f}")
        print(f"  Event Production (3-consecutive) - Sensitivity (TPR): {prod_event_tpr:.4f}, False Alarm Rate (FPR): {prod_event_fpr:.4f}")
        print(f"  Event (warm-up masked) - Sensitivity (TPR): {masked_event_m['tpr']:.4f}, False Alarm Rate (FPR): {masked_event_m['fpr']:.4f} "
              f"(TP={masked_event_m['tp']}, FP={masked_event_m['fp']}, excluded={masked_event_m['n_excluded']})")
        print(f"  Event Production warm-up masked - Sensitivity (TPR): {masked_prod_m['tpr']:.4f}, False Alarm Rate (FPR): {masked_prod_m['fpr']:.4f} "
              f"(TP={masked_prod_m['tp']}, FP={masked_prod_m['fp']}, excluded={masked_prod_m['n_excluded']})")
    
    # Create side-by-side comparison plots for all tested models
    if len(all_model_results) > 1:
        print(f"\n{TAG}: Creating comparison plots for {len(all_model_results)} model variants...")
        import seaborn as sns
        
        # Collect confusion matrices for all models (both datapoint and event level)
        model_cms = {}
        model_event_cms = {}
        for model_label in all_model_results.keys():
            results = all_model_results[model_label]
            pred = results['prediction']
            
            # Datapoint-level confusion matrix
            valid_mask_local = ~np.isnan(results['prediction_proba'][:,1])
            y_true_local = yTest[valid_mask_local] if len(yTest.shape) == 1 else yTest[valid_mask_local].flatten()
            pred_local = pred[valid_mask_local] if not valid_mask_local.all() else pred
            
            cm = sklearn.metrics.confusion_matrix(y_true_local, pred_local, labels=[0, 1])
            model_cms[model_label] = cm
            
            # Event-level confusion matrix
            model_event_cms[model_label] = results['event_cm']
        
        # Create side-by-side EVENT-LEVEL confusion matrix comparison (most important)
        n_models = len(model_event_cms)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        LABELS = ['Non-Seizure', 'Seizure']
        for idx, (model_label, cm) in enumerate(model_event_cms.items()):
            ax = axes[idx]
            model_name = model_label.upper()
            
            sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                       linewidths=0.1, fmt="d", cmap='YlGnBu', ax=ax)
            ax.set_title(f"{titlePrefix} ({model_name})\nEvent-Level", fontsize=12)
            ax.set_ylabel('True label' if idx == 0 else '')
            ax.set_xlabel('Predicted label')
        
        plt.tight_layout()
        event_comparison_fname = os.path.join(outputDir, f"{modelFnameRoot}_event_comparison_confusion.png")
        plt.savefig(event_comparison_fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"{TAG}: Saved EVENT-LEVEL comparison confusion matrix to {event_comparison_fname}")
        
        # Create side-by-side DATAPOINT-LEVEL confusion matrix comparison (for reference)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        LABELS = ['No-Alarm', 'Seizure']
        for idx, (model_label, cm) in enumerate(model_cms.items()):
            ax = axes[idx]
            model_name = model_label.upper()
            
            sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                       linewidths=0.1, fmt="d", cmap='YlGnBu', ax=ax)
            ax.set_title(f"{titlePrefix} ({model_name})\nDatapoint-Level", fontsize=12)
            ax.set_ylabel('True label' if idx == 0 else '')
            ax.set_xlabel('Predicted label')
        
        plt.tight_layout()
        comparison_fname = os.path.join(outputDir, f"{modelFnameRoot}_datapoint_comparison_confusion.png")
        plt.savefig(comparison_fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"{TAG}: Saved datapoint-level comparison confusion matrix to {comparison_fname}")
        
        # Create comparison summary text file with EVENT-LEVEL emphasis
        comparison_summary = os.path.join(outputDir, f"{modelFnameRoot}_comparison_summary.txt")
        with open(comparison_summary, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL VARIANT COMPARISON SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("*** EVENT-LEVEL COMPARISON (MOST IMPORTANT) ***\n")
            f.write("="*80 + "\n")
            for model_label in all_model_results.keys():
                results = all_model_results[model_label]
                event_cm = results['event_cm']
                event_tn, event_fp, event_fn, event_tp = event_cm.ravel()
                
                f.write(f"\n{model_label.upper()} Model - Event-Level:\n")
                f.write("-"*40 + "\n")
                f.write(f"  Accuracy: {results['event_accuracy']:.4f}\n")
                f.write(f"  Sensitivity/TPR: {results['event_tpr']:.4f}\n")
                f.write(f"  FPR: {results['event_fpr']:.4f}\n")
                f.write(f"  Confusion Matrix: TP={event_tp}, FP={event_fp}, TN={event_tn}, FN={event_fn}\n")
                
                event_stats_df = results['event_stats_df']
                n_events = len(event_stats_df)
                n_seizures = (event_stats_df['true_label'] == 1).sum()
                detected = (event_stats_df['model_pred'] == 1).sum()
                f.write(f"  Total Events: {n_events} ({n_seizures} seizures)\n")
                f.write(f"  Detected as Seizure: {detected}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("\nDatapoint-Level Comparison (for reference):\n")
            f.write("="*80 + "\n")
            for model_label in all_model_results.keys():
                results = all_model_results[model_label]
                valid_mask_local = ~np.isnan(results['prediction_proba'][:,1])
                y_true_local = yTest[valid_mask_local] if len(yTest.shape) == 1 else yTest[valid_mask_local].flatten()
                pred_local = results['prediction'][valid_mask_local] if not valid_mask_local.all() else results['prediction']
                
                cm = model_cms[model_label]
                tn, fp, fn, tp = cm.ravel()
                acc = sklearn.metrics.accuracy_score(y_true_local, pred_local)
                tpr_local, fpr_local = fpr_score(y_true_local, pred_local)
                
                f.write(f"\n{model_label.upper()} Model - Datapoint-Level:\n")
                f.write("-"*40 + "\n")
                f.write(f"  Test Accuracy: {results['test_acc']:.6f}\n")
                f.write(f"  Test Loss: {results['test_loss']:.6f}\n")
                f.write(f"  Sensitivity/TPR: {tpr_local:.4f}\n")
                f.write(f"  FPR: {fpr_local:.4f}\n")
                f.write(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}\n")
                f.write(f"  Samples tested: {valid_mask_local.sum()}/{len(yTest)}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"{TAG}: Saved comparison summary to {comparison_summary}")
    
    # Use the first model (.pt) for backward compatibility (original analysis flow)
    model = all_model_results['pt']['model']
    test_loss = all_model_results['pt']['test_loss']
    test_acc = all_model_results['pt']['test_acc']
    prediction_proba = all_model_results['pt']['prediction_proba']
    prediction = all_model_results['pt']['prediction']

    # Calculate epoch-level statistics
    # Check if yTest is one-hot encoded (2D) or class indices (1D)
    if len(yTest.shape) > 1 and yTest.shape[1] > 1:
        y_true = np.argmax(yTest, axis=1)
    else:
        y_true = yTest.flatten()
    y_pred = prediction
    
    # Epoch-level confusion matrix and metrics
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    tpr, fpr = fpr_score(y_true, y_pred)
    
    # Calculate OSD algorithm predictions from dataframe
    df['pred'] = y_pred
    df['osd_pred'] = df['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)
    yPredOsd = df['osd_pred'].values
    yTestOsd = y_true
    
    tprOsd, fprOsd = fpr_score(yTestOsd, yPredOsd)
    cmOsd = sklearn.metrics.confusion_matrix(yTestOsd, yPredOsd, labels=[0, 1])
    tnOsd, fpOsd, fnOsd, tpOsd = cmOsd.ravel()
    accuracyOsd = sklearn.metrics.accuracy_score(yTestOsd, yPredOsd)
    
    # For event-level OSD statistics, use the ORIGINAL dataframe (before filtering)
    # This is important because dp2vector may filter out datapoints that contain OSD alarms
    # We still use the filtered df for model predictions since those require valid dp2vector output
    df_original['osd_pred_orig'] = df_original['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)
    
    # Event-level statistics - iterate over ALL events in original data to ensure we don't miss any
    # This is critical because some events may be completely filtered out by dp2vector
    event_stats = []
    for eventId, group_orig in df_original.groupby('eventId'):
        true_label = group_orig['type'].iloc[0]
        
        # For OSD event prediction, use the ORIGINAL unfiltered data
        osd_event_pred = 1 if (group_orig['osd_pred_orig'] == 1).any() else 0
        
        # For model predictions, use the filtered data (if this event exists in filtered df)
        group_filtered = df[df['eventId'] == eventId]
        if len(group_filtered) > 0:
            model_event_pred = 1 if (group_filtered['pred'] == 1).any() else 0
            # Calculate max seizure probability for this event using the correct row indices
            # We need to map back to the kept_indices to index into prediction_proba correctly
            seizure_probs = prediction_proba[group_filtered.index, 1]
            max_prob = seizure_probs.max()

            # Store probabilities as a list (to be expanded into separate columns later)
            # Limit to first 50 datapoints to avoid excessive columns
            event_probs_list = seizure_probs[:50].tolist()

            # Warm-up masking (same convention as the per-variant block above):
            # -1 sentinel when the event has no non-warm datapoint at all.
            try:
                warm_full = group_filtered['is_warm'].values.astype(bool)
                if warm_full.shape[0] != seizure_probs.shape[0]:
                    warm_full = np.zeros(seizure_probs.shape, dtype=bool)
            except Exception:
                warm_full = np.zeros(seizure_probs.shape, dtype=bool)
            n_warm_dps = int(warm_full.sum())
            event_warm_list = warm_full[:50].tolist()
            if int((~warm_full).sum()) > 0:
                model_pred_masked = 1 if (seizure_probs[~warm_full] >= 0.5).any() else 0
                max_prob_masked = float(seizure_probs[~warm_full].max())
            else:
                model_pred_masked = -1
                max_prob_masked = 0.0
        else:
            # This event was completely filtered out - model cannot make a prediction
            model_event_pred = 0
            max_prob = 0.0
            event_probs_list = []
            n_warm_dps = 0
            event_warm_list = []
            model_pred_masked = -1
            max_prob_masked = 0.0
        
        # Debug event-level OSD predictions
        if debug and true_label == 1:  # Print for seizure events
            n_osd_alarms_original = (group_orig['osd_pred_orig'] == 1).sum()
            n_osd_alarms_filtered = (group_filtered['osd_pred'] == 1).sum() if len(group_filtered) > 0 else 0
            print(f"{TAG}: Event {eventId} (seizure): model_pred={model_event_pred}, " +
                  f"osd_pred={osd_event_pred} (filtered:{n_osd_alarms_filtered}/{len(group_filtered)}, original:{n_osd_alarms_original}/{len(group_orig)})")
        
        event_stats.append({
            'eventId': eventId,
            'true_label': true_label,
            'model_pred': model_event_pred,
            'osd_pred': osd_event_pred,
            'max_seizure_prob': max_prob,
            'event_probs_list': event_probs_list,
            'n_warm_dps': n_warm_dps,
            'model_pred_masked': model_pred_masked,
            'max_seizure_prob_masked': max_prob_masked,
            'event_warm_list': event_warm_list
        })
    event_stats_df = pd.DataFrame(event_stats)
    
    # Determine the maximum number of datapoints in any event (capped at 50)
    max_datapoints = min(max([len(probs) for probs in event_stats_df['event_probs_list']], default=0), 50)
    
    # Expand event probabilities into individual columns (dp0, dp1, ..., dpN)
    # Pad with 0.0 for events with fewer datapoints
    for dp_idx in range(max_datapoints):
        col_name = f'dp{dp_idx}'
        event_stats_df[col_name] = event_stats_df['event_probs_list'].apply(
            lambda probs: probs[dp_idx] if dp_idx < len(probs) else 0.0
        )
    
    # Keep event_probs_list for threshold and production-style analyses.
    
    # Load event metadata from allData.json for additional details
    # allData.json is at the training output root, not in the fold subdirectory
    # Search up the directory tree to find it
    allDataFilename = configObj['dataFileNames']['allDataFileJson']
    allDataPath = None
    
    # First try in the current dataDir
    candidate_path = os.path.join(dataDir, allDataFilename)
    if os.path.exists(candidate_path):
        allDataPath = candidate_path
    else:
        # Search up the directory tree (for fold subdirectories)
        current_dir = dataDir
        for _ in range(5):  # Search up to 5 levels up
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached root directory
                break
            candidate_path = os.path.join(parent_dir, allDataFilename)
            if os.path.exists(candidate_path):
                allDataPath = candidate_path
                break
            current_dir = parent_dir
    
    event_details_map = {}
    
    if allDataPath and os.path.exists(allDataPath):
        print("%s: Loading event details from %s" % (TAG, allDataPath))
        try:
            with open(allDataPath, 'r') as f:
                allData = json.load(f)
            
            # Build a map of eventId -> event details
            print(f"{TAG}: Parsing event details from allData")
            events_list = allData if isinstance(allData, list) else allData.get('events', [])
            print(f"{TAG}: Found {len(events_list)} events in allData")
            print(f"{TAG}: Event keys are: {list(events_list[0].keys()) if len(events_list) > 0 else 'N/A'} ")
            for event in events_list:
                # Convert event ID to string for consistent type matching
                event_id_str = str(event['id'])
                event_details_map[event_id_str] = {
                    'userId': event.get('userId', 'N/A'),
                    'typeStr': event.get('type', 'N/A'),
                    'subType': event.get('subType', 'N/A'),
                    'desc': event.get('desc', 'N/A'),
                    # Needed for alarm-latency: seizure start =
                    # event dataTime + seizureTimes[0] (see flattenData.py
                    # seizureTimes semantics: offsets in seconds).
                    'dataTime': event.get('dataTime', None),
                    'seizureTimes': event.get('seizureTimes', None),
                }
            print(f"{TAG}: Loaded metadata for {len(event_details_map)} events from {allDataPath}")
        except Exception as e:
            print(f"{TAG}: Warning - Could not load event details from {allDataPath}: {e}")
    else:
        print(f"{TAG}: Warning - allData file not found (searched in {dataDir} and parent directories)")
    
    # Enrich event_stats_df with metadata - convert eventId to string for consistent lookup
    event_stats_df['userId'] = event_stats_df['eventId'].map(lambda eid: event_details_map.get(str(eid), {}).get('userId', 'N/A'))
    event_stats_df['typeStr'] = event_stats_df['eventId'].map(lambda eid: event_details_map.get(str(eid), {}).get('typeStr', 'N/A'))
    event_stats_df['subType'] = event_stats_df['eventId'].map(lambda eid: event_details_map.get(str(eid), {}).get('subType', 'N/A'))
    event_stats_df['desc'] = event_stats_df['eventId'].map(lambda eid: event_details_map.get(str(eid), {}).get('desc', 'N/A'))
    
    # Save detailed event results to CSV with datapoint probability columns
    # This code is used for BOTH inner fold cross-validation and outer fold independent testing
    # Both code paths call testModel() with identical datapoint probability generation
    # Build column list: base columns + datapoint probability columns (dp0, dp1, ..., dpN)
    base_cols = ['eventId', 'userId', 'typeStr', 'subType', 'true_label',
                 'model_pred', 'osd_pred', 'max_seizure_prob',
                 'n_warm_dps', 'model_pred_masked', 'max_seizure_prob_masked']
    datapoint_cols = [f'dp{i}' for i in range(max_datapoints)]
    base_col_names = ['EventID', 'UserID', 'Type', 'SubType', 'ActualLabel',
                      'ModelPrediction', 'OSDPrediction', 'MaxSeizureProbability',
                      'NWarmDps', 'ModelPredictionMasked', 'MaxProbMasked']
    datapoint_col_names = [f'dp{i}' for i in range(max_datapoints)]
    
    event_results_csv = event_stats_df[base_cols + datapoint_cols + ['desc']].copy()
    event_results_csv.columns = base_col_names + datapoint_col_names + ['Description']
    
    csv_path = os.path.join(outputDir, f'{modelFnameRoot}_event_results.csv')
    event_results_csv.to_csv(csv_path, index=False)
    print(f"{TAG}: Event-level results saved to {csv_path}")

    interesting_events_path = os.path.join(outputDir, f'{modelFnameRoot}_interesting_events.csv')
    interesting_events_df = _export_interesting_events(
        event_results_csv,
        interesting_events_path,
        prod_threshold=0.5,
        top_k_per_category=40,
    )
    if len(interesting_events_df) > 0:
        print(f"{TAG}: Interesting events list saved to {interesting_events_path} ({len(interesting_events_df)} rows)")
    else:
        print(f"{TAG}: No interesting events were exported (insufficient datapoint traces)")
    
    # Event-level metrics
    event_tpr, event_fpr = fpr_score(event_stats_df['true_label'], event_stats_df['model_pred'])
    event_cm = sklearn.metrics.confusion_matrix(event_stats_df['true_label'], event_stats_df['model_pred'], labels=[0, 1])
    event_tn, event_fp, event_fn, event_tp = event_cm.ravel()
    
    osd_event_tpr, osd_event_fpr = fpr_score(event_stats_df['true_label'], event_stats_df['osd_pred'])
    osd_event_cm = sklearn.metrics.confusion_matrix(event_stats_df['true_label'], event_stats_df['osd_pred'], labels=[0, 1])
    osd_event_tn, osd_event_fp, osd_event_fn, osd_event_tp = osd_event_cm.ravel()

    # Production-style model metrics (3 consecutive datapoints >= threshold)
    prod_threshold = 0.5
    prod_pred_dp = np.zeros(len(y_true), dtype=int)
    p_seizure_all = prediction_proba[:, 1]
    valid_dp_mask = ~np.isnan(p_seizure_all)
    if valid_dp_mask.all():
        df_for_prod = df.copy().reset_index(drop=True)
        p_for_prod = p_seizure_all
        y_true_for_prod = y_true
    else:
        df_for_prod = df.iloc[valid_dp_mask].copy().reset_index(drop=True)
        p_for_prod = p_seizure_all[valid_dp_mask]
        y_true_for_prod = y_true[valid_dp_mask]
        prod_pred_dp = np.zeros(len(y_true_for_prod), dtype=int)

    if len(df_for_prod) == len(p_for_prod):
        for _, idx in df_for_prod.groupby('eventId', sort=False).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            prod_pred_dp[idx_arr] = _three_consecutive_predictions(
                p_for_prod[idx_arr],
                threshold=prod_threshold,
                consecutive_required=3,
            )
    else:
        prod_pred_dp = _three_consecutive_predictions(p_for_prod, threshold=prod_threshold, consecutive_required=3)

    prod_tpr_dp, prod_fpr_dp = fpr_score(y_true_for_prod, prod_pred_dp)
    prod_cm_dp = sklearn.metrics.confusion_matrix(y_true_for_prod, prod_pred_dp, labels=[0, 1])
    prod_tn_dp, prod_fp_dp, prod_fn_dp, prod_tp_dp = prod_cm_dp.ravel()
    prod_accuracy_dp = sklearn.metrics.accuracy_score(y_true_for_prod, prod_pred_dp)

    # Production-style event predictions from per-event probability traces
    prod_event_pred = event_stats_df['event_probs_list'].apply(
        lambda probs: _event_positive_from_probs(probs, prod_threshold, mode='production', consecutive_required=3)
    ).astype(int).values
    prod_event_tpr, prod_event_fpr = fpr_score(event_stats_df['true_label'].values, prod_event_pred)
    prod_event_cm = sklearn.metrics.confusion_matrix(event_stats_df['true_label'].values, prod_event_pred, labels=[0, 1])
    prod_event_tn, prod_event_fp, prod_event_fn, prod_event_tp = prod_event_cm.ravel()
    prod_event_accuracy = sklearn.metrics.accuracy_score(event_stats_df['true_label'].values, prod_event_pred)

    # Warm-up-masked operating-point metrics (threshold 0.5): same event and
    # production rules, ignoring datapoints whose window still contained
    # buffer pre-fill. Events with no non-warm datapoint are excluded.
    masked_event_m = _masked_event_metrics(
        event_stats_df['true_label'].values,
        event_stats_df['model_pred_masked'].values)
    prod_event_pred_masked = _prod_masked_preds(
        event_stats_df, prod_threshold, consecutive_required=3)
    masked_prod_m = _masked_event_metrics(
        event_stats_df['true_label'].values, prod_event_pred_masked)
    n_warm_dps_total = int(event_stats_df['n_warm_dps'].sum())

    # Tonic-clonic mask for subtype-specific threshold analysis
    tc_positive_mask = (
        (event_stats_df['true_label'] == 1) &
        event_stats_df['subType'].astype(str).str.contains('tonic-clonic', case=False, na=False)
    ).values
    tc_count = int(tc_positive_mask.sum())

    # NDA negative mask: events whose typeStr is 'nda' (case-insensitive) – for real-world FAR
    try:
        if 'typeStr' in event_stats_df.columns:
            _nda_series = event_stats_df['typeStr']
        elif 'Type' in event_stats_df.columns:
            _nda_series = event_stats_df['Type']
        else:
            _nda_series = event_stats_df['true_label'].astype(str)  # fallback empty
        nda_negative_mask = _is_nda_series(_nda_series).values & (event_stats_df['true_label'].values == 0)
    except Exception:
        nda_negative_mask = np.zeros(len(event_stats_df), dtype=bool)
    nda_count = int(nda_negative_mask.sum())
    # NDA event-level FAR at current 0.5 threshold (both decision rules)
    # Production event prediction already computed above as prod_event_pred
    try:
        _nda_event_pred_at_thr = event_stats_df['model_pred'].values[nda_negative_mask]
        _nda_prod_pred_at_thr = prod_event_pred[nda_negative_mask]
        nda_event_fp = int((_nda_event_pred_at_thr == 1).sum())
        nda_event_tn = int((_nda_event_pred_at_thr == 0).sum())
        nda_prod_fp = int((_nda_prod_pred_at_thr == 1).sum())
        nda_prod_tn = int((_nda_prod_pred_at_thr == 0).sum())
        nda_event_fpr = nda_event_fp / (nda_event_fp + nda_event_tn) if (nda_event_fp + nda_event_tn) > 0 else 0.0
        nda_prod_fpr = nda_prod_fp / (nda_prod_fp + nda_prod_tn) if (nda_prod_fp + nda_prod_tn) > 0 else 0.0
    except Exception:
        nda_event_fp = nda_event_tn = nda_prod_fp = nda_prod_tn = 0
        nda_event_fpr = nda_prod_fpr = 0.0
    nda_event_fa_per_day = _fa_per_day(nda_event_fpr)
    nda_prod_fa_per_day = _fa_per_day(nda_prod_fpr)
    # Warm-up-masked NDA FAR at 0.5 (same masks/exclusions as above).
    try:
        _mpm = event_stats_df['model_pred_masked'].values
        _nda_mm = _masked_event_metrics(
            event_stats_df['true_label'].values[nda_negative_mask],
            _mpm[nda_negative_mask])
        _nda_ppm = _prod_masked_preds(event_stats_df, prod_threshold,
                                      consecutive_required=3)[nda_negative_mask]
        _nda_pm = _masked_event_metrics(
            event_stats_df['true_label'].values[nda_negative_mask], _nda_ppm)
        nda_event_fp_m = int(_nda_mm['fp'])
        nda_event_tn_m = int(_nda_mm['tn'])
        nda_prod_fp_m = int(_nda_pm['fp'])
        nda_prod_tn_m = int(_nda_pm['tn'])
        nda_event_fpr_m = float(_nda_mm['fpr'])
        nda_prod_fpr_m = float(_nda_pm['fpr'])
        nda_event_excl_m = int(_nda_mm['n_excluded'])
        nda_prod_excl_m = int(_nda_pm['n_excluded'])
    except Exception:
        nda_event_fp_m = nda_event_tn_m = nda_prod_fp_m = nda_prod_tn_m = 0
        nda_event_fpr_m = nda_prod_fpr_m = 0.0
        nda_event_excl_m = nda_prod_excl_m = 0
    nda_event_fa_per_day_m = _fa_per_day(nda_event_fpr_m)
    nda_prod_fa_per_day_m = _fa_per_day(nda_prod_fpr_m)
    
    # Debug: Print OSD event-level predictions summary
    if debug:
        print(f"\n{TAG}: === OSD EVENT-LEVEL DEBUG ===")
        print(f"{TAG}: Total events: {len(event_stats_df)}")
        print(f"{TAG}: Seizure events (true_label=1): {(event_stats_df['true_label']==1).sum()}")
        print(f"{TAG}: Events with OSD prediction=1: {(event_stats_df['osd_pred']==1).sum()}")
        print(f"{TAG}: OSD Event TP={osd_event_tp}, FP={osd_event_fp}, FN={osd_event_fn}, TN={osd_event_tn}")
        print(f"{TAG}: OSD Event TPR={osd_event_tpr:.3f}, FPR={osd_event_fpr:.3f}")
        if osd_event_tp == 0 and (event_stats_df['true_label']==1).sum() > 0:
            print(f"{TAG}: WARNING: OSD detected 0 seizure events despite having seizure events in test set!")
            print(f"{TAG}: This may indicate that OSD alarms were filtered out during dp2vector processing.")
    
    # Convert NumPy scalars to native Python types
    def py(v):
        return v.item() if hasattr(v, 'item') else v
    
    # Build results dictionary
    num_positive_epoch = int((y_true == 1).sum())
    num_positive_event = int((event_stats_df['true_label'] == 1).sum())
    
    foldResults = {
        'num_positive_epoch': num_positive_epoch,
        'num_positive_event': num_positive_event,
        'num_positive_tc_event': tc_count,
        'accuracy': py(accuracy),
        'accuracyOsd': py(accuracyOsd),
        'tpr': py(tpr),
        'fpr': py(fpr),
        'prod_tpr_dp': py(prod_tpr_dp),
        'prod_fpr_dp': py(prod_fpr_dp),
        'tprOsd': py(tprOsd),
        'fprOsd': py(fprOsd),
        'tn': py(tn),
        'fp': py(fp),
        'fn': py(fn),
        'tp': py(tp),
        'prod_tn_dp': py(prod_tn_dp),
        'prod_fp_dp': py(prod_fp_dp),
        'prod_fn_dp': py(prod_fn_dp),
        'prod_tp_dp': py(prod_tp_dp),
        'tnOsd': py(tnOsd),
        'fpOsd': py(fpOsd),
        'fnOsd': py(fnOsd),
        'tpOsd': py(tpOsd),
        'event_tpr': py(event_tpr),
        'event_fpr': py(event_fpr),
        'event_tp': py(event_tp),
        'event_fp': py(event_fp),
        'event_fn': py(event_fn),
        'event_tn': py(event_tn),
        'prod_event_tpr': py(prod_event_tpr),
        'prod_event_fpr': py(prod_event_fpr),
        'prod_event_tp': py(prod_event_tp),
        'prod_event_fp': py(prod_event_fp),
        'prod_event_fn': py(prod_event_fn),
        'prod_event_tn': py(prod_event_tn),
        'osd_event_tpr': py(osd_event_tpr),
        'osd_event_fpr': py(osd_event_fpr),
        'osd_event_tp': py(osd_event_tp),
        'osd_event_fp': py(osd_event_fp),
        'osd_event_fn': py(osd_event_fn),
        'osd_event_tn': py(osd_event_tn),
        # NDA-only FAR – real-world false alarm rate on normal activity (3-min events)
        'nda_count': int(nda_count),
        'nda_event_fpr': py(nda_event_fpr),
        'nda_event_fp': int(nda_event_fp),
        'nda_event_tn': int(nda_event_tn),
        'nda_event_fa_per_day': py(nda_event_fa_per_day),
        'nda_prod_event_fpr': py(nda_prod_fpr),
        'nda_prod_event_fp': int(nda_prod_fp),
        'nda_prod_event_tn': int(nda_prod_tn),
        'nda_prod_event_fa_per_day': py(nda_prod_fa_per_day),
        # Warm-up-masked event metrics (threshold 0.5; warm-up datapoints
        # excluded from decisions; events with no non-warm datapoint excluded).
        'n_warm_dps_total': int(n_warm_dps_total),
        'n_events_excluded_masked': int(masked_event_m['n_excluded']),
        'event_tpr_masked': py(masked_event_m['tpr']),
        'event_fpr_masked': py(masked_event_m['fpr']),
        'event_tp_masked': int(masked_event_m['tp']),
        'event_fp_masked': int(masked_event_m['fp']),
        'event_tn_masked': int(masked_event_m['tn']),
        'event_fn_masked': int(masked_event_m['fn']),
        'prod_event_tpr_masked': py(masked_prod_m['tpr']),
        'prod_event_fpr_masked': py(masked_prod_m['fpr']),
        'prod_event_tp_masked': int(masked_prod_m['tp']),
        'prod_event_fp_masked': int(masked_prod_m['fp']),
        'prod_event_tn_masked': int(masked_prod_m['tn']),
        'prod_event_fn_masked': int(masked_prod_m['fn']),
        'n_events_excluded_prod_masked': int(masked_prod_m['n_excluded']),
        'nda_event_fpr_masked': py(nda_event_fpr_m),
        'nda_event_fp_masked': int(nda_event_fp_m),
        'nda_event_tn_masked': int(nda_event_tn_m),
        'nda_event_fa_per_day_masked': py(nda_event_fa_per_day_m),
        'nda_event_excluded_masked': int(nda_event_excl_m),
        'nda_prod_event_fpr_masked': py(nda_prod_fpr_m),
        'nda_prod_event_fp_masked': int(nda_prod_fp_m),
        'nda_prod_event_tn_masked': int(nda_prod_tn_m),
        'nda_prod_event_fa_per_day_masked': py(nda_prod_fa_per_day_m),
        'nda_prod_event_excluded_masked': int(nda_prod_excl_m)
    }
    
    # Save to JSON
    json_path = os.path.join(outputDir, 'testResults.json')
    with open(json_path, 'w') as f:
        json.dump(foldResults, f, indent=2)
    print(f"nnTester: foldResults written to {json_path}")
    
    # Echo formatted results to console
    print("\n===== Formatted foldResults =====")
    print(json.dumps(foldResults, indent=2))
    print("===== End foldResults =====\n")
    
    # Print detailed event-based metrics summary
    print("\n" + "="*70)
    print("EVENT-BASED ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nTotal Events: {len(event_stats_df)}")
    print(f"  Seizure Events: {num_positive_event}")
    print(f"  Non-Seizure Events: {len(event_stats_df) - num_positive_event}")
    
    print(f"\n{'METRIC':<30} {'MODEL':<15} {'OSD ALGORITHM':<15}")
    print("-" * 70)
    print(f"{'True Positives (TP)':<30} {py(event_tp):<15} {py(osd_event_tp):<15}")
    print(f"{'False Positives (FP)':<30} {py(event_fp):<15} {py(osd_event_fp):<15}")
    print(f"{'True Negatives (TN)':<30} {py(event_tn):<15} {py(osd_event_tn):<15}")
    print(f"{'False Negatives (FN)':<30} {py(event_fn):<15} {py(osd_event_fn):<15}")
    print("-" * 70)
    print(f"{'Sensitivity (TPR)':<30} {py(event_tpr):.4f}{'':<10} {py(osd_event_tpr):.4f}{'':<10}")
    print(f"{'False Alarm Rate (FAR/FPR)':<30} {py(event_fpr):.4f}{'':<10} {py(osd_event_fpr):.4f}{'':<10}")
    print(f"{'Production TPR (3-consecutive)':<30} {py(prod_event_tpr):.4f}{'':<10} {'N/A':<15}")
    print(f"{'Production FPR (3-consecutive)':<30} {py(prod_event_fpr):.4f}{'':<10} {'N/A':<15}")
    print(f"{'NDA FAR (event, 3-min)':<30} {nda_event_fpr:.4f} ({nda_event_fa_per_day:.2f} FA/day){'':<2} {'N/A':<15}")
    print(f"{'NDA FAR (prod, 3-min)':<30} {nda_prod_fpr:.4f} ({nda_prod_fa_per_day:.2f} FA/day){'':<2} {'N/A':<15}")
    print(f"{'Event TPR warm-up masked':<30} {masked_event_m['tpr']:.4f}{'':<10} {'':<15}")
    print(f"{'Event FPR warm-up masked':<30} {masked_event_m['fpr']:.4f} (TP={masked_event_m['tp']}, FP={masked_event_m['fp']}, excluded={masked_event_m['n_excluded']})")
    print(f"{'Production TPR masked':<30} {masked_prod_m['tpr']:.4f}{'':<10} {'':<15}")
    print(f"{'Production FPR masked':<30} {masked_prod_m['fpr']:.4f} (TP={masked_prod_m['tp']}, FP={masked_prod_m['fp']}, excluded={masked_prod_m['n_excluded']})")
    print(f"{'NDA FAR (event) masked':<30} {nda_event_fpr_m:.4f} ({nda_event_fa_per_day_m:.2f} FA/day)")
    print(f"{'NDA FAR (prod) masked':<30} {nda_prod_fpr_m:.4f} ({nda_prod_fa_per_day_m:.2f} FA/day)")
    print(f"Warm-up datapoints (excluded from masked decisions): {n_warm_dps_total}")
    
    # Calculate additional event-based metrics
    event_precision = event_tp / (event_tp + event_fp) if (event_tp + event_fp) > 0 else 0
    event_specificity = event_tn / (event_tn + event_fp) if (event_tn + event_fp) > 0 else 0
    event_f1 = 2 * event_tp / (2 * event_tp + event_fp + event_fn) if (2 * event_tp + event_fp + event_fn) > 0 else 0
    
    osd_event_precision = osd_event_tp / (osd_event_tp + osd_event_fp) if (osd_event_tp + osd_event_fp) > 0 else 0
    osd_event_specificity = osd_event_tn / (osd_event_tn + osd_event_fp) if (osd_event_tn + osd_event_fp) > 0 else 0
    osd_event_f1 = 2 * osd_event_tp / (2 * osd_event_tp + osd_event_fp + osd_event_fn) if (2 * osd_event_tp + osd_event_fp + osd_event_fn) > 0 else 0
    
    print(f"{'Precision (PPV)':<30} {event_precision:.4f}{'':<10} {osd_event_precision:.4f}{'':<10}")
    print(f"{'Specificity (TNR)':<30} {event_specificity:.4f}{'':<10} {osd_event_specificity:.4f}{'':<10}")
    print(f"{'F1 Score':<30} {event_f1:.4f}{'':<10} {osd_event_f1:.4f}{'':<10}")
    print("="*70)

    print("\n" + "="*70)
    print("DATAPOINT-LEVEL PRODUCTION METRICS (3 CONSECUTIVE DATAPOINTS)")
    print("="*70)
    print(f"Threshold: {prod_threshold:.2f}")
    print(f"Accuracy: {prod_accuracy_dp:.4f}")
    print(f"Sensitivity (TPR): {prod_tpr_dp:.4f}")
    print(f"False Alarm Rate (FPR): {prod_fpr_dp:.4f}")
    print(f"TP={prod_tp_dp}, FP={prod_fp_dp}, TN={prod_tn_dp}, FN={prod_fn_dp}")
    print("="*70)

    print("\n" + "="*70)
    print("NDA-ONLY REAL-WORLD FAR (3-min NDA events, 480/day)")
    print("="*70)
    print(f"NDA events in test set: {nda_count} (≈ {nda_count * NDA_EVENT_DURATION_MIN:.1f} min, {nda_count/NDA_EVENTS_PER_DAY:.2f} days equivalent)")
    print(f"Event-level NDA FAR: {nda_event_fpr:.4f}  →  {nda_event_fa_per_day:.2f} FA/day  (FP={nda_event_fp}, TN={nda_event_tn})")
    print(f"Production (3-consecutive) NDA FAR: {nda_prod_fpr:.4f}  →  {nda_prod_fa_per_day:.2f} FA/day  (FP={nda_prod_fp}, TN={nda_prod_tn})")
    print(f"For comparison, all non-seizure FAR (event): {py(event_fpr):.4f}, production: {py(prod_event_fpr):.4f}")
    print("="*70)
    
    # Threshold analyses: event-level and production-level, all seizures and tonic-clonic subset
    print("\n" + "="*70)
    print("THRESHOLD ANALYSIS (EVENT VS PRODUCTION)")
    print("="*70)

    event_threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    event_probs_list = event_stats_df['event_probs_list'].tolist()
    event_true_labels = event_stats_df['true_label'].values
    # Warm-up masks aligned with event_probs_list (first-50 truncation): used
    # to compute masked threshold curves alongside the standard ones.
    try:
        event_warm_masks = event_stats_df['event_warm_list'].tolist()
    except Exception:
        event_warm_masks = None

    threshold_data_event_all = _threshold_metrics_from_event_probs(
        event_probs_list,
        event_true_labels,
        event_threshold_list,
        mode='event',
        warm_masks=event_warm_masks,
    )
    threshold_data_prod_all = _threshold_metrics_from_event_probs(
        event_probs_list,
        event_true_labels,
        event_threshold_list,
        mode='production',
        consecutive_required=3,
        warm_masks=event_warm_masks,
    )

    threshold_data_event_tc = _threshold_metrics_from_event_probs(
        event_probs_list,
        event_true_labels,
        event_threshold_list,
        mode='event',
        positive_mask=tc_positive_mask,
        warm_masks=event_warm_masks,
    )
    threshold_data_prod_tc = _threshold_metrics_from_event_probs(
        event_probs_list,
        event_true_labels,
        event_threshold_list,
        mode='production',
        positive_mask=tc_positive_mask,
        consecutive_required=3,
        warm_masks=event_warm_masks,
    )

    # NDA-only FAR threshold curves: same TPR (seizures) but FPR computed on NDA events only
    threshold_data_event_nda = _threshold_metrics_from_event_probs(
        event_probs_list,
        event_true_labels,
        event_threshold_list,
        mode='event',
        negative_mask=nda_negative_mask,
        warm_masks=event_warm_masks,
    )
    threshold_data_prod_nda = _threshold_metrics_from_event_probs(
        event_probs_list,
        event_true_labels,
        event_threshold_list,
        mode='production',
        negative_mask=nda_negative_mask,
        consecutive_required=3,
        warm_masks=event_warm_masks,
    )

    print("\nAll-seizure event-level threshold analysis")
    print(f"{'Threshold':<12} {'TPR':<12} {'FPR':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-" * 70)
    for i, th in enumerate(event_threshold_list):
        print(f"{th:<12.1f} {threshold_data_event_all['tpr'][i]:<12.4f} {threshold_data_event_all['fpr'][i]:<12.4f} "
              f"{threshold_data_event_all['tp'][i]:<8} {threshold_data_event_all['fp'][i]:<8} "
              f"{threshold_data_event_all['tn'][i]:<8} {threshold_data_event_all['fn'][i]:<8}")

    print("\nAll-seizure production-level threshold analysis (3 consecutive datapoints)")
    print(f"{'Threshold':<12} {'TPR':<12} {'FPR':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-" * 70)
    for i, th in enumerate(event_threshold_list):
        print(f"{th:<12.1f} {threshold_data_prod_all['tpr'][i]:<12.4f} {threshold_data_prod_all['fpr'][i]:<12.4f} "
              f"{threshold_data_prod_all['tp'][i]:<8} {threshold_data_prod_all['fp'][i]:<8} "
              f"{threshold_data_prod_all['tn'][i]:<8} {threshold_data_prod_all['fn'][i]:<8}")

    print("\nTonic-clonic event-level threshold analysis")
    print(f"Tonic-clonic positives in test set: {tc_count}")
    print(f"{'Threshold':<12} {'TPR_TC':<12} {'FPR':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN_TC':<8}")
    print("-" * 70)
    for i, th in enumerate(event_threshold_list):
        print(f"{th:<12.1f} {threshold_data_event_tc['tpr'][i]:<12.4f} {threshold_data_event_tc['fpr'][i]:<12.4f} "
              f"{threshold_data_event_tc['tp'][i]:<8} {threshold_data_event_tc['fp'][i]:<8} "
              f"{threshold_data_event_tc['tn'][i]:<8} {threshold_data_event_tc['fn'][i]:<8}")

    print("\nNDA-only FAR threshold analysis (TPR on seizures, FAR on NDA events only)")
    print(f"NDA events in test set: {nda_count} (≈ {nda_count * NDA_EVENT_DURATION_MIN:.1f} min, {nda_count/NDA_EVENTS_PER_DAY:.2f} days equivalent)")
    print(f"{'Threshold':<12} {'TPR':<12} {'FPR_NDA':<12} {'FA/day':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-" * 80)
    for i, th in enumerate(event_threshold_list):
        fa_day_e = _fa_per_day(threshold_data_event_nda['fpr'][i])
        print(f"{th:<12.1f} {threshold_data_event_nda['tpr'][i]:<12.4f} {threshold_data_event_nda['fpr'][i]:<12.4f} {fa_day_e:<12.1f} "
              f"{threshold_data_event_nda['tp'][i]:<8} {threshold_data_event_nda['fp'][i]:<8} "
              f"{threshold_data_event_nda['tn'][i]:<8} {threshold_data_event_nda['fn'][i]:<8}")
    print("\nNDA production-level threshold analysis (3 consecutive)")
    print(f"{'Threshold':<12} {'TPR':<12} {'FPR_NDA':<12} {'FA/day':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-" * 80)
    for i, th in enumerate(event_threshold_list):
        fa_day_p = _fa_per_day(threshold_data_prod_nda['fpr'][i])
        print(f"{th:<12.1f} {threshold_data_prod_nda['tpr'][i]:<12.4f} {threshold_data_prod_nda['fpr'][i]:<12.4f} {fa_day_p:<12.1f} "
              f"{threshold_data_prod_nda['tp'][i]:<8} {threshold_data_prod_nda['fp'][i]:<8} "
              f"{threshold_data_prod_nda['tn'][i]:<8} {threshold_data_prod_nda['fn'][i]:<8}")

    if event_warm_masks is not None and 'tpr_masked' in threshold_data_event_all:
        print("\nWarm-up-masked threshold analysis (same rules, warm-up datapoints excluded "
              "from decisions; events with no non-warm datapoint excluded)")
        print(f"Excluded events (warm-only): {threshold_data_event_all.get('n_excluded_warm_only', 0)}")
        print(f"{'Threshold':<12} {'TPR_ev':<12} {'FPR_ev':<12} {'TPR_prod':<12} {'FPR_prod':<12}")
        print("-" * 62)
        for i, th in enumerate(event_threshold_list):
            print(f"{th:<12.1f} "
                  f"{threshold_data_event_all['tpr_masked'][i]:<12.4f} "
                  f"{threshold_data_event_all['fpr_masked'][i]:<12.4f} "
                  f"{threshold_data_prod_all['tpr_masked'][i]:<12.4f} "
                  f"{threshold_data_prod_all['fpr_masked'][i]:<12.4f}")

    # Save plots with explicit level naming
    threshold_plot_path_event = os.path.join(outputDir, f'{modelFnameRoot}_event_threshold_analysis.png')
    _plot_threshold_analysis(threshold_data_event_all, threshold_plot_path_event, titlePrefix, 'Event-Level (all seizures)')
    print(f"\n{TAG}: Event-level threshold analysis plot saved to {threshold_plot_path_event}")

    threshold_plot_path_prod = os.path.join(outputDir, f'{modelFnameRoot}_production_threshold_analysis.png')
    _plot_threshold_analysis(threshold_data_prod_all, threshold_plot_path_prod, titlePrefix, 'Production-Level (3 consecutive datapoints)')
    print(f"{TAG}: Production-level threshold analysis plot saved to {threshold_plot_path_prod}")

    threshold_plot_path_ev_vs_prod = os.path.join(outputDir, f'{modelFnameRoot}_event_vs_production_threshold_analysis.png')
    _plot_event_vs_production_thresholds(threshold_data_event_all, threshold_data_prod_all, threshold_plot_path_ev_vs_prod, titlePrefix,
                                         nda_event_data=threshold_data_event_nda, nda_production_data=threshold_data_prod_nda)
    print(f"{TAG}: Event-vs-production threshold comparison plot saved to {threshold_plot_path_ev_vs_prod} (includes NDA FAR)")

    # Dedicated NDA FAR comparison plot
    threshold_plot_path_nda = os.path.join(outputDir, f'{modelFnameRoot}_nda_threshold_analysis.png')
    _plot_nda_fa_threshold(threshold_data_event_nda, threshold_data_prod_nda, threshold_plot_path_nda, titlePrefix)
    print(f"{TAG}: NDA FAR threshold analysis plot saved to {threshold_plot_path_nda}")

    threshold_plot_path_event_tc = os.path.join(outputDir, f'{modelFnameRoot}_event_threshold_analysis_tonic_clonic.png')
    _plot_threshold_analysis(threshold_data_event_tc, threshold_plot_path_event_tc, titlePrefix, 'Event-Level (tonic-clonic seizures)')
    print(f"{TAG}: Tonic-clonic event-level threshold plot saved to {threshold_plot_path_event_tc}")

    threshold_plot_path_prod_tc = os.path.join(outputDir, f'{modelFnameRoot}_production_threshold_analysis_tonic_clonic.png')
    _plot_threshold_analysis(threshold_data_prod_tc, threshold_plot_path_prod_tc, titlePrefix, 'Production-Level (tonic-clonic seizures)')
    print(f"{TAG}: Tonic-clonic production-level threshold plot saved to {threshold_plot_path_prod_tc}")

    threshold_plot_path_ev_vs_prod_tc = os.path.join(outputDir, f'{modelFnameRoot}_event_vs_production_threshold_analysis_tonic_clonic.png')
    _plot_event_vs_production_thresholds(threshold_data_event_tc, threshold_data_prod_tc, threshold_plot_path_ev_vs_prod_tc, titlePrefix)
    print(f"{TAG}: Tonic-clonic event-vs-production threshold comparison plot saved to {threshold_plot_path_ev_vs_prod_tc}")

    # Save threshold analysis data to JSON
    threshold_json_path_event = os.path.join(outputDir, f'{modelFnameRoot}_event_threshold_data.json')
    with open(threshold_json_path_event, 'w') as f:
        json.dump(threshold_data_event_all, f, indent=2)
    print(f"{TAG}: Event-level threshold data saved to {threshold_json_path_event}")

    threshold_json_path_prod = os.path.join(outputDir, f'{modelFnameRoot}_production_threshold_data.json')
    with open(threshold_json_path_prod, 'w') as f:
        json.dump(threshold_data_prod_all, f, indent=2)
    print(f"{TAG}: Production-level threshold data saved to {threshold_json_path_prod}")

    threshold_json_path_event_tc = os.path.join(outputDir, f'{modelFnameRoot}_event_threshold_data_tonic_clonic.json')
    with open(threshold_json_path_event_tc, 'w') as f:
        json.dump(threshold_data_event_tc, f, indent=2)
    print(f"{TAG}: Tonic-clonic event threshold data saved to {threshold_json_path_event_tc}")

    threshold_json_path_prod_tc = os.path.join(outputDir, f'{modelFnameRoot}_production_threshold_data_tonic_clonic.json')
    with open(threshold_json_path_prod_tc, 'w') as f:
        json.dump(threshold_data_prod_tc, f, indent=2)
    print(f"{TAG}: Tonic-clonic production threshold data saved to {threshold_json_path_prod_tc}")

    threshold_json_path_event_nda = os.path.join(outputDir, f'{modelFnameRoot}_event_threshold_data_nda.json')
    with open(threshold_json_path_event_nda, 'w') as f:
        json.dump(threshold_data_event_nda, f, indent=2)
    print(f"{TAG}: NDA event threshold data saved to {threshold_json_path_event_nda}")

    threshold_json_path_prod_nda = os.path.join(outputDir, f'{modelFnameRoot}_production_threshold_data_nda.json')
    with open(threshold_json_path_prod_nda, 'w') as f:
        json.dump(threshold_data_prod_nda, f, indent=2)
    print(f"{TAG}: NDA production threshold data saved to {threshold_json_path_prod_nda}")

    # ---- Alarm latency analysis ----
    # latency = first datapoint dataTime with p(seizure) >= threshold
    #           minus seizure start (event dataTime + seizureTimes[0]).
    # Computed over the same threshold range as above, for all seizures
    # and the tonic-clonic subset. See flattenData.py for seizureTimes
    # semantics (offsets in seconds relative to the event reference time).
    print("\n" + "="*70)
    print("ALARM LATENCY ANALYSIS (vs seizure start = event dataTime + seizureTimes[0])")
    print("="*70)
    try:
        latency_data, latency_per_event_df = _compute_alarm_latency(
            event_stats_df, df, prediction_proba, event_details_map,
            event_threshold_list, debug=debug)

        def _fmt_lat(mean, std, n):
            if n == 0 or mean is None or (isinstance(mean, float) and np.isnan(mean)):
                return "n/a (n=0)"
            return f"{mean:.1f} ± {std:.1f}s (n={n})"

        print(f"{'Threshold':<12} {'All seizures':<28} {'Tonic-clonic':<28}")
        print("-" * 70)
        for i, th in enumerate(latency_data['thresholds']):
            m_a = latency_data['all']['mean'][i]
            s_a = latency_data['all']['std'][i]
            n_a = latency_data['all']['n_detected'][i]
            m_t = latency_data['tonic_clonic']['mean'][i]
            s_t = latency_data['tonic_clonic']['std'][i]
            n_t = latency_data['tonic_clonic']['n_detected'][i]
            print(f"{th:<12.1f} {_fmt_lat(m_a, s_a, n_a):<28} {_fmt_lat(m_t, s_t, n_t):<28}")
        print(f"Seizure events: {latency_data['all']['n_total']} all "
              f"({latency_data['all']['n_with_onset']} with known onset), "
              f"{latency_data['tonic_clonic']['n_total']} tonic-clonic "
              f"({latency_data['tonic_clonic']['n_with_onset']} with known onset)")
        print("Note: negative latency = alarm before annotated seizure start; "
              "stats over detected events with known onset only.")

        latency_json_path = os.path.join(outputDir, f'{modelFnameRoot}_latency_data.json')
        with open(latency_json_path, 'w') as f:
            json.dump(latency_data, f, indent=2)
        print(f"{TAG}: Alarm latency data saved to {latency_json_path}")

        latency_csv_path = os.path.join(outputDir, f'{modelFnameRoot}_latency_per_event.csv')
        latency_per_event_df.to_csv(latency_csv_path, index=False)
        print(f"{TAG}: Per-event alarm latencies saved to {latency_csv_path}")

        latency_plot_path = os.path.join(outputDir, f'{modelFnameRoot}_latency_vs_threshold.png')
        _plot_latency_vs_threshold(latency_data, latency_plot_path, titlePrefix)
        print(f"{TAG}: Alarm latency vs threshold plot saved to {latency_plot_path}")
    except Exception as e:
        print(f"{TAG}: Warning - alarm latency analysis failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()

    print("="*70)
    
    print("\nEvent-Level Confusion Matrix (Model):")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {py(event_tn):<10}        {py(event_fp):<10}")
    print(f"Actual Positive        {py(event_fn):<10}        {py(event_tp):<10}")
    
    print("\nEvent-Level Confusion Matrix (OSD Algorithm):")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {py(osd_event_tn):<10}        {py(osd_event_fp):<10}")
    print(f"Actual Positive        {py(osd_event_fn):<10}        {py(osd_event_tp):<10}")
    print("="*70 + "\n")
    
    # Plot event-based confusion matrices
    import seaborn as sns
    LABELS = ['Non-Seizure', 'Seizure']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Model event-level confusion matrix
    sns.heatmap(event_cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                linewidths=0.1, fmt="d", cmap='YlGnBu', ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f"{titlePrefix}: Event-Level Confusion Matrix\n(Model)", fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    
    # Add metrics text below model confusion matrix
    model_text = f"Sensitivity: {py(event_tpr):.3f}  Specificity: {event_specificity:.3f}\n"
    model_text += f"Precision: {event_precision:.3f}  F1: {event_f1:.3f}"
    ax1.text(0.5, -0.15, model_text, ha='center', va='top', transform=ax1.transAxes,
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # OSD algorithm event-level confusion matrix
    sns.heatmap(osd_event_cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                linewidths=0.1, fmt="d", cmap='OrRd', ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_title(f"{titlePrefix}: Event-Level Confusion Matrix\n(OSD Algorithm)", fontsize=13, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=11)
    ax2.set_xlabel('Predicted Label', fontsize=11)
    
    # Add metrics text below OSD confusion matrix
    osd_text = f"Sensitivity: {py(osd_event_tpr):.3f}  Specificity: {osd_event_specificity:.3f}\n"
    osd_text += f"Precision: {osd_event_precision:.3f}  F1: {osd_event_f1:.3f}"
    ax2.text(0.5, -0.15, osd_text, ha='center', va='top', transform=ax2.transAxes,
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    fname_event_cm = os.path.join(outputDir, f"{modelFnameRoot}_event_confusion.png")
    plt.savefig(fname_event_cm, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Event-level confusion matrices saved as {fname_event_cm}")
    
    # If both .pt and .ptl models were tested, generate comparison plots and statistics
    if len(all_model_results) > 1 and 'ptl' in all_model_results:
        print(f"\n{'='*70}")
        print(f"{TAG}: GENERATING COMPARISON BETWEEN .PT AND .PTL MODELS")
        print(f"{'='*70}\n")
        
        # Calculate statistics for both models
        model_comparison = {}
        for model_label in ['pt', 'ptl']:
            pred_proba = all_model_results[model_label]['prediction_proba']
            pred = all_model_results[model_label]['prediction']
            
            # Epoch-level metrics
            if len(yTest.shape) > 1 and yTest.shape[1] > 1:
                y_true_model = np.argmax(yTest, axis=1)
            else:
                y_true_model = yTest.flatten()
            
            cm_model = sklearn.metrics.confusion_matrix(y_true_model, pred, labels=[0, 1])
            tn_m, fp_m, fn_m, tp_m = cm_model.ravel()
            acc_m = sklearn.metrics.accuracy_score(y_true_model, pred)
            tpr_m, fpr_m = fpr_score(y_true_model, pred)
            
            # Event-level metrics (using same event_stats_df logic)
            df_temp = df.copy()
            df_temp['pred_model'] = pred
            event_stats_model = []
            for eventId, group_orig in df_original.groupby('eventId'):
                true_label_ev = group_orig['type'].iloc[0]
                group_filtered_temp = df_temp[df_temp['eventId'] == eventId]
                if len(group_filtered_temp) > 0:
                    model_event_pred_temp = 1 if (group_filtered_temp['pred_model'] == 1).any() else 0
                    max_prob_temp = pred_proba[group_filtered_temp.index, 1].max()
                else:
                    model_event_pred_temp = 0
                    max_prob_temp = 0.0
                event_stats_model.append({
                    'eventId': eventId,
                    'true_label': true_label_ev,
                    'model_pred': model_event_pred_temp,
                    'max_seizure_prob': max_prob_temp
                })
            event_stats_df_model = pd.DataFrame(event_stats_model)
            
            event_tpr_m, event_fpr_m = fpr_score(event_stats_df_model['true_label'], event_stats_df_model['model_pred'])
            event_cm_m = sklearn.metrics.confusion_matrix(event_stats_df_model['true_label'], event_stats_df_model['model_pred'], labels=[0, 1])
            event_tn_m, event_fp_m, event_fn_m, event_tp_m = event_cm_m.ravel()
            
            model_comparison[model_label] = {
                'test_loss': all_model_results[model_label]['test_loss'],
                'test_acc': all_model_results[model_label]['test_acc'],
                'epoch_acc': acc_m,
                'epoch_tpr': tpr_m,
                'epoch_fpr': fpr_m,
                'epoch_tp': tp_m,
                'epoch_fp': fp_m,
                'epoch_tn': tn_m,
                'epoch_fn': fn_m,
                'event_tpr': event_tpr_m,
                'event_fpr': event_fpr_m,
                'event_tp': event_tp_m,
                'event_fp': event_fp_m,
                'event_tn': event_tn_m,
                'event_fn': event_fn_m,
                'event_cm': event_cm_m
            }
        
        # Print comparison table
        print("\n" + "="*80)
        print("MODEL COMPARISON: .PT vs .PTL")
        print("="*80)
        print(f"{'METRIC':<35} {'.PT MODEL':<20} {'.PTL MODEL':<20}")
        print("-" * 80)
        print(f"{'Test Loss':<35} {model_comparison['pt']['test_loss']:<20.6f} {model_comparison['ptl']['test_loss']:<20.6f}")
        print(f"{'Test Accuracy':<35} {model_comparison['pt']['test_acc']:<20.6f} {model_comparison['ptl']['test_acc']:<20.6f}")
        print("-" * 80)
        print("EPOCH-LEVEL METRICS:")
        print(f"{'  Accuracy':<35} {model_comparison['pt']['epoch_acc']:<20.6f} {model_comparison['ptl']['epoch_acc']:<20.6f}")
        print(f"{'  Sensitivity (TPR)':<35} {model_comparison['pt']['epoch_tpr']:<20.6f} {model_comparison['ptl']['epoch_tpr']:<20.6f}")
        print(f"{'  False Alarm Rate (FPR)':<35} {model_comparison['pt']['epoch_fpr']:<20.6f} {model_comparison['ptl']['epoch_fpr']:<20.6f}")
        print(f"{'  True Positives (TP)':<35} {model_comparison['pt']['epoch_tp']:<20} {model_comparison['ptl']['epoch_tp']:<20}")
        print(f"{'  False Positives (FP)':<35} {model_comparison['pt']['epoch_fp']:<20} {model_comparison['ptl']['epoch_fp']:<20}")
        print(f"{'  True Negatives (TN)':<35} {model_comparison['pt']['epoch_tn']:<20} {model_comparison['ptl']['epoch_tn']:<20}")
        print(f"{'  False Negatives (FN)':<35} {model_comparison['pt']['epoch_fn']:<20} {model_comparison['ptl']['epoch_fn']:<20}")
        print("-" * 80)
        print("EVENT-LEVEL METRICS:")
        print(f"{'  Sensitivity (TPR)':<35} {model_comparison['pt']['event_tpr']:<20.6f} {model_comparison['ptl']['event_tpr']:<20.6f}")
        print(f"{'  False Alarm Rate (FPR)':<35} {model_comparison['pt']['event_fpr']:<20.6f} {model_comparison['ptl']['event_fpr']:<20.6f}")
        print(f"{'  True Positives (TP)':<35} {model_comparison['pt']['event_tp']:<20} {model_comparison['ptl']['event_tp']:<20}")
        print(f"{'  False Positives (FP)':<35} {model_comparison['pt']['event_fp']:<20} {model_comparison['ptl']['event_fp']:<20}")
        print(f"{'  True Negatives (TN)':<35} {model_comparison['pt']['event_tn']:<20} {model_comparison['ptl']['event_tn']:<20}")
        print(f"{'  False Negatives (FN)':<35} {model_comparison['pt']['event_fn']:<20} {model_comparison['ptl']['event_fn']:<20}")
        print("="*80 + "\n")
        
        # Save comparison to JSON
        comparison_json = {
            'pt_model': {k: py(v) if hasattr(v, 'item') else (v.tolist() if isinstance(v, np.ndarray) else v) 
                        for k, v in model_comparison['pt'].items() if k != 'event_cm'},
            'ptl_model': {k: py(v) if hasattr(v, 'item') else (v.tolist() if isinstance(v, np.ndarray) else v) 
                         for k, v in model_comparison['ptl'].items() if k != 'event_cm'}
        }
        comparison_json_path = os.path.join(outputDir, f'{modelFnameRoot}_pt_vs_ptl_comparison.json')
        with open(comparison_json_path, 'w') as f:
            json.dump(comparison_json, f, indent=2)
        print(f"{TAG}: Model comparison saved to {comparison_json_path}")
        
        # Generate comparison confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # PT model event-level confusion matrix
        sns.heatmap(model_comparison['pt']['event_cm'], xticklabels=LABELS, yticklabels=LABELS, annot=True,
                    linewidths=0.1, fmt="d", cmap='YlGnBu', ax=axes[0, 0], cbar_kws={'label': 'Count'})
        axes[0, 0].set_title(f"{titlePrefix}.pt: Event-Level", fontsize=13, fontweight='bold')
        axes[0, 0].set_ylabel('True Label', fontsize=11)
        axes[0, 0].set_xlabel('Predicted Label', fontsize=11)
        pt_text = f"TPR: {model_comparison['pt']['event_tpr']:.3f}  FPR: {model_comparison['pt']['event_fpr']:.3f}"
        axes[0, 0].text(0.5, -0.12, pt_text, ha='center', va='top', transform=axes[0, 0].transAxes,
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # PTL model event-level confusion matrix
        sns.heatmap(model_comparison['ptl']['event_cm'], xticklabels=LABELS, yticklabels=LABELS, annot=True,
                    linewidths=0.1, fmt="d", cmap='YlOrRd', ax=axes[0, 1], cbar_kws={'label': 'Count'})
        axes[0, 1].set_title(f"{titlePrefix}.ptl: Event-Level", fontsize=13, fontweight='bold')
        axes[0, 1].set_ylabel('True Label', fontsize=11)
        axes[0, 1].set_xlabel('Predicted Label', fontsize=11)
        ptl_text = f"TPR: {model_comparison['ptl']['event_tpr']:.3f}  FPR: {model_comparison['ptl']['event_fpr']:.3f}"
        axes[0, 1].text(0.5, -0.12, ptl_text, ha='center', va='top', transform=axes[0, 1].transAxes,
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        # PT model epoch-level confusion matrix
        pt_epoch_cm = np.array([[model_comparison['pt']['epoch_tn'], model_comparison['pt']['epoch_fp']],
                                [model_comparison['pt']['epoch_fn'], model_comparison['pt']['epoch_tp']]])
        sns.heatmap(pt_epoch_cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                    linewidths=0.1, fmt="d", cmap='YlGnBu', ax=axes[1, 0], cbar_kws={'label': 'Count'})
        axes[1, 0].set_title(f"{titlePrefix}.pt: Datapoint-Level", fontsize=13, fontweight='bold')
        axes[1, 0].set_ylabel('True Label', fontsize=11)
        axes[1, 0].set_xlabel('Predicted Label', fontsize=11)
        pt_epoch_text = f"TPR: {model_comparison['pt']['epoch_tpr']:.3f}  FPR: {model_comparison['pt']['epoch_fpr']:.3f}"
        axes[1, 0].text(0.5, -0.12, pt_epoch_text, ha='center', va='top', transform=axes[1, 0].transAxes,
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # PTL model epoch-level confusion matrix
        ptl_epoch_cm = np.array([[model_comparison['ptl']['epoch_tn'], model_comparison['ptl']['epoch_fp']],
                                 [model_comparison['ptl']['epoch_fn'], model_comparison['ptl']['epoch_tp']]])
        sns.heatmap(ptl_epoch_cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                    linewidths=0.1, fmt="d", cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Count'})
        axes[1, 1].set_title(f"{titlePrefix}.ptl: Datapoint-Level", fontsize=13, fontweight='bold')
        axes[1, 1].set_ylabel('True Label', fontsize=11)
        axes[1, 1].set_xlabel('Predicted Label', fontsize=11)
        ptl_epoch_text = f"TPR: {model_comparison['ptl']['epoch_tpr']:.3f}  FPR: {model_comparison['ptl']['epoch_fpr']:.3f}"
        axes[1, 1].text(0.5, -0.12, ptl_epoch_text, ha='center', va='top', transform=axes[1, 1].transAxes,
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        comparison_cm_path = os.path.join(outputDir, f"{modelFnameRoot}_pt_vs_ptl_confusion_matrices.png")
        plt.savefig(comparison_cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"{TAG}: Comparison confusion matrices saved as {comparison_cm_path}")
        
        print(f"{'='*70}")
        print(f"{TAG}: Model comparison complete")
        print(f"{'='*70}\n")

    # Generate per-event visualization charts in eventData subfolder (as per spec)
    # Each chart has two vertically stacked panels: top = raw accel (X/Y/Z or magnitude) with seizure shading + HR on secondary y-axis,
    # bottom = seizure probability vs time with same shading. Title includes eventId, type, subtype and desc subtitle.
    # Implemented via nnTrainer.generate_event_charts / plot_event_chart (reference: curator_tools/event_editor.py shading).
    try:
        TAG2 = "nnTester.testModel():eventData"
        print(f"{TAG2}: Generating per-seizure event charts in eventData subfolder...")
        # Ensure dataTime column present for time base; if missing, fallback will use index-based time.
        # df at this point is the filtered datapoint DataFrame whose row order matches prediction_proba.
        # event_details_map and event_stats_df were enriched after loading allData.json.
        # Use only seizure events (true_label==1) as per requirement.
        # Pass enriched structures to nnTrainer helper.
        # Guard against missing variables when called in kFold path where event_details_map may be undefined
        _ed_map = event_details_map if 'event_details_map' in locals() else {}
        _ev_stats = event_stats_df if 'event_stats_df' in locals() else None
        _df_for_plot = df if 'df' in locals() else df_original if 'df_original' in locals() else None
        _proba_for_plot = prediction_proba if 'prediction_proba' in locals() else None
        if _df_for_plot is not None and _proba_for_plot is not None:
            n_charts = nnTrainer.generate_event_charts(
                outputDir=outputDir,
                df=_df_for_plot,
                prediction_proba=_proba_for_plot,
                event_details_map=_ed_map,
                event_stats_df=_ev_stats,
                modelFnameRoot=modelFnameRoot,
                titlePrefix=titlePrefix,
                debug=debug,
                only_seizure=True
            )
            print(f"{TAG2}: Generated {n_charts} chart(s) in {os.path.join(outputDir, 'eventData')}")
        else:
            print(f"{TAG2}: Skipping - df or prediction_proba not available")
    except Exception as e:
        print(f"nnTester.testModel(): Warning - eventData chart generation failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    # Clean up memory
    if framework == 'pytorch':
        import torch
        del model, nnModel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"{TAG}: CUDA memory cleared")
    
    print("nnTester: Testing Complete")
    return foldResults



def calcTotals(yTest, pSeizure, th = 0.5):
    ''' Calculate true positive (TP), True Negative (TN), False Positive (FP)
    and False Negative (FN) totals, for the data in yTest (where 0=ok and 1 = seizure)
    and pSeizure which is the probability of the event being a seizure, uisng threshold th.

    FIXME: I am sure there is a more efficient python way of doing this - this is how I 
    would have written it in C  :)

    '''
    nTP = 0
    nTN = 0
    nFP = 0
    nFN = 0

    for i in range(0,len(yTest)):
        if (yTest[i] == 1):   # Event was a seizure
            if (pSeizure[i]>th):
                nTP += 1   # True Positive
            else:
                nFN += 1   # False Negative
        elif (yTest[i] ==0):  # Event was not a seizure
            if (pSeizure[i]>th):
                nFP += 1   # False Positive
            else:
                nTN += 1   # True Negative
        else:
            print("WARNING - Unrecognised yTest Value: %d" % yTest[i])

    return(nTP, nFP, nTN, nFN)


def calcConfusionMatrix(configObj, modelFnameRoot="best_model", 
                        xTest=None, yTest=None, dataDir=".", balanced=True, debug=False, titlePrefix=None,
                        prediction_proba=None, prediction=None, load_predictions_csv=False):
    """Calculate and save confusion matrix and statistics.
    
    Args:
        configObj: Configuration object
        modelFnameRoot: Root name for model files
        xTest: Test data (optional if predictions provided or loaded from CSV)
        yTest: Test labels (optional if loaded from CSV, otherwise required)
        dataDir: Directory for model and data files
        balanced: Whether to use balanced test data
        debug: Enable debug output
        titlePrefix: Title prefix for plots
        prediction_proba: Pre-computed prediction probabilities (optional, avoids re-running inference)
        prediction: Pre-computed predictions (optional, avoids re-running inference)
        load_predictions_csv: If True, try to load predictions from CSV file (for crash recovery)
    
    Returns:
        None - saves confusion matrix, statistics, and plots to files
    """
    TAG = "nnTrainer.calcConfusionMatrix()"
    print("____%s____" % (TAG))
    
    # If titlePrefix not specified, use modelFnameRoot
    if titlePrefix is None:
        titlePrefix = modelFnameRoot
    
    # Try to load predictions from CSV if requested
    predictions_csv = os.path.join(dataDir, f"{modelFnameRoot}_predictions.csv")
    if load_predictions_csv and os.path.exists(predictions_csv):
        print(f"{TAG}: Loading predictions from {predictions_csv}")
        try:
            predictions_df = pd.read_csv(predictions_csv)
            yTest = predictions_df['true_label'].values
            prediction = predictions_df['predicted_label'].values
            pSeizure = predictions_df['seizure_probability'].values
            # Reconstruct prediction_proba from seizure probability
            prediction_proba = np.column_stack([1 - pSeizure, pSeizure])
            print(f"{TAG}: Successfully loaded {len(yTest)} predictions from CSV")
        except Exception as e:
            print(f"{TAG}: Warning - Could not load predictions from CSV: {e}")
            load_predictions_csv = False
    
    # Detect framework
    framework = nnTrainer.get_framework_from_config(configObj)
    
    nnModelClassName = libosd.configUtils.getConfigParam("modelClass", configObj['modelConfig'])
    if (balanced):
        testDataFname = os.path.join(dataDir, libosd.configUtils.getConfigParam("testBalancedFileCsv", configObj['dataFileNames']))
    else:   
        testDataFname = os.path.join(dataDir, libosd.configUtils.getConfigParam("testDataFileCsv", configObj['dataFileNames']))

    inputDims = libosd.configUtils.getConfigParam("dims", configObj['modelConfig'])
    if (inputDims is None): inputDims = 1

    # Determine the correct file extension and base model name
    # If modelFnameRoot ends with _ptl or _pte, strip that suffix and use the appropriate extension
    # The actual model files are named like: deepEpiCnnModel_pytorch.pt, .ptl, or .pte
    # But the variant name for outputs includes the suffix: deepEpiCnnModel_pytorch_pte
    if modelFnameRoot.endswith('_ptl'):
        modelExt = '.ptl'
        baseModelName = modelFnameRoot[:-4]  # Remove '_ptl' suffix
    elif modelFnameRoot.endswith('_pte'):
        modelExt = '.pte'
        baseModelName = modelFnameRoot[:-4]  # Remove '_pte' suffix
    else:
        modelExt = get_model_extension(framework)
        baseModelName = modelFnameRoot
    modelFname = f"{baseModelName}{modelExt}"
    
    # Parse model class name properly
    parts = nnModelClassName.split('.')
    if len(parts) < 2:
        raise ValueError("modelClass must be a module path and class name, e.g. 'mod.submod.ClassName'")
    nnModuleId = '.'.join(parts[:-1])
    nnClassId = parts[-1]

    if (debug): print("%s: Importing nn Module %s" % (TAG, nnModuleId))
    nnModule = importlib.import_module(nnModuleId)
    # Instantiate the model class with modelConfig
    nnModel = getattr(nnModule, nnClassId)(configObj['modelConfig'])

    # Only load test data if we don't already have predictions from CSV
    if not load_predictions_csv or (xTest is None and (prediction_proba is None or prediction is None)):
        if (xTest is None or yTest is None):
            # Load the test data from file
            print("%s: Loading Test Data from File %s" % (TAG, testDataFname))
            df = augmentData.loadCsv(testDataFname, debug=debug)
            print("%s: Loaded %d datapoints" % (TAG, len(df)))
            #augmentData.analyseDf(df)

            print("%s: Re-formatting data for testing" % (TAG))
            xTest, yTest = nnTrainer.df2trainingData(df, nnModel)

            print("%s: Converting to np arrays" % (TAG))
            xTest = np.array(xTest)
            yTest = np.array(yTest)

            print("%s: re-shaping array for testing" % (TAG))
            if xTest.ndim == 2:
                xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
            elif xTest.ndim == 3:
                # Keep channel-last tensors as-is, e.g. (batch, 750, 3) for XYZ mode.
                pass
            elif xTest.ndim == 4 and inputDims == 2:
                xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], xTest.shape[2], 1))
            else:
                print(f"ERROR - unsupported xTest shape {xTest.shape} for inputDims={inputDims}")
                exit(-1)

    nClasses = len(np.unique(yTest))
    print("nClasses=%d" % nClasses)
    # In the following, yTest == 1 returns an array that is true (1), for all elements where yTest == 1, and false (0) for other values of yTest - we then count how many of those elements are not zero to give
    # the number of elements where yTest = 1.
    # In our case we could have just done count_nonzero(yTest), but doing it this way gives us the option of expanding to more than 2 categories of event.
    print("Testing using %d seizure datapoints and %d false alarm datapoints"
          % (np.count_nonzero(yTest == 1),
             np.count_nonzero(yTest == 0)))

    # Only run inference if predictions are not already provided
    if prediction_proba is None or prediction is None:
        print(f"{TAG}: No pre-computed predictions provided, running inference...")
        
        # Load the trained model back from disk and test it.
        modelFname = os.path.join(dataDir, f"{modelFnameRoot}{modelExt}")
        print("Loading trained model %s" % modelFname)
        model = load_model_for_testing(modelFname, nnModel, framework)
        print("Evaluating model....")
        test_loss, test_acc = evaluate_model(model, xTest, yTest, framework)
        print("Test Loss=%.2f, Test Acc=%.2f" % (test_loss, test_acc))

       
        if (debug): print("yTest=",yTest)
        # create an array of the indices of true seizure events.
        y_true=[]
        for element in yTest:
            y_true.append(np.argmax(element))
        if (debug): print("y_true=",y_true)

        print("Calculating seizure probabilities from test data")
        prediction_proba = predict_model(model, xTest, framework)
        if (debug): print("prediction_proba=",prediction_proba)
        prediction=np.argmax(prediction_proba,axis=1)
    else:
        print(f"{TAG}: Using pre-computed predictions (skipping inference)")
    
    # Threshold analysis and probability plot
    pSeizure = prediction_proba[:,1]
    seq = range(0,len(pSeizure))
    # Colour seizure data points red, and non-seizure data blue.
    colours = [ 'red' if seizureVal==1 else 'blue' for seizureVal in yTest]
    
    # Calculate statistics at different thresholds
    thLst = []
    nTPLst = []
    nFPLst = []
    nTNLst = []
    nFNLst = []
    TPRLst = []
    FPRLst = []
    
    thresholdLst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for th in thresholdLst:
        nTP, nFP, nTN, nFN = calcTotals(yTest, pSeizure, th)
        thLst.append(th)
        nTPLst.append(nTP)
        nFPLst.append(nFP)
        nTNLst.append(nTN)
        nFNLst.append(nFN)
        tp_denom = (nTP + nFN)
        fp_denom = (nFP + nTN)
        TPRLst.append((nTP / tp_denom) if tp_denom > 0 else float('nan'))
        FPRLst.append((nFP / fp_denom) if fp_denom > 0 else float('nan'))
    
    print("Threshold Analysis:")
    print("th", thLst)    
    print("nTP", nTPLst)
    print("nFP", nFPLst)
    print("nTN", nTNLst)
    print("nFN", nFNLst)
    print("TPR", TPRLst)
    print("FPR", FPRLst)
    
    # Create probability scatter plot
    fig, ax = plt.subplots(2,1)
    ax[0].title.set_text("%s: Datapoint-Level Seizure Probabilities" % titlePrefix)
    ax[0].set_ylabel('Probability')
    ax[0].set_xlabel('Datapoint')
    ax[0].scatter(seq, pSeizure, s=2.0, marker='x', c=colours)
    ax[1].plot(yTest)
    fname_prob = os.path.join(dataDir,"%s_probabilities.png" % modelFnameRoot)
    fig.savefig(fname_prob)
    plt.close()
    print("Probability plot saved as %s" % fname_prob)
       
    # Confusion Matrix
    import seaborn as sns
    LABELS = ['No-Alarm','Seizure']
    # cm = metrics.confusion_matrix(prediction, yTest)
    cm = metrics.confusion_matrix(yTest, prediction)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                linewidths = 0.1, fmt="d", cmap = 'YlGnBu');
    plt.title("%s: Datapoint-Level Confusion Matrix" % titlePrefix, fontsize = 15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fname = os.path.join(dataDir, "%s_confusion.png" % modelFnameRoot)
    plt.savefig(fname)
    plt.close()
    print("Confusion Matrix Saved as %s." % fname)
    
    nTrue = 0
    nFalse = 0
    nTP = 0
    nFN = 0
    nTN = 0
    nFP = 0
    for n in range (0,len(yTest)):
        if (yTest[n]==1):
            nTrue += 1
        else:
            nFalse += 1
        if (yTest[n]==1):
            if (prediction[n]==1):
                nTP += 1
            else:
                nFN += 1
        else:
            if (prediction[n]==1):
                nFP += 1
            else:
                nTN += 1


    fname = os.path.join(dataDir, "%s_stats.txt" % modelFnameRoot)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    total1=sum(sum(cm))
    with open(fname,"w") as outFile:
        outFile.write("\n|====================================================================|\n")
        outFile.write("****  Open Seizure Detector Classification Metrics Analysis  ****\n")
        outFile.write("|====================================================================|\n\n")
        
        # DATAPOINT-LEVEL METRICS SECTION
        outFile.write("DATAPOINT-LEVEL METRICS\n")
        outFile.write("="*70 + "\n")
        outFile.write("Analysis of %d datapoints (seizure and non-seizure)\n" % total1)
        outFile.write("Totals:  Seizures %d, non-Seizures %d\n\n" % (nTrue, nFalse))
        outFile.write("    nTP = %d,  nFN= %d\n" % (nTP, nFN))
        outFile.write("    nTN = %d,  nFP= %d\n" % (nTN, nFP))
        tpr_denom = (nTP + nFN)
        tpr = (nTP / tpr_denom) if tpr_denom > 0 else float('nan')
        outFile.write("    TPR = %.4f\n" % tpr)
        tnr_denom = (nTN + nFP)
        tnr = (nTN / tnr_denom) if tnr_denom > 0 else float('nan')
        outFile.write("    TNR = %.4f\n\n" % tnr)

        outFile.write("Metrics from Confusion Matrix:\n")
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = np.divide(TP, (TP + FN), out=np.full_like(TP, np.nan, dtype=float), where=(TP + FN) != 0)
        outFile.write("Sensitivity/recall or true positive rate: %.4f  %.4f\n" % tuple(TPR))
        # Specificity or true negative rate
        TNR = np.divide(TN, (TN + FP), out=np.full_like(TN, np.nan, dtype=float), where=(TN + FP) != 0)
        outFile.write("Specificity or true negative rate: %.4f  %.4f\n" % tuple(TNR))
        # Precision or positive predictive value
        PPV = np.divide(TP, (TP + FP), out=np.full_like(TP, np.nan, dtype=float), where=(TP + FP) != 0)
        outFile.write("Precision or positive predictive value: %.4f  %.4f\n" % tuple(PPV))
        # Negative predictive value
        NPV = np.divide(TN, (TN + FN), out=np.full_like(TN, np.nan, dtype=float), where=(TN + FN) != 0)
        outFile.write("Negative predictive value: %.4f  %.4f\n" % tuple(NPV))
        # Fall out or false positive rate
        FPR = np.divide(FP, (FP + TN), out=np.full_like(FP, np.nan, dtype=float), where=(FP + TN) != 0)
        outFile.write("Fall out or false positive rate: %.4f  %.4f\n" % tuple(FPR))
        # False negative rate
        FNR = np.divide(FN, (TP + FN), out=np.full_like(FN, np.nan, dtype=float), where=(TP + FN) != 0)
        outFile.write("False negative rate: %.4f  %.4f\n" % tuple(FNR))
        # False discovery rate
        FDR = np.divide(FP, (TP + FP), out=np.full_like(FP, np.nan, dtype=float), where=(TP + FP) != 0)
        outFile.write("False discovery rate: %.4f  %.4f\n" % tuple(FDR))
        # Overall accuracy
        ACC = np.divide((TP + TN), (TP + FP + FN + TN), out=np.full_like(TP, np.nan, dtype=float), where=(TP + FP + FN + TN) != 0)
        outFile.write("Classification Accuracy: %.4f  %.4f\n" % tuple(ACC))
        outFile.write("|====================================================================|\n")
        report = classification_report(yTest, prediction)
        outFile.write(report)
        outFile.write("|====================================================================|\n\n")
        
        # TensorFlow-specific model analysis
        if framework == 'tensorflow':
            from tensorflow import keras
            x=keras.metrics.sparse_categorical_accuracy(xTest, yTest)

            # summarize filter shapes
            for layer in model.layers:
            # check for convolutional layer
             if 'conv' not in layer.name:
                 continue

            # get filter weights
            filters, biases = layer.get_weights()
            filterStr = layer.name
            for n in filters.shape:
                filterStr="%s, %d" % (filterStr,n)
            filterStr="%s\n" % filterStr
            outFile.write(filterStr)


            # summarize feature map shapes
            for i in range(len(model.layers)):
                layer = model.layers[i]
                # check for convolutional layer
                if 'conv' not in layer.name:
                    continue
                # summarize output shape
                outFile.write("%d:  %s : " % (i, layer.name))
                for n in layer.output.shape:
                    if n is not None:
                        outFile.write("%d, " % n)
                outFile.write("\n")

    print("Statistics Summary saved as %s." % fname)





def testKFold(configObj, kfold, dataDir='.', rerun=False, debug=False):
    """Test all k-fold models and aggregate results.
    
    Args:
        configObj: Configuration object
        kfold: Number of folds to test
        dataDir: Base directory containing fold subdirectories
        rerun: If True, re-run testing even if results exist
        debug: Enable debug output
    
    Returns:
        Dictionary with aggregated results across all folds
    """
    TAG = "nnTester.testKFold()"
    print("____%s____" % TAG)
    print(f"{TAG}: Testing {kfold} folds")
    
    # Verify that fold directories exist
    fold_dirs = []
    for nFold in range(kfold):
        fold_dir = os.path.join(dataDir, f"fold{nFold}")
        if not os.path.exists(fold_dir):
            raise ValueError(f"{TAG}: Fold directory not found: {fold_dir}. Expected {kfold} folds but fold{nFold} does not exist.")
        fold_dirs.append(fold_dir)
    
    print(f"{TAG}: Found all {kfold} fold directories")
    
    # Test each fold
    foldResults = []
    for nFold in range(kfold):
        fold_dir = fold_dirs[nFold]
        results_file = os.path.join(fold_dir, 'testResults.json')
        
        print(f"\n{TAG}: ========== Processing Fold {nFold} ==========")
        
        # Check if results already exist
        if os.path.exists(results_file) and not rerun:
            print(f"{TAG}: Loading existing results from {results_file}")
            with open(results_file, 'r') as f:
                fold_result = json.load(f)
            foldResults.append(fold_result)
        else:
            if rerun and os.path.exists(results_file):
                print(f"{TAG}: Re-running test for fold {nFold} (--rerun specified)")
            else:
                print(f"{TAG}: No existing results found, running test for fold {nFold}")
            
            # Run the test (skip .ptl testing for k-fold validation, test_ptl=False)
            fold_result = testModel(configObj, dataDir=fold_dir, balanced=False, debug=debug, test_ptl=False)
            foldResults.append(fold_result)
        
        print(f"{TAG}: Fold {nFold} complete")
    
    # Compute average results across folds
    print(f"\n{TAG}: ========== Computing K-Fold Statistics ==========")
    avgResults = {}
    for key in foldResults[0].keys():
        avgResults[key] = sum(foldResult[key] for foldResult in foldResults) / len(foldResults)
    
    # Calculate standard deviation for each key
    for key in foldResults[0].keys():
        avgResults[key + "_std"] = np.std([foldResult[key] for foldResult in foldResults])
    
    # Save the results to files
    kfoldSummaryPath = os.path.join(dataDir, "kfold_summary.txt")
    kfoldJsonPath = os.path.join(dataDir, "kfold_summary.json")
    
    with open(kfoldSummaryPath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("K-Fold Cross-Validation Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Number of folds: {kfold}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write("="*70 + "\n\n")
        
        f.write("EPOCH-BASED ANALYSIS:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Metric':<30} {'Mean':<15} {'Std Dev':<15}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Accuracy (Model)':<30} {avgResults['accuracy']:<15.4f} {avgResults['accuracy_std']:<15.4f}\n")
        f.write(f"{'Accuracy (OSD)':<30} {avgResults['accuracyOsd']:<15.4f} {avgResults['accuracyOsd_std']:<15.4f}\n")
        f.write(f"{'TPR/Sensitivity (Model)':<30} {avgResults['tpr']:<15.4f} {avgResults['tpr_std']:<15.4f}\n")
        f.write(f"{'TPR/Sensitivity (OSD)':<30} {avgResults['tprOsd']:<15.4f} {avgResults['tprOsd_std']:<15.4f}\n")
        f.write(f"{'FPR (Model)':<30} {avgResults['fpr']:<15.4f} {avgResults['fpr_std']:<15.4f}\n")
        f.write(f"{'FPR (OSD)':<30} {avgResults['fprOsd']:<15.4f} {avgResults['fprOsd_std']:<15.4f}\n")
        f.write("\n")
        
        f.write("EVENT-BASED ANALYSIS:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Metric':<30} {'Mean':<15} {'Std Dev':<15}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'TPR/Sensitivity (Model)':<30} {avgResults['event_tpr']:<15.4f} {avgResults['event_tpr_std']:<15.4f}\n")
        f.write(f"{'TPR/Sensitivity (OSD)':<30} {avgResults['osd_event_tpr']:<15.4f} {avgResults['osd_event_tpr_std']:<15.4f}\n")
        f.write(f"{'FPR (Model)':<30} {avgResults['event_fpr']:<15.4f} {avgResults['event_fpr_std']:<15.4f}\n")
        f.write(f"{'FPR (OSD)':<30} {avgResults['osd_event_fpr']:<15.4f} {avgResults['osd_event_fpr_std']:<15.4f}\n")
        f.write("\n")
        
        f.write("DETAILED RESULTS BY FOLD:\n")
        f.write("="*70 + "\n")
        for nFold, result in enumerate(foldResults):
            f.write(f"\nFold {nFold}:\n")
            f.write(f"  Epoch-based - Accuracy: {result['accuracy']:.4f}, TPR: {result['tpr']:.4f}, FPR: {result['fpr']:.4f}\n")
            f.write(f"  Event-based - TPR: {result['event_tpr']:.4f}, FPR: {result['event_fpr']:.4f}\n")
    
    # Save JSON summary
    summary_data = {
        'kfold': kfold,
        'timestamp': str(pd.Timestamp.now()),
        'averages': avgResults,
        'fold_results': foldResults
    }
    
    with open(kfoldJsonPath, 'w') as jf:
        json.dump(summary_data, jf, indent=2)
    
    print(f"\n{TAG}: K-Fold summary saved to {kfoldSummaryPath}")
    print(f"{TAG}: K-Fold JSON data saved to {kfoldJsonPath}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(f"Number of folds: {kfold}\n")
    print("EPOCH-BASED ANALYSIS:")
    print(f"  Model - Accuracy: {avgResults['accuracy']:.4f} ± {avgResults['accuracy_std']:.4f}")
    print(f"  Model - TPR: {avgResults['tpr']:.4f} ± {avgResults['tpr_std']:.4f}")
    print(f"  Model - FPR: {avgResults['fpr']:.4f} ± {avgResults['fpr_std']:.4f}")
    print(f"  OSD   - Accuracy: {avgResults['accuracyOsd']:.4f} ± {avgResults['accuracyOsd_std']:.4f}")
    print(f"  OSD   - TPR: {avgResults['tprOsd']:.4f} ± {avgResults['tprOsd_std']:.4f}")
    print(f"  OSD   - FPR: {avgResults['fprOsd']:.4f} ± {avgResults['fprOsd_std']:.4f}")
    print("\nEVENT-BASED ANALYSIS:")
    print(f"  Model - TPR: {avgResults['event_tpr']:.4f} ± {avgResults['event_tpr_std']:.4f}")
    print(f"  Model - FPR: {avgResults['event_fpr']:.4f} ± {avgResults['event_fpr_std']:.4f}")
    print(f"  OSD   - TPR: {avgResults['osd_event_tpr']:.4f} ± {avgResults['osd_event_tpr_std']:.4f}")
    print(f"  OSD   - FPR: {avgResults['osd_event_fpr']:.4f} ± {avgResults['osd_event_fpr_std']:.4f}")
    print("="*70)
    
    return avgResults


def main():
    print("nnTester.main()")
    parser = argparse.ArgumentParser(description='Apply the training data to calculate statistcs on a trained model (specifid in the config file)')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing model configuration')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    parser.add_argument('--test-data', default=None,
                        help='Path to test data CSV file (overrides config file setting)')
    parser.add_argument('--test-ptl', action="store_true", default=False,
                        help='Also test the .ptl (PyTorch Lite) model if available')
    parser.add_argument('--test-pte', action="store_true", default=False,
                        help='Also test the .pte (ExecuTorch) model if available')
    parser.add_argument('--pte-test-percent', type=float, default=100.0,
                        help='Percentage of test data to use for PTE testing (1-100, default: 100)')
    parser.add_argument('--use-predictions-csv', action="store_true", default=False,
                        help='Load predictions from existing CSV file instead of re-running inference (for crash recovery and efficient re-analysis)')
    parser.add_argument('--kfold', type=int, default=None,
                        help='Number of folds for k-fold cross-validation testing. Tests all folds and aggregates results.')
    parser.add_argument('--rerun', action="store_true",
                        help='Re-run tests even if results already exist (only used with --kfold)')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    configObj = libosd.configUtils.loadConfig(args['config'])
    print("configObj=",configObj)
    # Load a separate OSDB Configuration file if it is included.
    if ("osdbCfg" in configObj):
        osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg",configObj)
        print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
        osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
        # Merge the contents of the OSDB Configuration file into configObj
        configObj = configObj | osdbCfgObj

    print("configObj=",configObj.keys())

    debug = configObj['debug']
    if args['debug']: debug=True

    # Check if k-fold testing is requested
    if args['kfold'] is not None:
        kfold = args['kfold']
        if kfold < 1:
            raise ValueError(f"kfold must be >= 1, got {kfold}")
        rerun = args['rerun']
        testKFold(configObj, kfold=kfold, dataDir='.', rerun=rerun, debug=debug)
    else:
        # Test single model with optional .ptl and .pte comparison
        test_ptl = args['test_ptl']
        test_pte = args['test_pte']
        test_percent = args.get('test_percent', args.get('pte_test_percent', 100.0))
        testModel(configObj, debug=debug, testDataCsv=args['test_data'], test_ptl=test_ptl, test_pte=test_pte, test_percent=test_percent)
        
    


if __name__ == "__main__":
    main()
