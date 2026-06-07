#!/usr/bin/env python3

"""Compare streaming seizure probabilities between local .pte model and device implementation.

Accepts an input CSV produced by either:
- user_tools/nnTraining2/flattenData.py  (one row per 5s datapoint, 125 accel samples)
- user_tools/nnTraining2/extractFeatures.py (window/epoch rows; we reconstruct a sample stream)

For each event, the script streams datapoints in order into:
- Local ExecuTorch .pte model (via user_tools/testRunner/nnAlg.NnAlg)
- Device model over HTTP (via user_tools/testRunner/deviceAlg.DeviceAlg)

It records per-datapoint seizure probability (when available), produces per-event plots/tables,
computes event-level metrics, and writes a simple markdown validation report.

Example:
  python user_tools/compare_streaming_algorithms.py \
    --input /home/graham/24/testData.csv \
    --pte-model /home/graham/24/deepEpiCnnModel_pytorch.pte \
    --device-ip 192.168.0.53 \
    --out-dir /tmp/osd_compare
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend (safe on headless systems)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Make repo imports work when run as a standalone script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# testRunner modules were historically written to be run from within
# user_tools/testRunner and use imports like `import sdAlg`.
# Add that directory to sys.path so these imports work from any CWD.
TESTRUNNER_DIR = os.path.join(REPO_ROOT, "user_tools", "testRunner")
if TESTRUNNER_DIR not in sys.path:
    sys.path.append(TESTRUNNER_DIR)

import libosd.dpTools

if TYPE_CHECKING:
    from user_tools.testRunner.deviceAlg import DeviceAlg


@dataclass
class AlgoResult:
    valid: bool
    alarm_state: Optional[int]
    p_seizure: Optional[float]
    raw: Optional[Dict[str, Any]]


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_event_id(event_id: Any) -> str:
    s = str(event_id)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _infer_label_from_row(row: pd.Series) -> Optional[int]:
    """Return 1 for seizure, 0 for non-seizure, None if unknown."""
    if 'type' in row and pd.notna(row['type']):
        try:
            t = int(float(row['type']))
            return 1 if t == 1 else 0
        except Exception:
            pass

    type_str = None
    for k in ('typeStr', 'type'):
        if k in row and isinstance(row[k], str) and row[k].strip():
            type_str = row[k]
            break

    if type_str:
        if 'seizure' in type_str.lower():
            return 1
        if 'false' in type_str.lower() or 'nda' in type_str.lower():
            return 0

    return None


def _extract_accel_from_flattened_row(row: pd.Series) -> Tuple[Optional[List[float]], Optional[int]]:
    """Return (rawData[125], hr) from flattenData style row."""
    mags: List[float] = []
    for n in range(125):
        col = f"M{n:03d}"
        v = row.get(col, np.nan)
        if pd.isna(v):
            return None, None
        mags.append(float(v))

    hr_val = row.get('hr', np.nan)
    hr = -1
    if pd.notna(hr_val):
        try:
            hr = int(float(hr_val))
        except Exception:
            hr = -1

    return mags, hr


def _sorted_sample_cols(cols: Iterable[str], prefix: str, suffix: str = "") -> List[str]:
    pat = re.compile(rf"^{re.escape(prefix)}(\\d{{3}}){re.escape(suffix)}$")
    matches = []
    for c in cols:
        m = pat.match(c)
        if m:
            matches.append((int(m.group(1)), c))
    matches.sort(key=lambda t: t[0])
    return [c for _, c in matches]


def _reconstruct_samples_from_extractfeatures(event_df: pd.DataFrame) -> List[float]:
    """Reconstruct a single sample stream from extractFeatures output.

    extractFeatures outputs overlapping windows with startSample/endSample and per-sample columns.
    We reconstruct a best-effort continuous series by taking the first window, then appending the
    non-overlapping tail of subsequent windows (in startSample order).
    """
    if 'startSample' not in event_df.columns or 'endSample' not in event_df.columns:
        raise ValueError("extractFeatures-style CSV must contain startSample/endSample")

    # Prefer M###_t-0 (common), else M###
    sample_cols = _sorted_sample_cols(event_df.columns, 'M', '_t-0')
    if not sample_cols:
        sample_cols = _sorted_sample_cols(event_df.columns, 'M')
    if not sample_cols:
        raise ValueError("No magnitude sample columns found (expected M000...)")

    df = event_df.copy()
    df = df.sort_values(['startSample', 'endSample'])

    samples: List[float] = []
    current_end = None
    for _, row in df.iterrows():
        start = int(row['startSample'])
        end = int(row['endSample'])
        win = [float(row[c]) for c in sample_cols]

        if current_end is None:
            samples.extend(win)
            current_end = end
            continue

        if start < current_end:
            overlap = current_end - start
            if overlap < len(win):
                samples.extend(win[overlap:])
        elif start == current_end:
            samples.extend(win)
        else:
            # gap: fill with zeros up to start, then add window
            samples.extend([0.0] * (start - current_end))
            samples.extend(win)

        current_end = max(current_end, end)

    return samples


def _chunk_samples_to_datapoints(samples: List[float], samples_per_dp: int = 125) -> List[List[float]]:
    dps = []
    n = len(samples) // samples_per_dp
    for i in range(n):
        dps.append(samples[i * samples_per_dp:(i + 1) * samples_per_dp])
    return dps


def _std_reject(acc_data: List[float], mode: str, thresh: float) -> bool:
    """Return True if datapoint should be rejected."""
    if thresh <= 0:
        return False

    arr = np.asarray(acc_data, dtype=float)
    if arr.size == 0:
        return True

    std = float(arr.std())

    if mode == 'abs':
        return std < thresh

    # pct-of-mean (matches existing nnAlg behaviour)
    mean = float(arr.mean())
    if mean == 0:
        return True
    std_pct = 100.0 * std / mean
    return std_pct < thresh


def _parse_algo_json(ret: str) -> AlgoResult:
    if ret is None:
        return AlgoResult(valid=False, alarm_state=None, p_seizure=None, raw=None)

    try:
        obj = json.loads(ret)
    except Exception:
        return AlgoResult(valid=False, alarm_state=None, p_seizure=None, raw={'rawText': ret})

    valid = bool(obj.get('valid', True))
    alarm_state = obj.get('alarmState', None)
    if alarm_state is not None:
        try:
            alarm_state = int(alarm_state)
        except Exception:
            alarm_state = None

    # try multiple likely keys
    p = None
    for k in ('pSeizure', 'p_seizure', 'prob', 'probability', 'seizureProb', 'seizureProbability'):
        if k in obj and obj[k] is not None:
            try:
                p = float(obj[k])
                break
            except Exception:
                pass

    return AlgoResult(valid=valid, alarm_state=alarm_state, p_seizure=p, raw=obj)


def _make_dp(acc_data: List[float], hr: int, data_time: Optional[str] = None) -> Dict[str, Any]:
    return {
        'dataTime': data_time or '2000-01-01T00:00:00Z',
        'alarmState': 0,
        'specPower': 0,
        'roiPower': 0,
        'hr': int(hr),
        'o2Sat': -1,
        'rawData': list(acc_data),
        'maxVal': 0,
        'minVal': 0,
        'maxFreq': 0,
        'alarmPhrase': ''
    }


def _send_zero_flush(alg: DeviceAlg, event_id: Any, n_datapoints: int = 6) -> None:
    zero_dp = _make_dp([0.0] * 125, hr=-1)
    raw = libosd.dpTools.dp2rawData(zero_dp, debug=False)
    for _ in range(n_datapoints):
        try:
            alg.processDp(raw, str(event_id))
        except Exception:
            # best-effort
            pass


def _event_plot(
    out_png: str,
    df: pd.DataFrame,
    event_id: Any,
    prob_thresh: float,
) -> None:
    plt.figure(figsize=(10, 4))
    x = df['dpIndex'].to_numpy()

    # Shade background only for periods where the *device-reported* alarm state is ALARM (2).
    # We treat each datapoint index as a 5s bin and shade contiguous runs.
    if len(x) and 'alarmState_device' in df.columns:
        alarm = pd.to_numeric(df['alarmState_device'], errors='coerce')
        is_alarm = alarm.fillna(-1).astype(int).to_numpy() == 2
        # Optional: respect validity gating (warmup), if present.
        if 'valid_device' in df.columns:
            valid_dev = df['valid_device'].fillna(False).astype(bool).to_numpy()
            is_alarm = is_alarm & valid_dev

        # Convert to spans on x-axis. Each dpIndex i shades [i-0.5, i+0.5].
        ax = plt.gca()
        start_idx = None
        for i, flag in enumerate(is_alarm):
            if flag and start_idx is None:
                start_idx = i
            if (not flag) and start_idx is not None:
                x0 = float(x[start_idx]) - 0.5
                x1 = float(x[i - 1]) + 0.5
                ax.axvspan(x0, x1, color='lightskyblue', alpha=0.18, zorder=0)
                start_idx = None
        if start_idx is not None:
            x0 = float(x[start_idx]) - 0.5
            x1 = float(x[len(x) - 1]) + 0.5
            ax.axvspan(x0, x1, color='lightskyblue', alpha=0.18, zorder=0)

    # Plot local first, slightly thicker, so it remains visible underneath the device line.
    if 'pSeizure_local' in df.columns:
        plt.plot(
            x,
            df['pSeizure_local'].to_numpy(dtype=float),
            label='local .pte',
            linewidth=3.0,
            zorder=1,
        )

    if 'pSeizure_device' in df.columns:
        plt.plot(
            x,
            df['pSeizure_device'].to_numpy(dtype=float),
            label='device',
            linewidth=1.5,
            zorder=2,
        )

    plt.axhline(prob_thresh, color='k', linestyle='--', linewidth=1, alpha=0.6, label=f"thresh={prob_thresh}")
    plt.ylim(-0.02, 1.02)
    plt.xlim(x.min() if len(x) else 0, x.max() if len(x) else 1)
    plt.xlabel('datapoint index (5s steps)')
    plt.ylabel('seizure probability')
    plt.title(f"Event {event_id} seizure probability")
    plt.grid(True, alpha=0.25)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _compute_event_pred(max_prob: Optional[float], prob_thresh: float) -> Optional[int]:
    if max_prob is None or np.isnan(max_prob):
        return None
    return 1 if max_prob >= prob_thresh else 0


def _compute_event_pred_from_alarm_state2(
    alarm_states: pd.Series,
) -> Optional[int]:
    """Return 1 if any datapoint is in ALARM (state==2), else 0.

    If there are no usable datapoints, returns None.
    """
    if alarm_states is None:
        return None
    s = pd.to_numeric(alarm_states, errors='coerce').dropna()
    if s.empty:
        return None
    return 1 if (s.astype(int) == 2).any() else 0


def _confusion_counts(rows: List[Tuple[Optional[int], Optional[int]]]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for y_true, y_pred in rows:
        if y_true is None or y_pred is None:
            continue
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def _metrics_from_counts(c: Dict[str, int]) -> Dict[str, float]:
    tp, fp, tn, fn = c['tp'], c['fp'], c['tn'], c['fn']
    tpr = tp / (tp + fn) if (tp + fn) else float('nan')
    fpr = fp / (fp + tn) if (fp + tn) else float('nan')
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float('nan')
    return {'TPR': tpr, 'FPR': fpr, 'ACC': acc}


def _round_prob(v: Optional[float], ndp: int = 4) -> Optional[float]:
    if v is None:
        return None
    try:
        if np.isnan(v):
            return None
    except Exception:
        pass
    try:
        return round(float(v), ndp)
    except Exception:
        return None


def _fmt_prob(v: Optional[float], ndp: int = 4) -> str:
    r = _round_prob(v, ndp=ndp)
    return "None" if r is None else f"{r:.{ndp}f}"


def _clean_csv_header_name(header: Any) -> str:
    if header is None:
        return ''
    if not isinstance(header, str):
        header = str(header)
    h = header.replace('\ufeff', '').replace('\r', '').replace('\n', '').strip()
    # Tolerate files that quote every header name
    if (len(h) >= 2) and ((h[0] == '"' and h[-1] == '"') or (h[0] == "'" and h[-1] == "'")):
        h = h[1:-1].strip()
    return h


def _read_csv_headers(path: str) -> List[str]:
    # Use csv.reader so quoted headers and embedded commas/newlines are handled correctly.
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
    return [_clean_csv_header_name(h) for h in headers]


def _detect_format(headers: List[str]) -> str:
    headers_set = set(_clean_csv_header_name(h) for h in headers)
    if 'eventId' in headers_set and any(h.startswith('M000') for h in headers_set) and 'M124' in headers_set:
        return 'flattenData'
    if 'eventId' in headers_set and 'startSample' in headers_set and 'endSample' in headers_set:
        return 'extractFeatures'
    # Heuristic: look for M000_t-0 and start/end sample
    if any(h.startswith('M000_t-0') for h in headers_set) and ('startSample' in headers_set or 'endSample' in headers_set):
        return 'extractFeatures'
    raise ValueError("Unrecognised CSV format. Expected flattenData or extractFeatures output.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare device vs local .pte streaming seizure probabilities")
    ap.add_argument('--input', required=True, help='Input CSV from flattenData.py or extractFeatures.py')

    ap.add_argument('--pte-model', required=True, help='Path to deepEpiCnnModel_pytorch.pte')

    ap.add_argument('--device-ip', default=None, help='Device IP to query (if omitted, device is skipped)')
    ap.add_argument('--device-delay-ms', type=int, default=1, help='Optional delay after each device datapoint')

    ap.add_argument('--buffer-seconds', type=float, default=30.0, help='Model buffer seconds (default 30)')
    ap.add_argument('--sample-freq', type=float, default=25.0, help='Sampling frequency Hz (default 25)')

    ap.add_argument('--normalise', action='store_true', help='Normalise buffer (match training normalise option)')
    ap.add_argument('--prob-thresh', type=float, default=0.5, help='Probability threshold for event-level detection')

    ap.add_argument('--sd-thresh', type=float, default=0.0, help='Low-motion rejection threshold (0 disables)')
    ap.add_argument('--sd-mode', choices=['pct-of-mean', 'abs'], default='pct-of-mean', help='Stddev threshold mode')

    ap.add_argument('--warmup-datapoints', type=int, default=6, help='Ignore first N accepted datapoints in metrics')

    ap.add_argument('--out-dir', default=None, help='Output directory')
    ap.add_argument('--max-events', type=int, default=None, help='Process at most N events')
    ap.add_argument('--skip-local', action='store_true', help='Skip local .pte inference (device-only run)')

    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.getcwd(), f"compare_out_{_now_tag()}")
    os.makedirs(out_dir, exist_ok=True)
    events_dir = os.path.join(out_dir, "events")
    os.makedirs(events_dir, exist_ok=True)

    # Read header to detect format
    headers = _read_csv_headers(args.input)
    fmt = _detect_format(headers)

    # Instantiate algorithms (lazy import so we can surface missing deps clearly)
    local_alg = None
    if not args.skip_local:
        try:
            from user_tools.testRunner.nnAlg import NnAlg
        except ModuleNotFoundError as e:
            raise SystemExit(
                "Local .pte runner couldn't be imported.\n"
                "This is usually one of:\n"
                "- Missing Python package in your venv (e.g. executorch)\n"
                "- Running from a CWD where legacy testRunner imports can't find sdAlg\n\n"
                "Fixes:\n"
                "- Use the repo venv: source venv/bin/activate\n"
                "- Ensure you're running this script from the repo (it adds repo paths automatically)\n\n"
                f"Missing module: {getattr(e, 'name', str(e))}\n"
                f"Original error: {e}"
            )

        local_settings = {
            'name': 'local_pte',
            'modelFname': args.pte_model,
            'sampleFreq': args.sample_freq,
            'bufferSeconds': args.buffer_seconds,
            'normalise': bool(args.normalise),
            # Keep built-in sdThresh off; script applies rejection consistently for both
            'sdThresh': 0.0,
            'probThresh': float(args.prob_thresh),
            # Make alarmState track threshold per datapoint (parity-friendly)
            'mode': 'single',
            'samplePeriod': 5.0,
            'warnTime': 0,
            'alarmTime': 0,
        }
        local_alg = NnAlg(json.dumps(local_settings), debug=False)

    device_alg = None
    if args.device_ip:
        from user_tools.testRunner.deviceAlg import DeviceAlg
        device_settings = {
            'name': 'device',
            'ipAddr': args.device_ip,
            'delayMs': int(args.device_delay_ms),
        }
        device_alg = DeviceAlg(json.dumps(device_settings), debug=False)

    # Load input data
    if fmt == 'flattenData':
        df = pd.read_csv(args.input)
    else:
        # extractFeatures files can be large; still read fully for now
        df = pd.read_csv(args.input)

    if 'eventId' not in df.columns:
        raise ValueError("Input CSV missing required column 'eventId'")

    all_dp_rows: List[pd.DataFrame] = []
    event_level_rows: List[Dict[str, Any]] = []

    # Process each event
    for ev_idx, (event_id, ev_df) in enumerate(df.groupby('eventId', sort=False)):
    
        if args.max_events is not None and ev_idx >= args.max_events:
            print(f"Maximum events limit {args.max_events} reached, stopping.")
            break

        print(f"Processing event {ev_idx + 1}: {event_id}")
        safe_id = _safe_event_id(event_id)
        event_out_csv = os.path.join(events_dir, f"event_{safe_id}_datapoints.csv")
        event_out_png = os.path.join(events_dir, f"event_{safe_id}_prob.png")

        # Reset algorithms to event boundary
        if local_alg is not None:
            local_alg.resetAlg()
        if device_alg is not None:
            try:
                device_alg.resetAlg()
            except Exception:
                pass
            # Device requires flush to clear internal buffer
            _send_zero_flush(device_alg, event_id, n_datapoints=6)

        # Build datapoint list
        if fmt == 'flattenData':
            ev_df = ev_df.sort_values('dataTime') if 'dataTime' in ev_df.columns else ev_df
            dp_sources: List[Tuple[List[float], int, Optional[str]]] = []
            for _, row in ev_df.iterrows():
                acc, hr = _extract_accel_from_flattened_row(row)
                if acc is None:
                    continue
                dt = row.get('dataTime', None)
                dp_sources.append((acc, hr if hr is not None else -1, dt if isinstance(dt, str) else None))
        else:
            samples = _reconstruct_samples_from_extractfeatures(ev_df)
            dp_chunks = _chunk_samples_to_datapoints(samples, samples_per_dp=125)
            dp_sources = [(chunk, -1, None) for chunk in dp_chunks]

        label = None
        if len(ev_df) > 0:
            label = _infer_label_from_row(ev_df.iloc[0])

        rows_out: List[Dict[str, Any]] = []
        accepted_count = 0
        for dp_i, (acc_data, hr, dt) in enumerate(dp_sources):
            rejected = _std_reject(acc_data, mode=args.sd_mode, thresh=float(args.sd_thresh))

            base_row: Dict[str, Any] = {
                'eventId': event_id,
                'dpIndex': dp_i,
                'accepted': not rejected,
                'labelSeizure': label,
                'dataTime': dt,
            }

            if rejected:
                base_row.update({
                    'valid_local': False,
                    'alarmState_local': None,
                    'pSeizure_local': None,
                    'valid_device': False if device_alg else None,
                    'alarmState_device': None,
                    'pSeizure_device': None,
                })
                rows_out.append(base_row)
                continue

            accepted_count += 1
            dp_obj = _make_dp(acc_data, hr=hr, data_time=dt)
            raw_json = libosd.dpTools.dp2rawData(dp_obj, debug=False)

            local_ret = None
            if local_alg is not None:
                local_ret = _parse_algo_json(local_alg.processDp(raw_json, str(event_id)))
            else:
                local_ret = AlgoResult(valid=False, alarm_state=None, p_seizure=None, raw=None)

            device_ret = None
            if device_alg is not None:
                device_ret = _parse_algo_json(device_alg.processDp(raw_json, str(event_id)))

            # Warmup gating for metrics (still record everything)
            warm = accepted_count <= int(args.warmup_datapoints)

            base_row.update({
                'valid_local': bool(local_ret.valid) and not warm,
                'alarmState_local': local_ret.alarm_state,
                'pSeizure_local': local_ret.p_seizure,
                'valid_device': (bool(device_ret.valid) and not warm) if device_ret else None,
                'alarmState_device': device_ret.alarm_state if device_ret else None,
                'pSeizure_device': device_ret.p_seizure if device_ret else None,
            })
            rows_out.append(base_row)

        event_df_out = pd.DataFrame(rows_out)

        # Reduce significant figures for readability in outputs.
        for c in ('pSeizure_local', 'pSeizure_device'):
            if c in event_df_out.columns:
                event_df_out[c] = pd.to_numeric(event_df_out[c], errors='coerce').round(4)

        device_alarm_pred = None
        if device_alg is not None and 'valid_device' in event_df_out.columns:
            device_valid_alarm_states = event_df_out.loc[event_df_out['valid_device'] == True, 'alarmState_device']
            device_alarm_pred = _compute_event_pred_from_alarm_state2(device_valid_alarm_states)

        event_df_out.to_csv(event_out_csv, index=False)
        _event_plot(event_out_png, event_df_out, event_id=event_id, prob_thresh=float(args.prob_thresh))

        # Event-level summary (max prob after warmup-valid points)
        local_valid_probs = event_df_out.loc[event_df_out['valid_local'] == True, 'pSeizure_local'].dropna()
        device_valid_probs = event_df_out.loc[event_df_out.get('valid_device', False) == True, 'pSeizure_device'].dropna() if device_alg else pd.Series(dtype=float)

        local_max = float(local_valid_probs.max()) if len(local_valid_probs) else None
        device_max = float(device_valid_probs.max()) if len(device_valid_probs) else None

        event_level_rows.append({
            'eventId': event_id,
            'labelSeizure': label,
            'local_maxProb': local_max,
            'device_maxProb': device_max,
            'local_pred': _compute_event_pred(local_max, float(args.prob_thresh)),
            'device_pred': _compute_event_pred(device_max, float(args.prob_thresh)) if device_alg else None,
            'device_alarm_pred': device_alarm_pred,
            'event_csv': os.path.relpath(event_out_csv, out_dir),
            'event_plot': os.path.relpath(event_out_png, out_dir),
        })

        all_dp_rows.append(event_df_out)

        # Flush device between events as required
        if device_alg is not None:
            _send_zero_flush(device_alg, event_id, n_datapoints=6)

    # Combine outputs
    all_dp_df = pd.concat(all_dp_rows, ignore_index=True) if all_dp_rows else pd.DataFrame()
    all_dp_csv = os.path.join(out_dir, 'all_datapoints.csv')
    all_dp_df.to_csv(all_dp_csv, index=False)

    event_summary_df = pd.DataFrame(event_level_rows)

    for c in ('local_maxProb', 'device_maxProb'):
        if c in event_summary_df.columns:
            event_summary_df[c] = pd.to_numeric(event_summary_df[c], errors='coerce').round(4)

    event_summary_csv = os.path.join(out_dir, 'event_summary.csv')
    event_summary_df.to_csv(event_summary_csv, index=False)

    # Metrics
    local_counts = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    local_metrics = {'TPR': float('nan'), 'FPR': float('nan'), 'ACC': float('nan')}
    if local_alg is not None:
        local_pairs = [(r.get('labelSeizure'), r.get('local_pred')) for r in event_level_rows]
        local_counts = _confusion_counts(local_pairs)
        local_metrics = _metrics_from_counts(local_counts)

    device_metrics = None
    device_counts = None
    if device_alg is not None:
        dev_pairs = [(r.get('labelSeizure'), r.get('device_pred')) for r in event_level_rows]
        device_counts = _confusion_counts(dev_pairs)
        device_metrics = _metrics_from_counts(device_counts)

    device_alarm_metrics = None
    device_alarm_counts = None
    if device_alg is not None:
        dev_alarm_pairs = [(r.get('labelSeizure'), r.get('device_alarm_pred')) for r in event_level_rows]
        device_alarm_counts = _confusion_counts(dev_alarm_pairs)
        device_alarm_metrics = _metrics_from_counts(device_alarm_counts)

    # Parity stats (on datapoints where both have probs and both valid)
    parity_df = all_dp_df.copy()
    if not parity_df.empty and device_alg is not None:
        parity_df = parity_df[(parity_df['valid_local'] == True) & (parity_df['valid_device'] == True)]
        parity_df = parity_df.dropna(subset=['pSeizure_local', 'pSeizure_device'])

    parity_stats = {}
    if device_alg is not None and not parity_df.empty:
        diff = (parity_df['pSeizure_local'].astype(float) - parity_df['pSeizure_device'].astype(float)).abs()
        parity_stats = {
            'meanAbsDiff': round(float(diff.mean()), 4),
            'maxAbsDiff': round(float(diff.max()), 4),
            'nCompared': int(len(diff)),
        }

    # Write report
    report_md = os.path.join(out_dir, 'validation_report.md')
    with open(report_md, 'w') as f:
        f.write(f"# Streaming Algorithm Validation Report\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write(f"## Inputs\n")
        f.write(f"- input: {args.input}\n")
        f.write(f"- format: {fmt}\n")
        f.write(f"- local .pte: {args.pte_model}\n")
        if args.device_ip:
            f.write(f"- device ip: {args.device_ip}\n")
        else:
            f.write(f"- device ip: (skipped)\n")
        f.write("\n")

        f.write("## Settings\n")
        f.write(f"- bufferSeconds: {args.buffer_seconds}\n")
        f.write(f"- sampleFreq: {args.sample_freq}\n")
        f.write(f"- normalise: {bool(args.normalise)}\n")
        f.write(f"- probThresh: {args.prob_thresh}\n")
        f.write(f"- sdThresh: {args.sd_thresh} ({args.sd_mode})\n")
        f.write(f"- warmupDatapoints: {args.warmup_datapoints}\n")
        f.write("\n")

        f.write("## Outputs\n")
        f.write(f"- all datapoints: {os.path.relpath(all_dp_csv, out_dir)}\n")
        f.write(f"- event summary: {os.path.relpath(event_summary_csv, out_dir)}\n")
        f.write(f"- per-event plots/tables: events/\n")
        f.write("\n")

        f.write("## Event-level Metrics (max probability per event)\n")
        f.write(f"Local CNN counts: {local_counts}\n\n")
        f.write(f"Local CNN metrics: {local_metrics}\n\n")
        if device_metrics is not None:
            f.write(f"Device CNN prob counts: {device_counts}\n\n")
            f.write(f"Device CNN prob metrics: {device_metrics}\n\n")
        if device_alarm_metrics is not None:
            f.write("## Event-level Metrics (device alarmState==2 per event)\n")
            f.write(f"Device alarmState counts: {device_alarm_counts}\n\n")
            f.write(f"Device alarmState metrics: {device_alarm_metrics}\n\n")

        if parity_stats:
            f.write("## Probability Parity (datapoints where both valid)\n")
            f.write(f"{parity_stats}\n\n")

        f.write("## Events\n")
        for r in event_level_rows:
            f.write(
                f"- event {r['eventId']}: label={r['labelSeizure']} "
                f"local_max={_fmt_prob(r.get('local_maxProb'))} device_max={_fmt_prob(r.get('device_maxProb'))} "
                f"device_alarm={r.get('device_alarm_pred')} "
                f"plot={r['event_plot']} table={r['event_csv']}\n"
            )

    # Close algorithms
    if local_alg is not None:
        try:
            local_alg.close()
        except Exception:
            pass
    if device_alg is not None:
        try:
            device_alg.close()
        except Exception:
            pass

    print(f"Wrote outputs to: {out_dir}")
    print(f"Report: {report_md}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
