"""Add feature history columns to per-event feature CSVs.

Clean single implementation with useful debug prints.
"""

import os
import json
import time
import multiprocessing
import traceback
from typing import Optional

import pandas as pd

try:
    from user_tools.nnTraining2 import io_utils
except Exception:
    import io_utils


def _worker_build_rows(args):
    eventId, group, n_history, feature_cols, keep_cols = args
    if len(group) < n_history:
        return []
    g = group.reset_index(drop=True)
    out = []
    for i in range(n_history - 1, len(g)):
        row = {}
        for col in keep_cols:
            if col in g.columns:
                row[col] = g.loc[i, col]
        for col in feature_cols:
            for h in range(n_history):
                row[f"{col}_t-{n_history-1-h}"] = g.loc[i-h, col]
        out.append(row)
    return out


def process_feature_history(input_csv, output_csv, event_col: str, n_history: int,
                            configObj: Optional[dict] = None, debug: bool = False):
    cfg = configObj.get('dataProcessing', {}) if configObj else {}
    batch_size = cfg.get('batch_size', 1000)
    stream_chunksize = cfg.get('stream_chunksize', 20000)
    progress_interval = cfg.get('progress_interval', 100)
    worker_count = cfg.get('worker_count', None)
    if worker_count is None:
        try:
            worker_count = max(1, multiprocessing.cpu_count() - 1)
        except Exception:
            worker_count = 1

    # determine debug flag (explicit arg OR configObj.debug OR configObj.dataProcessing.debug)
    cfg_debug = False
    if configObj:
        cfg_debug = bool(configObj.get('debug') or configObj.get('dataProcessing', {}).get('debug', False))
    debug_enabled = bool(debug or cfg_debug)

    print("cfg_debug:", cfg_debug, debug, debug_enabled)

    def dbg_print(*a, **kw):
        if debug_enabled:
            print(*a, **kw)

    # Fast path - DataFrame input
    if hasattr(input_csv, 'columns'):
        df = input_csv
        raw_cols = ['x', 'y', 'z', 'magnitude']
        exclude_cols = [event_col, 'eventId', 'userId', 'typeStr', 'type', 'dataTime',
                        'osdAlarmState', 'osdSpecPoewr', 'osdRoiPower', 'startSample', 'endSample']
        feature_cols = [c for c in df.columns if c not in raw_cols and c not in exclude_cols]
        keep_cols = [c for c in df.columns if c not in feature_cols]

        grouped = list(df.groupby(event_col, sort=False))
        args = [(eid, grp.copy(), n_history, feature_cols, keep_cols) for eid, grp in grouped]

        out_rows = []
        processed = 0
        start = time.time()
        dbg_print(f"[process_feature_history] in-memory: total_events={len(args)} workers={worker_count}")
        with multiprocessing.Pool(processes=worker_count) as pool:
            for rows in pool.imap(_worker_build_rows, args):
                if rows:
                    out_rows.extend(rows)
                processed += 1
                if processed % progress_interval == 0 or processed == len(args):
                    elapsed = time.time() - start
                    rate = processed / elapsed if elapsed > 0 else None
                    pct = (processed / len(args)) * 100
                    pct = min(100.0, pct)
                    print(f"[addFeatureHistory][progress] {processed}/{len(args)} events ({pct:.1f}%) rate={rate if rate else 'N/A'}")

        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(output_csv, index=False)
        print(f"Saved {len(out_df)} rows to {output_csv}")
        return

    # Streaming path - input is filename
    inFname = input_csv
    tmp_out = None
    if configObj is not None:
        tmp_out = configObj.get('dataFileNames', {}).get('streamTmpOut', None)
    if tmp_out is None:
        import tempfile
        fd, tmp_out = tempfile.mkstemp(suffix='.csv')
        os.close(fd)

    write_target = tmp_out
    batch = []
    feature_cols = None
    keep_cols = None
    processed_events = 0
    start = time.time()

    # quick input checks
    try:
        exists = os.path.exists(inFname)
        size = os.path.getsize(inFname) if exists else None
        dbg_print(f"[process_feature_history] start input={inFname!r} exists={exists} size={size} n_history={n_history} debug={debug}")
        if exists:
            with open(inFname, 'rb') as _f:
                head = _f.read(1024)
            dbg_print(f"[process_feature_history] opened input ok, head_bytes={len(head)}")
    except Exception:
        print("[process_feature_history][ERROR] input file sanity checks failed")
        traceback.print_exc()

    # estimate total events
    total_events = None
    try:
        total_events = sum(1 for _ in io_utils.stream_events_from_flattened_csv(inFname, chunksize=stream_chunksize))
    except Exception:
        total_events = None
    dbg_print(f"[process_feature_history] estimated total_events={total_events}")

    # generator yields worker args and prints samples
    ev_sample_count = 0

    def event_gen_debug():
        nonlocal ev_sample_count, feature_cols, keep_cols
        for eventId, group in io_utils.stream_events_from_flattened_csv(inFname, chunksize=stream_chunksize):
            ev_sample_count += 1
            if feature_cols is None:
                raw_cols = ['x', 'y', 'z', 'magnitude']
                exclude_cols = [event_col, 'eventId', 'userId', 'typeStr', 'type', 'dataTime', 'osdAlarmState',
                                'osdSpecPoewr', 'osdRoiPower', 'startSample', 'endSample']
                feature_cols = [c for c in group.columns if c not in raw_cols and c not in exclude_cols]
                keep_cols = [c for c in group.columns if c not in feature_cols]
            if ev_sample_count <= 5:
                try:
                    print(f"[process_feature_history] sample event #{ev_sample_count}: eventId={eventId!r} rows={len(group)} feature_cols={len(feature_cols)}")
                except Exception:
                    print(f"[process_feature_history] sample event #{ev_sample_count}: eventId={eventId!r} (cannot show group)")
            yield (eventId, group.copy(), n_history, feature_cols, keep_cols)

    print(f"[process_feature_history] launching pool with worker_count={worker_count}")
    with multiprocessing.Pool(processes=worker_count) as pool:
        for rows in pool.imap(_worker_build_rows, event_gen_debug()):
            if rows:
                batch.extend(rows)
                # detailed worker-return info is debug-only
                dbg_print(f"[process_feature_history] worker returned {len(rows)} rows; batch size now {len(batch)}")
            processed_events += 1
            if processed_events % progress_interval == 0 or (total_events and processed_events == total_events):
                elapsed = time.time() - start
                rate = processed_events / elapsed if elapsed > 0 else None
                if rate and total_events:
                    remaining = max(0, total_events - processed_events)
                    eta = remaining / rate
                    hrs, rem = divmod(int(eta), 3600)
                    mins, secs = divmod(rem, 60)
                    eta_str = f"{hrs:d}h{mins:02d}m{secs:02d}s" if hrs else f"{mins:02d}m{secs:02d}s"
                    pct = (processed_events / total_events) * 100
                    pct = min(100.0, pct)
                    elapsed_str = time.strftime('%Hh%Mm%Ss', time.gmtime(int(elapsed)))
                    print(f"[addFeatureHistory][progress] {processed_events}/{total_events} events ({pct:.1f}%) rate={rate:.2f} ev/s elapsed={elapsed_str} ETA={eta_str}")
                else:
                    elapsed_str = time.strftime('%Hh%Mm%Ss', time.gmtime(int(elapsed)))
                    print(f"[addFeatureHistory][progress] {processed_events} events processed elapsed={elapsed_str}")
            if batch_size and len(batch) >= batch_size:
                header_needed = (not os.path.exists(write_target)) or (os.path.exists(write_target) and os.path.getsize(write_target) == 0)
                dbg_print(f"[process_feature_history] writing batch of {len(batch)} rows to {write_target} header={header_needed}")
                io_utils.write_rows_batch_csv(write_target, batch, header=header_needed, mode='a')
                batch = []

    # summary of processing (keep this as normal output)
    print(f"[process_feature_history] pool finished processed_events={processed_events} final_batch_size={len(batch)}")
    if batch:
        header_needed = (not os.path.exists(write_target)) or (os.path.exists(write_target) and os.path.getsize(write_target) == 0)
        dbg_print(f"[process_feature_history] final write batch of {len(batch)} rows to {write_target} header={header_needed}")
        io_utils.write_rows_batch_csv(write_target, batch, header=header_needed, mode='a')

    if not os.path.exists(write_target):
        raise RuntimeError('No output written to {}'.format(write_target))
    else:
        print(f"[process_feature_history] output written to {write_target}")

    print(f"[process_feature_history] reading output from {write_target} in streaming mode")

    # Stream the temp file in chunks, filter accidental header rows and write incrementally
    meta_check_cols = ['eventId', 'userId', 'typeStr', 'type', 'dataTime']
    rows_written = 0
    first_write = True
    # use stream_chunksize (falls back to default set earlier)
    chunksize = stream_chunksize or 10000
    try:
        for chunk in pd.read_csv(write_target, chunksize=chunksize):
            # build mask for header-like rows in this chunk
            mask = pd.Series(False, index=chunk.index)
            for c in meta_check_cols:
                if c in chunk.columns:
                    mask |= chunk[c].astype(str).str.strip().str.lower() == c.lower()
            if mask.any():
                chunk = chunk[~mask].reset_index(drop=True)
            if len(chunk) == 0:
                continue
            chunk.to_csv(output_csv, mode='w' if first_write else 'a', header=first_write, index=False)
            rows_written += len(chunk)
            first_write = False
    except pd.errors.EmptyDataError:
        # no data written
        pass

    print(f"Saved {rows_written} rows to {output_csv}")


def add_feature_history(configObj, foldOutFolder=None):
    n_history = configObj.get('dataProcessing', {}).get('nHistory', 1)
    dataFileNames = configObj.get('dataFileNames', {})
    event_col = 'eventId'

    def full_path(fname):
        return os.path.join(foldOutFolder, fname) if foldOutFolder else fname

    print(f"[addFeatureHistory] called with foldOutFolder={foldOutFolder!r} n_history={n_history}")

    # Train
    train_input = full_path(dataFileNames.get('trainFeaturesFileCsv', 'trainFeatures.csv'))
    train_output = full_path(dataFileNames.get('trainFeaturesHistoryFileCsv', 'trainDataFeaturesHistory.csv'))
    print(f"[addFeatureHistory] train_input={train_input} exists={os.path.exists(train_input)}")
    try:
        process_feature_history(train_input, train_output, event_col, n_history, configObj=configObj, debug=False)
    except Exception:
        print("[addFeatureHistory][ERROR] processing train features")
        traceback.print_exc()

    # Test
    test_input = full_path(dataFileNames.get('testFeaturesFileCsv', 'testFeatures.csv'))
    test_output = full_path(dataFileNames.get('testFeaturesHistoryFileCsv', 'testDataFeaturesHistory.csv'))
    print(f"[addFeatureHistory] test_input={test_input} exists={os.path.exists(test_input)}")
    try:
        process_feature_history(test_input, test_output, event_col, n_history, configObj=configObj, debug=False)
    except Exception:
        print("[addFeatureHistory][ERROR] processing test features")
        traceback.print_exc()




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add feature history columns to features CSV')
    parser.add_argument('--config', default='nnConfig.json', help='Path to config JSON file (default: nnConfig.json)')
    parser.add_argument('--folder', default=None, help='Output folder for fold (default: None)')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configObj = json.load(f)
    add_feature_history(configObj, foldOutFolder=args.folder)

