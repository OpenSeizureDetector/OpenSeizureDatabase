
import pandas as pd
import numpy as np
import scipy.signal
import libosd.osdAlgTools
try:
    from user_tools.nnTraining2 import accelFeatures
    from user_tools.nnTraining2 import io_utils
except ImportError:
    import accelFeatures
    import io_utils

# Move process_event to top-level function
def process_event_simple(args):
    """
    Lightweight event processor for simpleMagnitudeOnly mode.
    Skips heavy spectral/feature calculations and only extracts raw acceleration data.
    """
    eventId, event_df, window, step, features = args
    event_df = event_df.sort_values('dataTime')
    userId = event_df['userId'].iloc[0] if 'userId' in event_df else None
    typeStr = event_df['typeStr'].iloc[0] if 'typeStr' in event_df else None
    typeVal = event_df['type'].iloc[0] if 'type' in event_df else None
    dataTime = event_df['dataTime'].iloc[-1] if 'dataTime' in event_df else np.nan
    osdAlarmState = event_df['osdAlarmState'].iloc[-1] if 'osdAlarmState' in event_df else np.nan
    
    # Determine which raw features are needed
    need_magnitude = 'acc_magnitude' in features
    need_xyz = any(f in features for f in ['accX', 'accY', 'accZ'])
    
    # Build raw axis arrays - only for needed features
    acc_mag = [] if need_magnitude else None
    accX = [] if need_xyz else None
    accY = [] if need_xyz else None
    accZ = [] if need_xyz else None
    
    for _, row in event_df.iterrows():
        if need_magnitude:
            acc_mag.extend([row.get(f"M{n:03d}_t-0", np.nan) for n in range(125)])
        if need_xyz:
            accX.extend([row.get(f"X{n:03d}_t-0", np.nan) for n in range(125)])
            accY.extend([row.get(f"Y{n:03d}_t-0", np.nan) for n in range(125)])
            accZ.extend([row.get(f"Z{n:03d}_t-0", np.nan) for n in range(125)])
    
    total_samples = len(acc_mag) if need_magnitude else len(accX)
    
    # Split into windows
    rows = []
    for start in range(0, total_samples - window + 1, step):
        end = start + window
        row_dict = {
            'eventId': eventId,
            'userId': userId,
            'typeStr': typeStr,
            'type': typeVal,
            'dataTime': dataTime,
            'osdAlarmState': osdAlarmState,
            'hr': np.nan,
            'o2sat': np.nan,
            'startSample': start,
            'endSample': end
        }
        # Only include columns for requested features
        if need_magnitude:
            for i in range(window):
                row_dict[f"M{i:03d}_t-0"] = acc_mag[start + i]
        if need_xyz:
            for i in range(window):
                row_dict[f"X{i:03d}_t-0"] = accX[start + i]
                row_dict[f"Y{i:03d}_t-0"] = accY[start + i]
                row_dict[f"Z{i:03d}_t-0"] = accZ[start + i]
        rows.append(row_dict)
    return rows


def process_event(args):
    eventId, event_df, window, step, features, highPassFreq, highPassOrder, debug = args
    event_df = event_df.sort_values('dataTime')
    acc_mag, accX, accY, accZ = [], [], [], []
    userId = event_df['userId'].iloc[0] if 'userId' in event_df else None
    typeStr = event_df['typeStr'].iloc[0] if 'typeStr' in event_df else None
    typeVal = event_df['type'].iloc[0] if 'type' in event_df else None

    # Interpolate HR and O2SAT values onto same timebase as accelerometer data.
    hr_raw, o2sat_raw, sample_indices = [], [], []
    sample_count = 0
    for _, row in event_df.iterrows():
        hr_val = row.get('hr', np.nan)
        o2sat_val = row.get('o2sat', np.nan)
        if not np.isnan(hr_val):
            hr_raw.append(hr_val)
            sample_indices.append(sample_count)
        if not np.isnan(o2sat_val):
            o2sat_raw.append(o2sat_val)
        sample_count += 125
    total_samples = len(event_df) * 125
    if len(hr_raw) == 0:
        hr_interp = np.full(total_samples, np.nan)
    else:
        hr_interp = np.interp(np.arange(total_samples), sample_indices, hr_raw)
    if len(o2sat_raw) == 0:
        o2sat_interp = np.full(total_samples, np.nan)
    else:
        o2sat_interp = np.interp(np.arange(total_samples), sample_indices, o2sat_raw)

    # Produce a single time series of each accelerometer axis, covering the entire event.
    for _, row in event_df.iterrows():
        acc_mag.extend([row.get(f"M{n:03d}_t-0", np.nan) for n in range(125)])
        accX.extend([row.get(f"X{n:03d}_t-0", np.nan) for n in range(125)])
        accY.extend([row.get(f"Y{n:03d}_t-0", np.nan) for n in range(125)])
        accZ.extend([row.get(f"Z{n:03d}_t-0", np.nan) for n in range(125)])

    # Convert lists to numpy arrays
    acc_mag = np.array(acc_mag, dtype=float)
    accX = np.array(accX, dtype=float)
    accY = np.array(accY, dtype=float)
    accZ = np.array(accZ, dtype=float)
    hr_interp = np.array(hr_interp, dtype=float)
    o2sat_interp = np.array(o2sat_interp, dtype=float)

    # Apply high pass filter to accelerometer data to remove gravity and slow movement components.
    if (highPassFreq is not None):
        acc_mag = high_pass_filter(acc_mag, cutoff=highPassFreq, fs=25, order=highPassOrder)
        accX = high_pass_filter(accX, cutoff=highPassFreq, fs=25, order=highPassOrder)
        accY = high_pass_filter(accY, cutoff=highPassFreq, fs=25, order=highPassOrder)
        accZ = high_pass_filter(accZ, cutoff=highPassFreq, fs=25, order=highPassOrder)

    # Transpose values from datapoint meta data onto accelerometer data timebase
    meta_cols = ['dataTime', 'osdAlarmState', 'osdSpecPower', 'osdRoiPower', 'hr', 'o2sat']
    meta_arrays = {col: [] for col in meta_cols}
    for _, row in event_df.iterrows():
        for col in meta_cols:
            meta_arrays[col].extend([row.get(col, np.nan)] * 125)
    meta_arrays['hr'] = hr_interp.tolist()
    meta_arrays['o2sat'] = o2sat_interp.tolist()

    # Split the data into overlapping windows/epochs
    total_samples = len(acc_mag)
    rows = []
    for start in range(0, total_samples - window + 1, step):
        end = start + window
        acc_mag_window = acc_mag[start:end]
        accX_window = accX[start:end]
        accY_window = accY[start:end]
        accZ_window = accZ[start:end]

        # Create a dictionary to hold the epoch data
        epoch_data = {
            'x': accX_window,
            'y': accY_window,
            'z': accZ_window,
            'magnitude': acc_mag_window,
            'hr': hr_interp[start:end],
            'o2sat': o2sat_interp[start:end]
        }
        # Define frequency bands for feature extraction
        freq_bands = {
            'osdRoi': (3.0, 8.0),
            'osdSpec': (0.5, 12.5),
            'osdFlap': (2.0, 4.0),
            'seizure_main': (1.0, 4.0),
            'initial_clonus': (3.0, 5.0),
            'late_clonus': (1.0, 2.0),
            '1-3Hz': (1.0, 3.0),
            '2-4Hz': (2.0, 4.0),
            '3-5Hz': (3.0, 5.0),
            '4-6Hz': (4.0, 6.0),
            '5-7Hz': (5.0, 7.0),
            '6-8Hz': (6.0, 8.0),
            '7-9Hz': (7.0, 9.0),
            '8-10Hz': (8.0, 10.0)
        }

        # Calculate features for the current epoch
        featuresObj = accelFeatures.calculate_epoch_features(epoch_data, sf=25, freq_bands=freq_bands)

        # Build the output row
        row = {
            'eventId': eventId,
            'userId': userId,
            'typeStr': typeStr,
            'type': typeVal,
            'dataTime': meta_arrays['dataTime'][end-1] if end-1 < len(meta_arrays['dataTime']) else np.nan,
            'osdAlarmState': meta_arrays['osdAlarmState'][end-1] if end-1 < len(meta_arrays['osdAlarmState']) else np.nan,
            'osdSpecPower': meta_arrays['osdSpecPower'][end-1] if end-1 < len(meta_arrays['osdSpecPower']) else np.nan,
            'osdRoiPower': meta_arrays['osdRoiPower'][end-1] if end-1 < len(meta_arrays['osdRoiPower']) else np.nan,
            'hr': meta_arrays['hr'][end-1] if end-1 < len(meta_arrays['hr']) else np.nan,
            'o2sat': meta_arrays['o2sat'][end-1] if end-1 < len(meta_arrays['o2sat']) else np.nan,
            'startSample': start,
            'endSample': end
        }
        # Always include all calculated features from featuresObj
        for feature_name, feature_value in featuresObj.items():
            row[feature_name] = feature_value

        for i in range(window):
            row[f"M{i:03d}_t-0"] = acc_mag[start + i]
        for i in range(window):
            row[f"X{i:03d}_t-0"] = accX[start + i]
        for i in range(window):
            row[f"Y{i:03d}_t-0"] = accY[start + i]
        for i in range(window):
            row[f"Z{i:03d}_t-0"] = accZ[start + i]

        rows.append(row)
    return rows

def low_pass_filter(data, cutoff=0.5, fs=25, order=4):
    """
    Applies a digital 1st-o rder Butterworth low-pass filter to time series data.
    
    Parameters:
        data (array-like): Input signal array
        cutoff (float): Cutoff frequency in Hz (default: 0.5)
        fs (int): Sampling frequency in Hz (default: 25)
        order (int): Filter order (default: 4, higher gives sharper transition but may amplify noise)
        
    Returns:
        filtered_data (array-like): The filtered signal
    """
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff / nyquist
    
    # Create Butterworth low-pass filter with given order and normalized  cutoff frequency
    b, a = scipy.signal.butter(order, normal_cutoff, analog=False)
    
    # Apply zero-phase filtering
    filtered_data = scipy.signal.filtfilt(b, a, data)
    
    return filtered_data


def high_pass_filter(data, cutoff=0.5, fs=25, order=2):
    """
    Applies a digital 1st-order Butterworth high-pass filter to time series data.
    
    Parameters:
        data (array-like): Input signal array
        cutoff (float): High-pass cutoff frequency in Hz (default: 0.5)
        fs (int): Sampling frequency in Hz (default: 25)
        order (int): Filter order (default: 4, higher gives sharper transition but may amplify noise)
        
    Returns:
        filtered_data (array-like): The high-pass filtered signal
    """
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff / nyquist
    
    # Create Butterworth high-pass filter with given order and normalized cutoff frequency
    b, a = scipy.signal.butter(order, normal_cutoff, analog=False, btype='high')
    
    # Apply zero-phase filtering
    filtered_data = scipy.signal.filtfilt(b, a, data)
    
    return filtered_data



def extract_features(df, configObj, debug=False):
    # Data processing parameters
    window = configObj['dataProcessing'].get('window', 125)
    step = configObj['dataProcessing'].get('step', window)
    features = configObj['dataProcessing']['features']
    highPassFreq = configObj['dataProcessing'].get('highPassFreq', None)
    highPassOrder = configObj['dataProcessing'].get('highPassOrder', 2)
    simpleMagnitudeOnly = configObj['dataProcessing'].get('simpleMagnitudeOnly', False)

    # Performance / IO tuning knobs (optional in config)
    worker_count = configObj.get('dataProcessing', {}).get('worker_count', None)
    batch_size = configObj.get('dataProcessing', {}).get('batch_size', 1000)
    stream_chunksize = configObj.get('dataProcessing', {}).get('stream_chunksize', 20000)
    progress_interval = configObj.get('dataProcessing', {}).get('progress_interval', 100)

    # If worker_count not set, use CPU count - 1 (leave one core free) but at least 1
    import multiprocessing as _mp
    import time
    if worker_count is None:
        try:
            worker_count = max(1, _mp.cpu_count() - 1)
        except Exception:
            worker_count = 1

    print("extract_features():  window=%d, step=%d" % (window, step))
    if highPassFreq is not None:
        print("extract_features(): highPassFreq=%.1f, highPassOrder=%d" % (highPassFreq, highPassOrder))
    else:
        print(f"extract_features(): highPassFreq=None, highPassOrder={highPassOrder}")

    # We'll compute input statistics differently depending on whether `df`
    # is an in-memory DataFrame or a filename (streaming mode).

    import multiprocessing
    print("extract_features():  grouping events by eventId")

    # If df is small (already loaded), we can process similarly to before.
    if isinstance(df, pd.DataFrame) and len(df) < 200000:
        # Statistics for input (in-memory)
        input_seizure = (df['type'] == 1).sum()
        input_nonseizure = (df['type'] == 0).sum()
        print(f"extract_features(): Input rows: {len(df)}")
        print(f"extract_features():   Seizure rows (type=1): {input_seizure}")
        print(f"extract_features():   Non-seizure rows (type=0): {input_nonseizure}")

        grouped = df.groupby('eventId', sort=False)
        total_events = grouped.ngroups
        print("extractFeatures() - analysing each event in turn.... (parallel)")
        event_args = [
            (eventId, event_df.copy(), window, step, features, highPassFreq, highPassOrder, debug)
            for eventId, event_df in grouped
        ]
        # Use imap_unordered so we can report progress as results arrive
        # Use multiprocessing to process each event. If simpleMagnitudeOnly is set
        # we can use a much lighter-weight per-event processor that skips the
        # heavy spectral/feature calculations.
        if simpleMagnitudeOnly:
            print("extractFeatures() - simpleMagnitudeOnly mode enabled; using lightweight processing")
            event_args = [
                (eventId, event_df.copy(), window, step, features)
                for eventId, event_df in grouped
            ]
            with multiprocessing.Pool(processes=worker_count) as pool:
                it = pool.imap_unordered(process_event_simple, event_args)
                out_rows = []
                processed = 0
                start_time = time.time()
                for res in it:
                    out_rows.extend(res)
                    processed += 1
                    if processed % progress_interval == 0 or processed == total_events:
                        elapsed = time.time() - start_time
                        pct = (processed / total_events) * 100
                        pct = min(100.0, pct)
                        elapsed_str = time.strftime('%Hh%Mm%Ss', time.gmtime(int(elapsed)))
                        print(f"[extractFeatures][progress] {processed}/{total_events} events ({pct:.1f}%) elapsed={elapsed_str}")
            out_df = pd.DataFrame(out_rows)
            print("extractFeatures() - finished (simpleMagnitudeOnly) processing events")
        else:
            with multiprocessing.Pool(processes=worker_count) as pool:
                it = pool.imap_unordered(process_event, event_args)
                out_rows = []
                processed = 0
                start_time = time.time()
                for res in it:
                    out_rows.extend(res)
                    processed += 1
                    if processed % progress_interval == 0 or processed == total_events:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else None
                        eta = None
                        if rate and rate > 0:
                            remaining = max(0, total_events - processed)
                            eta = remaining / rate
                        pct = (processed / total_events) * 100
                        pct = min(100.0, pct)
                        if eta is not None:
                            hrs, rem = divmod(int(eta), 3600)
                            mins, secs = divmod(rem, 60)
                            eta_str = f"{hrs:d}h{mins:02d}m{secs:02d}s" if hrs else f"{mins:02d}m{secs:02d}s"
                            elapsed_str = time.strftime('%Hh%Mm%Ss', time.gmtime(int(elapsed)))
                            eta_time_str = time.strftime('%H:%M:%S', time.localtime(time.time() + eta))
                            print(f"[extractFeatures][progress] {processed}/{total_events} events ({pct:.1f}%) rate={rate:.2f} ev/s elapsed={elapsed_str} ETA={eta_str} ETA_time={eta_time_str}")
                        else:
                            elapsed_str = time.strftime('%Hh%Mm%Ss', time.gmtime(int(elapsed)))
                            print(f"[extractFeatures][progress] {processed}/{total_events} events ({pct:.1f}%) elapsed={elapsed_str}")
            out_df = pd.DataFrame(out_rows)
            print("extractFeatures() - finished processing events")
        # continue to ordering and return
    else:
        # df is not an in-memory DataFrame of manageable size. Assume caller
        # provided a filename earlier and used streaming helpers; but if
        # df is a string filename here, handle streaming read.
        if isinstance(df, str):
            inFname = df
        else:
            raise ValueError("extract_features: expected small DataFrame or filename for streaming")

        # For streaming mode compute input statistics by scanning the file in chunks
        total_rows = 0
        seizure_count = 0
        nonseizure_count = 0
        for chunk in pd.read_csv(inFname, usecols=['type'], chunksize=200000):
            total_rows += len(chunk)
            if 'type' in chunk:
                seizure_count += (chunk['type'] == 1).sum()
                nonseizure_count += (chunk['type'] == 0).sum()

        print(f"extract_features(): Input rows: {total_rows}")
        print(f"extract_features():   Seizure rows (type=1): {seizure_count}")
        print(f"extract_features():   Non-seizure rows (type=0): {nonseizure_count}")
        # Stream events and process in parallel, writing output incrementally
        pool = multiprocessing.Pool(processes=worker_count)
        out_rows = []
        out_df = None
        first_write = True
        print("extractFeatures() - streaming events and processing in parallel")
        # Ensure we have a unique temporary filename for this run
        tmp_path = configObj['dataFileNames'].get('streamTmpOut', None)
        if tmp_path is None:
            import tempfile
            import os
            fd, tmp_path = tempfile.mkstemp(suffix='.csv')
            os.close(fd)
            configObj.setdefault('dataFileNames', {})
            configObj['dataFileNames']['streamTmpOut'] = tmp_path
        try:
            # Use imap_unordered for early results and lower memory footprint
            # Choose processor based on simpleMagnitudeOnly mode
            if simpleMagnitudeOnly:
                print("extractFeatures() - simpleMagnitudeOnly mode enabled for streaming; using lightweight processing")
                processor_func = process_event_simple
                event_args_gen = (
                    (eventId, event_df, window, step, features)
                    for eventId, event_df in io_utils.stream_events_from_flattened_csv(inFname, chunksize=stream_chunksize)
                )
            else:
                processor_func = process_event
                event_args_gen = (
                    (eventId, event_df, window, step, features, highPassFreq, highPassOrder, debug)
                    for eventId, event_df in io_utils.stream_events_from_flattened_csv(inFname, chunksize=stream_chunksize)
                )
            
            it = pool.imap_unordered(processor_func, event_args_gen)
            batch = []
            processed_events = 0
            start_time = time.time()
            # Try to compute total events for a percentage if feasible (streaming)
            total_events = None
            try:
                # Use the same event-streaming generator we use for processing to
                # compute the total number of event blocks. This guarantees the
                # count matches the processing semantics (handles chunking,
                # quoted fields, and events spanning chunk boundaries).
                total_events = 0
                for _eventId, _edf in io_utils.stream_events_from_flattened_csv(inFname, chunksize=stream_chunksize):
                    total_events += 1
            except Exception:
                total_events = None

            for rows in it:
                # rows is list of dicts
                batch.extend(rows)
                processed_events += 1
                if total_events and (processed_events % progress_interval == 0 or processed_events == total_events):
                    elapsed = time.time() - start_time
                    rate = processed_events / elapsed if elapsed > 0 else None
                    if rate and rate > 0:
                        remaining = max(0, total_events - processed_events)
                        eta = remaining / rate
                        hrs, rem = divmod(int(eta), 3600)
                        mins, secs = divmod(rem, 60)
                        eta_str = f"{hrs:d}h{mins:02d}m{secs:02d}s" if hrs else f"{mins:02d}m{secs:02d}s"
                        pct = (processed_events / total_events) * 100
                        pct = min(100.0, pct)
                        elapsed_str = time.strftime('%Hh%Mm%Ss', time.gmtime(int(elapsed)))
                        eta_time_str = time.strftime('%H:%M:%S', time.localtime(time.time() + eta))
                        print(f"[extractFeatures][progress] {processed_events}/{total_events} events ({pct:.1f}%) rate={rate:.2f} ev/s elapsed={elapsed_str} ETA={eta_str} ETA_time={eta_time_str}")
                    else:
                        pct = (processed_events / total_events) * 100
                        pct = min(100.0, pct)
                        elapsed_str = time.strftime('%Hh%Mm%Ss', time.gmtime(int(elapsed)))
                        print(f"[extractFeatures][progress] {processed_events}/{total_events} events processed ({pct:.1f}%) elapsed={elapsed_str}")
                if len(batch) >= batch_size:
                    if out_df is None:
                        # write header first time
                        io_utils.write_rows_batch_csv(tmp_path, batch, header=True, mode='a')
                        out_df = True
                    else:
                        io_utils.write_rows_batch_csv(tmp_path, batch, header=False, mode='a')
                    batch = []

            # write remaining
            if batch:
                io_utils.write_rows_batch_csv(tmp_path, batch, header=first_write, mode='a')
        finally:
            pool.close()
            pool.join()

        # Load the streamed temporary file into a DataFrame for post-processing
        # Use a unique temp file per run to avoid conflicts. If a path was
        # provided in configObj it will be used, otherwise we created one
        # earlier (io_utils writes directly to that path). To be robust we
        # support dtype mapping and low_memory flag from config.
        tmp_path = configObj['dataFileNames'].get('streamTmpOut', None)
        if tmp_path is None:
            raise RuntimeError('No temporary stream output path available')

        # Support optional dtype mapping to avoid mixed-type warnings
        dtype_map = configObj.get('dataProcessing', {}).get('stream_dtype_map', None)
        low_memory_flag = configObj.get('dataProcessing', {}).get('stream_low_memory', False)

        if dtype_map:
            out_df = pd.read_csv(tmp_path, dtype=dtype_map, low_memory=low_memory_flag)
        else:
            out_df = pd.read_csv(tmp_path, low_memory=low_memory_flag)

        # Clean up temporary streamed file to avoid accumulating large files
        try:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            # Not fatal; leave file if removal fails
            pass

    # Ensure all calculated features are included
    # Derive calculated features either from the rows computed in-memory or
    # from the output DataFrame (streaming path).
    calculated_features = set()
    if out_rows:
        for row in out_rows:
            calculated_features.update([k for k in row.keys() if k not in ['eventId', 'userId', 'typeStr', 'type', 'dataTime', 'osdAlarmState', 'osdSpecPower', 'osdRoiPower', 'hr', 'o2sat', 'startSample', 'endSample'] and not k.startswith(('M', 'X', 'Y', 'Z'))])
    else:
        # infer from out_df column names
        if out_df is not None:
            for k in out_df.columns:
                if k in ['eventId', 'userId', 'typeStr', 'type', 'dataTime', 'osdAlarmState', 'osdSpecPower', 'osdRoiPower', 'hr', 'o2sat', 'startSample', 'endSample']:
                    continue
                if k.startswith(('M', 'X', 'Y', 'Z')):
                    continue
                calculated_features.add(k)

    # Order columns: metadata, calculated features, then raw data
    meta_cols = ['eventId', 'userId', 'typeStr', 'type', 'dataTime', 'osdAlarmState', 'osdSpecPower', 'osdRoiPower', 'hr', 'o2sat', 'startSample', 'endSample']
    raw_cols = [col for col in out_df.columns if col.startswith(('M', 'X', 'Y', 'Z'))]
    feature_cols = sorted(list(calculated_features))
    ordered_cols = meta_cols + feature_cols + raw_cols
    ordered_cols = [col for col in ordered_cols if col in out_df.columns]
    out_df = out_df[ordered_cols]

    # Defensive cleanup: drop any accidental header-rows that may have been
    # appended into the streamed CSV (these typically contain literal
    # column names in metadata fields). Only check a small set of meta cols.
    meta_check_cols = ['eventId', 'userId', 'typeStr', 'type', 'dataTime']
    mask = pd.Series(False, index=out_df.index)
    for c in meta_check_cols:
        if c in out_df.columns:
            mask |= out_df[c].astype(str).str.strip().str.lower() == c.lower()
    if mask.any():
        out_df = out_df[~mask].reset_index(drop=True)

    # Ensure 'type' is numeric so comparisons below work reliably even if
    # CSV contained strings like '0'/'1'. Leave NaN for unknowns.
    if 'type' in out_df.columns:
        out_df['type'] = pd.to_numeric(out_df['type'].astype(str).str.strip(), errors='coerce')
        output_seizure = int((out_df['type'] == 1).sum())
        output_nonseizure = int((out_df['type'] == 0).sum())
    else:
        output_seizure = 0
        output_nonseizure = 0

    print(f"Output rows: {len(out_df)}")
    print(f"  Seizure rows (type=1): {output_seizure}")
    print(f"  Non-seizure rows (type=0): {output_nonseizure}")

    return out_df
def extractFeatures(inFname, outFname, configObj, debug=False):
    """
    Reads flattened CSV from inFname, extracts features, and writes to outFname.
    """
    # For large files we prefer streaming inside extract_features; pass the
    # filename through and let extract_features decide how to process.
    df_or_fname = inFname
    # provide a temporary output file for streaming writer
    stream_tmp = configObj.get('dataFileNames', {}).get('streamTmpOut', outFname + '.tmp.csv')
    configObj.setdefault('dataFileNames', {})
    configObj['dataFileNames']['streamTmpOut'] = stream_tmp

    df_feat = extract_features(df_or_fname, configObj, debug=debug)
    # write final output
    print(f"Writing features to {outFname}")
    df_feat.to_csv(outFname, index=False)
    return outFname

def main():
    import argparse
    import libosd.configUtils

    parser = argparse.ArgumentParser(description='Extract features from flattened OSDB CSV')
    parser.add_argument('--config', default="nnConfig.json")
    parser.add_argument('-i', required=True, help='Input flattened CSV')
    parser.add_argument('-o', help='Output CSV with features')
    parser.add_argument('--test', action="store_true", help='Extract features for test data')
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    configObj = libosd.configUtils.loadConfig(args.config)

    # Determine output file from config if not specified
    if args.o:
        out_csv = args.o
    else:
        if args.test:
            out_csv = configObj['dataFileNames']['testFeaturesFileCsv']
        else:
            out_csv = configObj['dataFileNames']['trainFeaturesFileCsv']

    # Pass filename to extract_features so it can stream instead of loading
    df_feat = extract_features(args.i, configObj, debug=args.debug)
    df_feat.to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()