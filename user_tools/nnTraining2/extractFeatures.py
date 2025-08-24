import pandas as pd
import numpy as np
import scipy.signal
import libosd.osdAlgTools
try:
    from user_tools.nnTraining2 import accelFeatures
except ImportError:
    import accelFeatures

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
    window = configObj['dataProcessing'].get('window', 125)
    step = configObj['dataProcessing'].get('step', window)
    features = configObj['dataProcessing']['features']
    highPassFreq = configObj['dataProcessing'].get('highPassFreq',None)
    highPassOrder = configObj['dataProcessing'].get('highPassOrder',2)

    print("extract_features():  window=%d, step=%d" % (window, step))
    if highPassFreq is not None:
        print("extract_features(): highPassFreq=%.1f, highPassOrder=%d" % (highPassFreq, highPassOrder))
    else:
        print(f"extract_features(): highPassFreq=None, highPassOrder={highPassOrder}")

    # Statistics for input
    input_seizure = (df['type'] == 1).sum()
    input_nonseizure = (df['type'] == 0).sum()
    print(f"extract_features(): Input rows: {len(df)}")
    print(f"extract_features():   Seizure rows (type=1): {input_seizure}")
    print(f"extract_features():   Non-seizure rows (type=0): {input_nonseizure}")

    out_rows = []
    print("extract_features():  grouping events by eventId")
    grouped = df.groupby('eventId', sort=False)

    print("extractFeatures() - analysing each event in turn....")
    for eventId, event_df in grouped:
        event_df = event_df.sort_values('dataTime')

        acc_mag, accX, accY, accZ = [], [], [], []
        userId = event_df['userId'].iloc[0] if 'userId' in event_df else None
        typeStr = event_df['typeStr'].iloc[0] if 'typeStr' in event_df else None
        typeVal = event_df['type'].iloc[0] if 'type' in event_df else None
        
        # Interpolate hr and o2sat to match the length of accX, accY, etc.
        hr_raw, o2sat_raw, sample_indices = [], [], []
        sample_count = 0
        for _, row in event_df.iterrows():
            hr_val = row.get('hr', np.nan)
            o2sat_val = row.get('o2sat', np.nan)
            # Only add if not nan
            if not np.isnan(hr_val):
                hr_raw.append(hr_val)
                sample_indices.append(sample_count)
            if not np.isnan(o2sat_val):
                o2sat_raw.append(o2sat_val)
            sample_count += 125
        total_samples = len(event_df) * 125
        # If no valid hr/o2sat, fill with nan
        if len(hr_raw) == 0:
            hr_interp = np.full(total_samples, np.nan)
        else:
            hr_interp = np.interp(np.arange(total_samples), sample_indices, hr_raw)
        if len(o2sat_raw) == 0:
            o2sat_interp = np.full(total_samples, np.nan)
        else:
            o2sat_interp = np.interp(np.arange(total_samples), sample_indices, o2sat_raw)

        # Build full arrays for magnitude, X, Y, Z
        for _, row in event_df.iterrows():
            acc_mag.extend([row.get(f"M{n:03d}", np.nan) for n in range(125)])
            accX.extend([row.get(f"X{n:03d}", np.nan) for n in range(125)])
            accY.extend([row.get(f"Y{n:03d}", np.nan) for n in range(125)])
            accZ.extend([row.get(f"Z{n:03d}", np.nan) for n in range(125)])

        acc_mag = np.array(acc_mag, dtype=float)
        accX = np.array(accX, dtype=float)

        accY = np.array(accY, dtype=float)
        accZ = np.array(accZ, dtype=float)
        hr_interp = np.array(hr_interp, dtype=float)
        o2sat_interp = np.array(o2sat_interp, dtype=float)


        if (highPassFreq is not None):
            #print("extractFeatures() - applying high pass filter with cutoff=%.1f and order %d" % (highPassFreq, highPassOrder))
            acc_mag = high_pass_filter(acc_mag, cutoff=highPassFreq, fs=25, order=highPassOrder)
            accX = high_pass_filter(accX, cutoff=highPassFreq, fs=25, order=highPassOrder)
            accY = high_pass_filter(accY, cutoff=highPassFreq, fs=25, order=highPassOrder)
            accZ = high_pass_filter(accZ, cutoff=highPassFreq, fs=25, order=highPassOrder)


        # For aligning meta columns
        meta_cols = ['dataTime', 'osdAlarmState', 'osdSpecPower', 'osdRoiPower', 'hr', 'o2sat']
        meta_arrays = {col: [] for col in meta_cols}
        for _, row in event_df.iterrows():
            for col in meta_cols:
                meta_arrays[col].extend([row.get(col, np.nan)] * 125)


        # Overwrite meta_arrays['hr'] and meta_arrays['o2sat'] with interpolated arrays
        meta_arrays['hr'] = hr_interp.tolist()
        meta_arrays['o2sat'] = o2sat_interp.tolist()

        # Loop through the accelerometer data, creating 'epochs' of length 'window' every 'step' samples
        total_samples = len(acc_mag)
        for start in range(0, total_samples - window + 1, step):
            end = start + window

            acc_mag_window = acc_mag[start:end]
            accX_window = accX[start:end]
            accY_window = accY[start:end]
            accZ_window = accZ[start:end]

            epoch_data = {
                'x': accX_window,
                'y': accY_window,
                'z': accZ_window,
                'magnitude': acc_mag_window,
                'hr': hr_interp[start:end],
                'o2sat': o2sat_interp[start:end]
            }

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

            featuresObj = accelFeatures.calculate_epoch_features(epoch_data, sf=25, freq_bands=freq_bands)

            print(f"Event {eventId}: Extracted features for samples {start} to {end}")
            #print(f"Event {eventId}: Extracted features: {featuresObj.keys()}")

            # Populate row with meta columns
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
            # Only include magnitude/X/Y/Z arrays if requested
            if 'acc_magnitude' in features:
                for i in range(window):
                    row[f"M{i:03d}"] = acc_mag[start + i]
            if 'acc_X' in features:
                for i in range(window):
                    row[f"X{i:03d}"] = accX[start + i]
            if 'acc_Y' in features:
                for i in range(window):
                    row[f"Y{i:03d}"] = accY[start + i]
            if 'acc_Z' in features:
                for i in range(window):
                    row[f"Z{i:03d}"] = accZ[start + i]

            # Add only requested features from accelFeatures
            for feature in features:
                if feature in row:
                    continue
                # Map mean_X, mean_Y, mean_Z to mean_x, mean_y, mean_z in featuresObj
                mapped_feature = feature
                if feature.startswith('mean_') and feature[-1] in ['X','Y','Z']:
                    mapped_feature = f"mean_{feature[-1].lower()}"
                if mapped_feature in featuresObj:
                    row[feature] = featuresObj[mapped_feature]
                else:
                    print(f"extractFeatures: Unknown feature {feature} (mapped: {mapped_feature}), skipping.")
            # Only include magnitude/X/Y/Z arrays if requested
            if 'acc_magnitude' in features:
                for i in range(window):
                    row[f"M{i:03d}"] = acc_mag[start + i]
            if 'acc_X' in features:
                for i in range(window):
                    row[f"X{i:03d}"] = accX[start + i]
            if 'acc_Y' in features:
                for i in range(window):
                    row[f"Y{i:03d}"] = accY[start + i]
            if 'acc_Z' in features:
                for i in range(window):
                    row[f"Z{i:03d}"] = accZ[start + i]

            # Feature extraction loop
            for feature in features:
                if feature in ["acc_magnitude", "acc_X", "acc_Y", "acc_Z", "hr", "o2sat", "osdSpecPower", "osdRoiPower"]:
                    continue
                if feature in featuresObj:
                    continue
                elif feature == "specPower":
                    row["specPower"] = libosd.osdAlgTools.getSpecPower(acc_mag_window)
                elif feature == "roiPower":
                    row["roiPower"] = libosd.osdAlgTools.getRoiPower(acc_mag_window, debug=debug)
                elif feature.startswith("powerMag_"):
                    freq_range = feature.split("_")[1]
                    min_freq, max_freq = map(float, freq_range.split("-"))
                    row[feature] = libosd.osdAlgTools.getRoiPower(acc_mag_window, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug)
                elif feature == "meanLineLengthMag":
                    row["meanLineLengthMag"] = libosd.osdAlgTools.getMeanLineLength(acc_mag_window)
                elif feature.startswith("powerX_"):
                    freq_range = feature.split("_")[1]
                    min_freq, max_freq = map(float, freq_range.split("-"))
                    row[feature] = libosd.osdAlgTools.getRoiPower(accX_window, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug)
                elif feature == "meanLineLengthX":
                    row["meanLineLengthX"] = libosd.osdAlgTools.getMeanLineLength(accX_window)
                elif feature.startswith("powerY_"):
                    freq_range = feature.split("_")[1]
                    min_freq, max_freq = map(float, freq_range.split("-"))
                    row[feature] = libosd.osdAlgTools.getRoiPower(accY_window, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug)
                elif feature == "meanLineLengthY":
                    row["meanLineLengthY"] = libosd.osdAlgTools.getMeanLineLength(accY_window)
                elif feature.startswith("powerZ_"):
                    freq_range = feature.split("_")[1]
                    min_freq, max_freq = map(float, freq_range.split("-"))
                    row[feature] = libosd.osdAlgTools.getRoiPower(accZ_window, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug)
                elif feature == "meanLineLengthZ":
                    row["meanLineLengthZ"] = libosd.osdAlgTools.getMeanLineLength(accZ_window)
                else:
                    print(f"extractFeatures: Unknown feature {feature}, skipping.")

            out_rows.append(row)

    out_df = pd.DataFrame(out_rows)

    # Statistics for output
    output_seizure = (out_df['type'] == 1).sum()
    output_nonseizure = (out_df['type'] == 0).sum()
    print(f"Output rows: {len(out_df)}")
    print(f"  Seizure rows (type=1): {output_seizure}")
    print(f"  Non-seizure rows (type=0): {output_nonseizure}")

    return out_df

def extractFeatures(inFname, outFname, configObj, debug=False):
    """
    Reads flattened CSV from inFname, extracts features, and writes to outFname.
    """
    df = pd.read_csv(inFname)
    df_feat = extract_features(df, configObj, debug=debug)
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
    df = pd.read_csv(args.i)

    # Determine output file from config if not specified
    if args.o:
        out_csv = args.o
    else:
        if args.test:
            out_csv = configObj['dataFileNames']['testFeaturesFileCsv']
        else:
            out_csv = configObj['dataFileNames']['trainFeaturesFileCsv']

    df_feat = extract_features(df, configObj, debug=args.debug)
    df_feat.to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()