import pandas as pd
import numpy as np
import scipy.signal
import libosd.osdAlgTools

def low_pass_filter(data, cutoff=0.5, fs=25, order=4):
    """
    Applies a digital 1st-order Butterworth low-pass filter to time series data.
    
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
    print("extract_features(): highPassFreq=%.1f, highPassOrder=%d" % (highPassFreq, highPassOrder))

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

        total_samples = len(acc_mag)
        for start in range(0, total_samples - window + 1, step):
            end = start + window
            row = {
                'eventId': eventId,
                'userId': userId,
                'typeStr': typeStr,
                'type': typeVal,
                # Use most recent value in window for these columns
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
            if 'meanLineLengthMag' in features or any(f.startswith('powerMag_') for f in features) or 'specPower' in features or 'roiPower' in features:
                mag_window = acc_mag[start:end]
            if 'meanLineLengthX' in features or any(f.startswith('powerX_') for f in features):
                x_window = accX[start:end]
            if 'meanLineLengthY' in features or any(f.startswith('powerY_') for f in features):
                y_window = accY[start:end]
            if 'meanLineLengthZ' in features or any(f.startswith('powerZ_') for f in features):
                z_window = accZ[start:end]

            # Feature extraction loop
            for feature in features:
                if feature in ["acc_magnitude", "acc_X", "acc_Y", "acc_Z", "hr", "o2sat", "osdSpecPower", "osdRoiPower"]:
                    continue
                elif feature == "specPower":
                    row["specPower"] = libosd.osdAlgTools.getSpecPower(mag_window)
                elif feature == "roiPower":
                    row["roiPower"] = libosd.osdAlgTools.getRoiPower(mag_window, debug=debug)
                elif feature.startswith("powerMag_"):
                    freq_range = feature.split("_")[1]
                    min_freq, max_freq = map(float, freq_range.split("-"))
                    row[feature] = libosd.osdAlgTools.getRoiPower(mag_window, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug)
                elif feature == "meanLineLengthMag":
                    row["meanLineLengthMag"] = libosd.osdAlgTools.getMeanLineLength(mag_window)
                elif feature.startswith("powerX_"):
                    freq_range = feature.split("_")[1]
                    min_freq, max_freq = map(float, freq_range.split("-"))
                    row[feature] = libosd.osdAlgTools.getRoiPower(x_window, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug)
                elif feature == "meanLineLengthX":
                    row["meanLineLengthX"] = libosd.osdAlgTools.getMeanLineLength(x_window)
                elif feature.startswith("powerY_"):
                    freq_range = feature.split("_")[1]
                    min_freq, max_freq = map(float, freq_range.split("-"))
                    row[feature] = libosd.osdAlgTools.getRoiPower(y_window, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug)
                elif feature == "meanLineLengthY":
                    row["meanLineLengthY"] = libosd.osdAlgTools.getMeanLineLength(y_window)
                elif feature.startswith("powerZ_"):
                    freq_range = feature.split("_")[1]
                    min_freq, max_freq = map(float, freq_range.split("-"))
                    row[feature] = libosd.osdAlgTools.getRoiPower(z_window, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug)
                elif feature == "meanLineLengthZ":
                    row["meanLineLengthZ"] = libosd.osdAlgTools.getMeanLineLength(z_window)
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