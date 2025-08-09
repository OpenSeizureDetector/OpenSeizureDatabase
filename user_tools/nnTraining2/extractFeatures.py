import pandas as pd
import numpy as np
import libosd.osdAlgTools

def extract_features(df, configObj, debug=False):
    # Assume rawData columns are named M000, M001, ..., M124
    acc_cols = [f"M{n:03d}" for n in range(125)]
    # If 3D data is present, assume columns: rawData3D_X_0 ... rawData3D_Z_124
    accX_cols = [f"rawData3D_X_{n}" for n in range(125)]
    accY_cols = [f"rawData3D_Y_{n}" for n in range(125)]
    accZ_cols = [f"rawData3D_Z_{n}" for n in range(125)]

    # Feature extraction loop
    for feature in configObj['dataProcessing']['features']:
        if feature == "acc_magnitude":
            df[[f"M{n:03d}" for n in range(125)]] = df[acc_cols]
            # Already present in flattened CSV
        elif feature == "hr":
            # Already present
            continue
        elif feature == "o2sat":
            # Already present
            continue
        elif feature == "specPower":
            df["specPower"] = df[acc_cols].apply(lambda row: libosd.osdAlgTools.getSpecPower(row.values), axis=1)
        elif feature == "roiPower":
            df["roiPower"] = df[acc_cols].apply(lambda row: libosd.osdAlgTools.getRoiPower(row.values, debug=debug), axis=1)
        elif feature.startswith("powerMag_"):
            freq_range = feature.split("_")[1]
            min_freq, max_freq = map(float, freq_range.split("-"))
            df[feature] = df[acc_cols].apply(lambda row: libosd.osdAlgTools.getRoiPower(row.values, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug), axis=1)
        elif feature == "meanLineLengthMag":
            df["meanLineLengthMag"] = df[acc_cols].apply(lambda row: libosd.osdAlgTools.getMeanLineLength(row.values), axis=1)
        elif feature.startswith("powerX_"):
            freq_range = feature.split("_")[1]
            min_freq, max_freq = map(float, freq_range.split("-"))
            df[feature] = df[accX_cols].apply(lambda row: libosd.osdAlgTools.getRoiPower(row.values, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug), axis=1)
        elif feature == "meanLineLengthX":
            df["meanLineLengthX"] = df[accX_cols].apply(lambda row: libosd.osdAlgTools.getMeanLineLength(row.values), axis=1)
        elif feature.startswith("powerY_"):
            freq_range = feature.split("_")[1]
            min_freq, max_freq = map(float, freq_range.split("-"))
            df[feature] = df[accY_cols].apply(lambda row: libosd.osdAlgTools.getRoiPower(row.values, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug), axis=1)
        elif feature == "meanLineLengthY":
            df["meanLineLengthY"] = df[accY_cols].apply(lambda row: libosd.osdAlgTools.getMeanLineLength(row.values), axis=1)
        elif feature.startswith("powerZ_"):
            freq_range = feature.split("_")[1]
            min_freq, max_freq = map(float, freq_range.split("-"))
            df[feature] = df[accZ_cols].apply(lambda row: libosd.osdAlgTools.getRoiPower(row.values, alarmFreqMin=min_freq, alarmFreqMax=max_freq, debug=debug), axis=1)
        elif feature == "meanLineLengthZ":
            df["meanLineLengthZ"] = df[accZ_cols].apply(lambda row: libosd.osdAlgTools.getMeanLineLength(row.values), axis=1)
        else:
            print(f"extractFeatures: Unknown feature {feature}, skipping.")

    return df

def main():
    import argparse
    import libosd.configUtils

    parser = argparse.ArgumentParser(description='Extract features from flattened OSDB CSV')
    parser.add_argument('--config', default="flattenConfig.json")
    parser.add_argument('-i', required=True, help='Input flattened CSV')
    parser.add_argument('-o', required=True, help='Output CSV with features')
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    configObj = libosd.configUtils.loadConfig(args.config)
    df = pd.read_csv(args.i)
    df_feat = extract_features(df, configObj, debug=args.debug)
    df_feat.to_csv(args.o, index=False)

if __name__ == "__main__":
    main()