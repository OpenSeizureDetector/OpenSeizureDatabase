import os
import json
import shutil
import copy
import pandas as pd
from runSequence import run_sequence

def run_and_collect(base_config_path, params_to_vary, param_values, out_dir='./optimisation_output'):
    os.makedirs(out_dir, exist_ok=True)
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    results = []
    # Run baseline
    baseline_out = os.path.join(out_dir, 'baseline')
    os.makedirs(baseline_out, exist_ok=True)
    baseline_cfg_path = os.path.join(baseline_out, 'nnConfig.json')
    with open(baseline_cfg_path, 'w') as f:
        json.dump(base_config, f, indent=2)
    args = {
        'config': baseline_cfg_path,
        'kfold': 1,
        'rerun': 0,
        'outDir': baseline_out,
        'train': True,
        'test': False,
        'clean': False,
        'debug': False
    }
    run_sequence(args)
    baseline_result_path = os.path.join(baseline_out, 'rfModel', '1', 'kfold_summary.json')
    if os.path.exists(baseline_result_path):
        with open(baseline_result_path, 'r') as f:
            res = json.load(f)[0]  # The json file is an array of objects - we just use the first as we are not using k-fold validation
        res['param'] = 'baseline'
        res['value'] = None
        results.append(res)

    # Sweep each parameter
    print("\n\nBaseline run complete - sweeping each parameter...")
    for param in params_to_vary:
        print(f"\n--------------------------------\nSweeping parameter: {param}")
        for val in param_values[param]:
            print(f" - Testing {param} = {val}")
            run_out = os.path.join(out_dir, f'{param}_{val}')
            print(f" - Output directory: {run_out}")
            os.makedirs(run_out, exist_ok=True)
            config = copy.deepcopy(base_config)
            if param in config:
                config[param] = val
            elif param in config.get('modelConfig', {}):
                config['modelConfig'][param] = val
            elif param in config.get('dataProcessing', {}):
                config['dataProcessing'][param] = val
            config_path = os.path.join(run_out, 'nnConfig.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            args = {
                'config': config_path,
                'outDir': run_out,
                'rerun': 1,
                'debug': False,
                'train': True,
                'test': False,
                'kfold': 1,
                'clean': False
            }

            baseline_rf_dir = os.path.join(baseline_out, 'rfModel', '1')
            sweep_rf_dir = os.path.join(run_out, 'rfModel', '1')
            os.makedirs(sweep_rf_dir, exist_ok=True)
            dataFilesLst = ["allData.json", "testData.json", "trainData.json", "testData.csv", "trainData.csv", "trainDataAugmented.csv"]
            if param not in ["window", "step", "highPassOrder"]:
                dataFilesLst.extend(["testBalancedFile.csv", "trainFeatures.csv", "testFeatures.csv"])  
            print("Copying baseline files to save re-generating them..", dataFilesLst)
            for fname in os.listdir(baseline_rf_dir):
                if fname in dataFilesLst:
                    src = os.path.join(baseline_rf_dir, fname)
                    dst = os.path.join(sweep_rf_dir, fname)
                    if os.path.isfile(src):
                        print(f"Copying {fname} from baseline")
                        shutil.copy2(src, dst)
                    else:
                        print(f"ERROR: {src} is not a file.") 
                        exit(-1)
            run_sequence(args)
            result_path = os.path.join(run_out, 'rfModel', '1', 'kfold_summary.json')
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    res = json.load(f)[0]  # The json file is an array of objects - we just use the first as we are not using k-fold validation
                res['param'] = param
                res['value'] = val
                results.append(res)
    return results

def main():
    print("Starting main()")
    params_to_vary = ['n_estimators', 'max_depth', 'window', 'step', 'highPassOrder']
    param_values = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'window': [125, 250, 375, 500],
        'step': [25, 50, 125],
        'highPassOrder': [2, 4, 8]
    }

    import argparse
    parser = argparse.ArgumentParser(description='Optimise meta parameters for seizure model')
    parser.add_argument('--config', default='nnConfig.json', help='Baseline config file')
    parser.add_argument('--outDir', default='./optimisation_output', help='Output directory')
    parser.add_argument('--analyse-only', action='store_true', help='Only analyse pre-calculated results')
    args = parser.parse_args()

    base_config_path = args.config
    out_dir = args.outDir

    if args.analyse_only:
        print("Starting analysis-only mode...")
        results = []
        for root, dirs, files in os.walk(out_dir):
            for fname in files:
                if fname == 'kfold_summary.json':
                    fpath = os.path.join(root, fname)
                    with open(fpath, 'r') as f:
                        res = json.load(f)[0]  # The json file is an array of objects - we just use the first as we are not using k-fold validation
                    folder = os.path.basename(root)
                    for param in params_to_vary:
                        if folder.startswith(param + '_'):
                            val = folder[len(param)+1:]
                            try:
                                val = int(val)
                            except:
                                pass
                            print(res)
                            res['param'] = param
                            res['value'] = val
                    results.append(res)
        print(f"Loaded {len(results)} result sets for analysis.")
    else:
        print("Starting full optimisation run...")
        results = run_and_collect(base_config_path, params_to_vary, param_values, out_dir)
        print(f"Collected {len(results)} result sets from optimisation runs.")

    print("Beginning analysis of results...")
    df = pd.DataFrame(results)
    print("Results DataFrame created.")
    csv_path = os.path.join(out_dir, 'optimisation_summary.csv')
    json_path = os.path.join(out_dir, 'optimisation_summary.json')
    df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as jf:
        json.dump(results, jf, indent=2)
    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved summary JSON to {json_path}")

    for param in params_to_vary:
        print(f"\nParameter: {param}")
        subset = df[df['param'] == param]
        if not subset.empty:
            print(subset[['value', 'tpr', 'fpr', 'event_tpr', 'event_fpr']])
        else:
            print("No results for this parameter.")

    print("Analysis complete.")

if __name__ == '__main__':
    main()
