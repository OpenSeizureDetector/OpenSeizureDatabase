import os
import json
import shutil
import copy
import pandas as pd
import matplotlib.pyplot as plt
from runSequence import run_sequence

def vary_param(config, param, values):
    configs = []
    for v in values:
        new_cfg = copy.deepcopy(config)
        # Support nested keys for modelConfig and dataProcessing
        if param in new_cfg.get('modelConfig', {}):
            new_cfg['modelConfig'][param] = v
        elif param in new_cfg.get('dataProcessing', {}):
            new_cfg['dataProcessing'][param] = v
        else:
            new_cfg[param] = v
        configs.append((v, new_cfg))
    return configs

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
    # Copy data files for reuse
    data_files = [k for k in base_config['dataFileNames'].values()]
    feature_files = [base_config['dataFileNames'].get('trainFeaturesFileCsv'),
                    base_config['dataFileNames'].get('testFeaturesFileCsv')]
    for param in params_to_vary:
        for v, cfg in vary_param(base_config, param, param_values[param]):
            run_name = f'{param}_{v}'
            run_out = os.path.join(out_dir, run_name)
            os.makedirs(run_out, exist_ok=True)
            cfg_path = os.path.join(run_out, 'nnConfig.json')
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            # Pre-populate data files, but skip feature files if param affects features
            skip_features = param in ['window', 'step', 'highPassOrder']
            for fname in data_files:
                if skip_features and fname in feature_files:
                    continue
                src = os.path.join(baseline_out, fname)
                dst = os.path.join(run_out, fname)
                if os.path.exists(src):
                    shutil.copy(src, dst)
            args = {
                'config': cfg_path,
                'kfold': 1,
                'rerun': 1,
                'outDir': run_out,
                'train': True,
                'test': False,
                'clean': False,
                'debug': False
            }
            run_sequence(args)
            # Collect results
            result_json = os.path.join(run_out, 'kfold_summary.json')
            if os.path.exists(result_json):
                with open(result_json, 'r') as f:
                    res = json.load(f)
                res['param'] = param
                res['value'] = v
                results.append(res)
    return results

def analyse_results(results, out_dir):
    df = pd.DataFrame(results)
    summary_path = os.path.join(out_dir, 'optimisation_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f'Summary written to {summary_path}')
    # Plot sensitivity for each parameter
    for param in df['param'].unique():
        sub = df[df['param'] == param]
        plt.figure()
        plt.plot(sub['value'], sub['accuracy'], marker='o')
        plt.xlabel(param)
        plt.ylabel('Accuracy')
        plt.title(f'Sensitivity of accuracy to {param}')
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f'{param}_sensitivity.png'))
        # Estimate optimum
        opt_idx = sub['accuracy'].idxmax()
        opt_val = sub.loc[opt_idx, 'value']
        print(f'Optimum {param}: {opt_val} (accuracy={sub.loc[opt_idx, "accuracy"]:.4f})')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Optimise meta parameters for seizure model')
    parser.add_argument('--config', default='nnConfig.json', help='Baseline config file')
    parser.add_argument('--outDir', default='./optimisation_output', help='Output directory')
    parser.add_argument('--analyse-only', action='store_true', help='Only analyse pre-calculated results')
    args = parser.parse_args()

    base_config_path = args.config
    out_dir = args.outDir
    params_to_vary = ['n_estimators', 'max_depth', 'window', 'step', 'highPassOrder']
    param_values = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'window': [125, 250, 375, 500],
        'step': [25, 50, 125],
        'highPassOrder': [2, 4, 8]
    }

    if args.analyse_only:
        # Aggregate all kfold_summary.json files in out_dir
        results = []
        for root, dirs, files in os.walk(out_dir):
            for fname in files:
                if fname == 'kfold_summary.json':
                    fpath = os.path.join(root, fname)
                    with open(fpath, 'r') as f:
                        res = json.load(f)
                    # Try to infer param/value from folder name
                    folder = os.path.basename(root)
                    for param in params_to_vary:
                        if folder.startswith(param + '_'):
                            val = folder[len(param)+1:]
                            try:
                                val = int(val)
                            except:
                                pass
                            res['param'] = param
                            res['value'] = val
                    results.append(res)
        analyse_results(results, out_dir)
    else:
        results = run_and_collect(base_config_path, params_to_vary, param_values, out_dir)
        analyse_results(results, out_dir)

if __name__ == '__main__':
    main()
