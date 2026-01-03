#!/usr/bin/env python3
"""
Ensemble Predictor - Combines predictions from multiple k-fold models using Harrell-Davis quantile.

Based on Spahr et al. 2025 methodology for tunable sensitivity seizure detection.

Usage:
    python ensemblePredictor.py --config nnConfig_deep_pytorch.json --outputDir ./output/model_name/run_number --quantile 0.6
    
    Or test on specific data:
    python ensemblePredictor.py --config nnConfig_deep_pytorch.json --outputDir ./output/model_name/run_number --quantile 0.6 --testData path/to/test.csv
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.configUtils


def harrell_davis_quantile(values, quantile):
    """
    Calculate the Harrell-Davis quantile estimator.
    
    The Harrell-Davis quantile uses a weighted average of order statistics,
    providing a smooth and robust quantile estimate.
    
    Args:
        values: Array of prediction scores from different models
        quantile: Desired quantile (0-1)
    
    Returns:
        Weighted quantile value
    """
    from scipy.stats import beta
    
    n = len(values)
    if n == 0:
        return 0.0
    
    # Sort values
    sorted_values = np.sort(values)
    
    # Calculate weights using beta distribution
    # a and b parameters for beta distribution
    a = (n + 1) * quantile
    b = (n + 1) * (1 - quantile)
    
    # Calculate weights for each order statistic
    weights = np.zeros(n)
    for i in range(n):
        # CDF values at boundaries of order statistic
        lower = beta.cdf((i) / n, a, b) if i > 0 else 0
        upper = beta.cdf((i + 1) / n, a, b)
        weights[i] = upper - lower
    
    # Weighted sum
    result = np.sum(weights * sorted_values)
    
    return result


def simple_quantile(values, quantile):
    """
    Simple numpy quantile for comparison.
    
    Args:
        values: Array of prediction scores
        quantile: Desired quantile (0-1)
    
    Returns:
        Quantile value
    """
    return np.quantile(values, quantile)


class EnsemblePredictor:
    """Ensemble predictor using Harrell-Davis quantile aggregation."""
    
    def __init__(self, config_path, output_dir, quantile=0.6, use_harrell_davis=True):
        """
        Initialize ensemble predictor.
        
        Args:
            config_path: Path to configuration file
            output_dir: Base output directory containing fold subdirectories
            quantile: Quantile value for aggregation (0-1)
            use_harrell_davis: If True, use Harrell-Davis quantile; otherwise use simple quantile
        """
        self.config = libosd.configUtils.loadConfig(config_path)
        self.output_dir = Path(output_dir)
        self.quantile = quantile
        self.use_harrell_davis = use_harrell_davis
        self.models = []
        self.fold_dirs = []
        
    def load_models(self):
        """Load all trained models from fold directories."""
        print("\n" + "="*80)
        print("LOADING ENSEMBLE MODELS")
        print("="*80)
        
        # Find all fold directories
        fold_pattern = self.output_dir / "fold*"
        self.fold_dirs = sorted(glob.glob(str(fold_pattern)))
        
        if not self.fold_dirs:
            print(f"ERROR: No fold directories found in {self.output_dir}")
            return False
        
        print(f"\nFound {len(self.fold_dirs)} fold directories:")
        
        # Determine framework
        framework = self.config['modelConfig'].get('framework', 
                   self.config['modelConfig'].get('modelType', 'tensorflow'))
        
        # Load each model
        for fold_dir in self.fold_dirs:
            fold_path = Path(fold_dir)
            fold_name = fold_path.name
            
            # Find model file
            model_fname_root = self.config['modelConfig']['modelFname']
            
            if framework == 'pytorch':
                model_files = list(fold_path.glob(f"{model_fname_root}*.pth"))
            else:  # tensorflow
                model_files = list(fold_path.glob(f"{model_fname_root}*.keras")) + \
                             list(fold_path.glob(f"{model_fname_root}*.h5"))
            
            if model_files:
                model_path = model_files[0]
                print(f"  {fold_name}: {model_path.name}")
                self.models.append(str(model_path))
            else:
                print(f"  {fold_name}: WARNING - No model file found")
        
        print(f"\nLoaded {len(self.models)} models")
        return len(self.models) > 0
    
    def predict_single_model(self, model_path, X):
        """
        Get predictions from a single model.
        
        Args:
            model_path: Path to model file
            X: Input features
            
        Returns:
            Prediction probabilities (N x 2 array for binary classification)
        """
        framework = self.config['modelConfig'].get('framework',
                   self.config['modelConfig'].get('modelType', 'tensorflow'))
        
        if framework == 'pytorch':
            import torch
            model = torch.load(model_path, weights_only=False)
            model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1).numpy()
            
            return probs
        
        else:  # tensorflow
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            probs = model.predict(X, verbose=0)
            return probs
    
    def predict_ensemble(self, X, return_individual=False):
        """
        Get ensemble predictions using quantile aggregation.
        
        Args:
            X: Input features (N x features array)
            return_individual: If True, also return individual model predictions
            
        Returns:
            If return_individual=False:
                ensemble_scores: Final ensemble scores (N,)
            If return_individual=True:
                (ensemble_scores, individual_predictions, individual_scores)
        """
        if len(self.models) == 0:
            raise ValueError("No models loaded. Call load_models() first.")
        
        print(f"\nMaking ensemble predictions with {len(self.models)} models...")
        print(f"Quantile: {self.quantile}")
        print(f"Method: {'Harrell-Davis' if self.use_harrell_davis else 'Simple quantile'}")
        
        # Get predictions from each model
        all_predictions = []
        all_scores = []  # Probability of seizure class
        
        for i, model_path in enumerate(self.models):
            print(f"  Model {i+1}/{len(self.models)}: {Path(model_path).parent.name}", end='')
            
            probs = self.predict_single_model(model_path, X)
            
            # Binary classification: probs[:,1] is probability of positive class (seizure)
            seizure_prob = probs[:, 1]
            
            all_predictions.append(probs)
            all_scores.append(seizure_prob)
            
            print(f" - mean score: {seizure_prob.mean():.3f}")
        
        # Stack all scores (models x samples)
        all_scores = np.array(all_scores)
        
        # Apply quantile aggregation across models for each sample
        ensemble_scores = np.zeros(all_scores.shape[1])
        
        quantile_func = harrell_davis_quantile if self.use_harrell_davis else simple_quantile
        
        for i in range(all_scores.shape[1]):
            sample_scores = all_scores[:, i]
            ensemble_scores[i] = quantile_func(sample_scores, self.quantile)
        
        print(f"\nEnsemble score range: [{ensemble_scores.min():.3f}, {ensemble_scores.max():.3f}]")
        print(f"Ensemble score mean: {ensemble_scores.mean():.3f}")
        
        if return_individual:
            return ensemble_scores, all_predictions, all_scores
        else:
            return ensemble_scores
    
    def evaluate(self, test_csv_path, threshold=0.5):
        """
        Evaluate ensemble on test data.
        
        Args:
            test_csv_path: Path to test CSV file with features
            threshold: Decision threshold (default 0.5)
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*80)
        print("ENSEMBLE EVALUATION")
        print("="*80)
        
        # Load test data
        print(f"\nLoading test data from: {test_csv_path}")
        df = pd.read_csv(test_csv_path)
        
        # Extract features and labels
        # Assuming 'type' column has labels and other columns are features
        if 'type' in df.columns:
            y_true = df['type'].values
            # Drop non-feature columns
            feature_cols = [col for col in df.columns if col not in 
                          ['eventId', 'userId', 'typeStr', 'type', 'dataTime', 
                           'osdAlarmState', 'osdSpecPower', 'osdRoiPower', 'hr', 'o2sat']]
            X = df[feature_cols].values
        else:
            print("ERROR: 'type' column not found in test data")
            return None
        
        print(f"Test samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        print(f"Positive samples: {y_true.sum()}")
        print(f"Negative samples: {len(y_true) - y_true.sum()}")
        
        # Get ensemble predictions
        ensemble_scores = self.predict_ensemble(X)
        y_pred = (ensemble_scores >= threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import confusion_matrix, classification_report
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        results = {
            'quantile': self.quantile,
            'threshold': threshold,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'n_models': len(self.models)
        }
        
        print(f"\n{'-'*80}")
        print(f"RESULTS (quantile={self.quantile}, threshold={threshold})")
        print(f"{'-'*80}")
        print(f"TP: {tp:5d}   FP: {fp:5d}")
        print(f"FN: {fn:5d}   TN: {tn:5d}")
        print(f"")
        print(f"Sensitivity:  {sensitivity:.1%}")
        print(f"Specificity:  {specificity:.1%}")
        print(f"Precision:    {precision:.1%}")
        print(f"F1 Score:     {f1:.1%}")
        print(f"{'-'*80}")
        
        return results
    
    def sweep_quantiles(self, test_csv_path, quantiles=None, threshold=0.5):
        """
        Evaluate ensemble across multiple quantile values.
        
        Args:
            test_csv_path: Path to test CSV
            quantiles: List of quantile values to test (default: 0.1 to 0.9 in steps of 0.1)
            threshold: Decision threshold
            
        Returns:
            List of result dictionaries
        """
        if quantiles is None:
            quantiles = np.arange(0.1, 1.0, 0.1)
        
        print("\n" + "="*80)
        print("QUANTILE SWEEP")
        print("="*80)
        
        all_results = []
        
        for q in quantiles:
            self.quantile = q
            results = self.evaluate(test_csv_path, threshold)
            if results:
                all_results.append(results)
        
        # Print summary table
        print("\n" + "="*80)
        print("QUANTILE SWEEP SUMMARY")
        print("="*80)
        print(f"\n{'Quantile':>10} {'Sensitivity':>12} {'Specificity':>12} {'Precision':>12} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
        print("-"*80)
        
        for r in all_results:
            print(f"{r['quantile']:10.2f} {r['sensitivity']:11.1%} {r['specificity']:11.1%} "
                  f"{r['precision']:11.1%} {r['f1_score']:7.1%} "
                  f"{r['tp']:5d} {r['fp']:5d} {r['fn']:5d}")
        
        return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Ensemble predictor for k-fold trained models'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Configuration file path'
    )
    parser.add_argument(
        '--outputDir',
        required=True,
        help='Base output directory containing fold subdirectories'
    )
    parser.add_argument(
        '--quantile',
        type=float,
        default=0.6,
        help='Quantile value for aggregation (0-1, default: 0.6)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Decision threshold (default: 0.5)'
    )
    parser.add_argument(
        '--testData',
        help='Path to test CSV file. If not provided, uses test data from fold0'
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Sweep through multiple quantile values'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple numpy quantile instead of Harrell-Davis'
    )
    
    args = parser.parse_args()
    
    # Create ensemble predictor
    predictor = EnsemblePredictor(
        args.config,
        args.outputDir,
        quantile=args.quantile,
        use_harrell_davis=not args.simple
    )
    
    # Load models
    if not predictor.load_models():
        print("ERROR: Failed to load models")
        sys.exit(1)
    
    # Determine test data path
    if args.testData:
        test_csv_path = args.testData
    else:
        # Use test data from fold0 by default
        config = predictor.config
        test_fname = config['dataFileNames']['testFeaturesHistoryFileCsv']
        test_csv_path = Path(args.outputDir) / "fold0" / test_fname
    
    if not Path(test_csv_path).exists():
        print(f"ERROR: Test data not found: {test_csv_path}")
        sys.exit(1)
    
    # Evaluate
    if args.sweep:
        results = predictor.sweep_quantiles(test_csv_path, threshold=args.threshold)
        
        # Save results
        output_file = Path(args.outputDir) / "ensemble_quantile_sweep.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    else:
        results = predictor.evaluate(test_csv_path, threshold=args.threshold)
        
        # Save results
        output_file = Path(args.outputDir) / f"ensemble_q{args.quantile:.2f}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
