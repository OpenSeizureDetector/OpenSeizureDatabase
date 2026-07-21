#!/usr/bin/env python3
"""
Quick integration guide for subtype weighting in nnTrainer.py

This file shows exactly where and how to add subtype weighting support
to the existing PyTorch training pipeline.
"""

# ============================================================================
# STEP 1: Add these imports at the top of nnTrainer.py
# ============================================================================

# Add this line to the existing imports:
try:
    from subtype_weighting import create_subtype_weighted_sampler
except ImportError:
    create_subtype_weighted_sampler = None
    print("WARNING: subtype_weighting module not found - subtype weighting unavailable")


# ============================================================================
# STEP 2: Add configuration parameters in load_config_params()
# ============================================================================

def load_config_params(configObj):
    """Extract all training configuration parameters from configObj."""
    # ... existing code ...
    
    # Add these lines in the "Data processing" section:
    params['use_subtype_weighting'] = libosd.configUtils.getConfigParam(
        "useSubtypeWeighting", configObj['modelConfig']
    )
    if params['use_subtype_weighting'] is None:
        params['use_subtype_weighting'] = False
    
    params['subtype_weights'] = libosd.configUtils.getConfigParam(
        "subtypeWeights", configObj['modelConfig']
    )
    if params['subtype_weights'] is None:
        params['subtype_weights'] = {}
    
    if params['use_subtype_weighting'] and params['subtype_weights']:
        print(f"Subtype weighting enabled with weights: {params['subtype_weights']}")
    
    # ... rest of existing code ...


# ============================================================================
# STEP 3: Modify the training data loading to keep the dataframe
# ============================================================================

def train_pytorch_model(configObj, nnModel, params, TAG):
    """Train PyTorch model with optional subtype weighting."""
    
    # ... existing data loading code ...
    
    # After loading training CSV, keep the dataframe for subtype weighting
    print(f"{TAG}: Loading training data from file {trainAugCsvFnamePath}")
    df_train = augmentData.loadCsv(trainAugCsvFnamePath, debug=debug)
    print(f"{TAG}: Loaded {len(df_train)} training datapoints")
    
    # ... existing preprocessing ...
    # xTrain, yTrain = ... (existing code)
    
    # Now, when creating the data loader, use subtype weighting if enabled:
    # ========================================================================
    # REPLACE THIS SECTION:
    # ========================================================================
    
    # OLD CODE (before):
    """
    train_dataset = TensorDataset(xTrain_tensor, yTrain_tensor)
    val_dataset = TensorDataset(xVal_tensor, yVal_tensor)
    
    use_balanced_batches = params.get('use_balanced_batches', False)
    if use_balanced_batches and params['use_lr_schedule']:
        print(f"{TAG}: Using balanced batch sampling")
        class_counts = torch.bincount(yTrain_tensor)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[yTrain_tensor]
        
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)
    """
    
    # NEW CODE (with subtype weighting):
    # ========================================================================
    train_dataset = TensorDataset(xTrain_tensor, yTrain_tensor)
    val_dataset = TensorDataset(xVal_tensor, yVal_tensor)
    
    use_balanced_batches = params.get('use_balanced_batches', False)
    use_subtype_weighting = params.get('use_subtype_weighting', False) and len(params.get('subtype_weights', {})) > 0
    
    if use_subtype_weighting and create_subtype_weighted_sampler is not None:
        # Use subtype-aware weighting
        print(f"{TAG}: Using subtype-aware sample weighting")
        sampler = create_subtype_weighted_sampler(
            df=df_train,
            y_values=yTrain,
            subtype_weights=params['subtype_weights'],
            debug=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            sampler=sampler,
            drop_last=True
        )
        print(f"{TAG}: Subtype weights applied: {params['subtype_weights']}")
    
    elif use_balanced_batches and params['use_lr_schedule']:
        # Use class-based balanced sampling (existing code)
        print(f"{TAG}: Using balanced batch sampling (class-based)")
        class_counts = torch.bincount(yTrain_tensor)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[yTrain_tensor]
        
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            sampler=sampler,
            drop_last=True
        )
        print(f"{TAG}: Class distribution in training data: {class_counts.tolist()}")
    
    else:
        # Standard DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            drop_last=True
        )
        print(f"{TAG}: Using standard DataLoader (no sampling)")
    
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # ... rest of existing training code ...


# ============================================================================
# STEP 4: Example configuration file (osdbCfg.json)
# ============================================================================

"""
{
  "modelConfig": {
    "framework": "pytorch",
    "modelClass": "deepEpiCnnModel_torch.DeepEpiCnnModelPyTorch",
    
    "useSubtypeWeighting": true,
    "subtypeWeights": {
      "Tonic-Clonic": 2.0,
      "Focal": 1.0,
      "Generalized": 1.0,
      "Unknown": 1.0
    },
    
    "useBalancedBatches": false,
    "useLrSchedule": true,
    
    ... other config ...
  }
}
"""


# ============================================================================
# STEP 5: Testing the integration
# ============================================================================

def test_subtype_weighting():
    """Test that subtype weighting works correctly."""
    import pandas as pd
    import numpy as np
    
    # Create mock training data
    df = pd.DataFrame({
        'eventId': ['1', '2', '3', '4', '5', '6'],
        'subType': ['Tonic-Clonic', 'Tonic-Clonic', 'Focal', 'Focal', 'Focal', 'Other'],
        'type': [1, 1, 1, 1, 1, 0]
    })
    
    # Mock labels (class 1 for seizure, 0 for non-seizure)
    yTrain = np.array([1, 1, 1, 1, 1, 0])
    
    # Test with subtype weights
    subtype_weights = {
        'Tonic-Clonic': 2.0,
        'Focal': 1.0,
        'Other': 1.0
    }
    
    if create_subtype_weighted_sampler is not None:
        sampler = create_subtype_weighted_sampler(
            df=df,
            y_values=yTrain,
            subtype_weights=subtype_weights,
            debug=True
        )
        print("✓ Subtype weighting works!")
        return True
    else:
        print("✗ Subtype weighting module not available")
        return False


if __name__ == '__main__':
    test_subtype_weighting()
