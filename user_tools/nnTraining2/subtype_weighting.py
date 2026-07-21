#!/usr/bin/env python3
"""
Subtype-aware sample weighting for PyTorch training.

This module provides utilities to apply differential weights to training samples
based on seizure subtype (e.g., giving higher weight to tonic-clonic seizures).

This approach works within PyTorch's DataLoader using WeightedRandomSampler,
allowing flexible adjustment of weights without modifying the dataset.

Usage:
    # In training configuration
    "subtypeWeights": {
        "Tonic-Clonic": 2.0,      # 2x weight for tonic-clonic seizures
        "Other": 1.0               # 1x weight for other seizure types
    },
    "useSubtypeWeighting": True

    # In training code
    from subtype_weighting import create_subtype_weighted_sampler
    
    sampler = create_subtype_weighted_sampler(
        df=train_df,
        y_values=yTrain,
        subtype_weights=params['subtypeWeights'],
        debug=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
import pandas as pd


def normalize_subtype(subtype_str):
    """
    Normalize seizure subtype strings to a canonical form.
    
    Args:
        subtype_str: Raw subtype string from data
        
    Returns:
        Normalized subtype string
        
    Examples:
        'Tonic-Clonic' -> 'Tonic-Clonic'
        'tonic-clonic' -> 'Tonic-Clonic'
        'TONIC-CLONIC' -> 'Tonic-Clonic'
        'tonic clonic' -> 'Tonic-Clonic'
        'Focal' -> 'Focal'
        None -> 'Unknown'
    """
    if subtype_str is None or not isinstance(subtype_str, str):
        return 'Unknown'
    
    subtype_lower = subtype_str.lower().strip()
    
    # Handle tonic-clonic variants
    if 'tonic' in subtype_lower and 'clonic' in subtype_lower:
        return 'Tonic-Clonic'
    
    # Map other common subtypes
    subtype_mapping = {
        'focal': 'Focal',
        'generalized': 'Generalized',
        'absence': 'Absence',
        'atonic': 'Atonic',
        'myoclonic': 'Myoclonic',
        'unknown': 'Unknown'
    }
    
    for key, canonical in subtype_mapping.items():
        if key in subtype_lower:
            return canonical
    
    # Return normalized original if no match
    return ' '.join(word.capitalize() for word in subtype_lower.split('-'))


def create_subtype_weighted_sampler(df, y_values, subtype_weights=None, debug=False):
    """
    Create a WeightedRandomSampler that weights samples based on seizure subtype.
    
    This sampler oversamples minority classes (e.g., tonic-clonic seizures) to give
    them higher weight during training without modifying the dataset.
    
    Args:
        df: Pandas DataFrame with 'eventId' and 'subType' columns
        y_values: Array/list of class labels (1=seizure, 0=non-seizure)
        subtype_weights: Dict mapping normalized subtype names to weight multipliers
                        Example: {'Tonic-Clonic': 2.0, 'Other': 1.0}
                        If None, uses balanced class weights
        debug: Print debug information
        
    Returns:
        WeightedRandomSampler configured for the training data
        
    Raises:
        ValueError: If required columns missing from dataframe
        
    Example:
        >>> subtype_weights = {
        ...     'Tonic-Clonic': 2.0,  # 2x weight
        ...     'Focal': 1.5,
        ...     'Other': 1.0
        ... }
        >>> sampler = create_subtype_weighted_sampler(
        ...     df=train_df,
        ...     y_values=yTrain,
        ...     subtype_weights=subtype_weights
        ... )
        >>> train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if 'eventId' not in df.columns:
        raise ValueError("DataFrame must contain 'eventId' column")
    
    if 'subType' not in df.columns:
        print("WARNING: 'subType' column not found in DataFrame")
        print(f"Available columns: {list(df.columns[:20])}")
        print("Falling back to class-based weighting")
        subtype_weights = None
    
    y_values = np.array(y_values)
    
    if len(df) != len(y_values):
        raise ValueError(f"DataFrame length ({len(df)}) must match y_values length ({len(y_values)})")
    
    # Default to class-balanced weighting if no subtype weights provided
    if subtype_weights is None:
        class_counts = np.bincount(y_values.astype(int))
        class_weights = 1.0 / class_counts[class_counts > 0].astype(float)
        # Map class weights to all samples
        sample_weights = np.zeros(len(y_values))
        for class_idx, weight in enumerate(class_weights):
            sample_weights[y_values == class_idx] = weight
        
        if debug:
            print("Using class-based weighting (no subtype weights provided)")
            print(f"Class distribution: {dict(zip(*np.unique(y_values, return_counts=True)))}")
    else:
        # Initialize sample weights based on class (seizure vs non-seizure)
        class_counts = np.bincount(y_values.astype(int))
        sample_weights = np.zeros(len(y_values))
        
        # First, apply class-level weighting
        class_weights = 1.0 / class_counts.astype(float)
        for class_idx, weight in enumerate(class_weights):
            sample_weights[y_values == class_idx] = weight
        
        # Then, apply subtype-specific multipliers for seizure events (class=1)
        seizure_mask = y_values == 1
        if seizure_mask.any():
            subtype_counts = {}
            subtype_multipliers = np.ones(len(y_values))
            
            # Determine subtype for each sample
            df_copy = df.copy()
            df_copy['sample_idx'] = range(len(df_copy))
            
            # Group by eventId to get unique subtype per event
            event_subtypes = {}
            for event_id, group in df_copy.groupby('eventId', sort=False):
                subtype = group.iloc[0]['subType']
                norm_subtype = normalize_subtype(subtype)
                event_subtypes[event_id] = norm_subtype
                
                if norm_subtype not in subtype_counts:
                    subtype_counts[norm_subtype] = 0
                subtype_counts[norm_subtype] += 1
            
            # Apply subtype-specific weights to seizure samples
            for idx in range(len(df)):
                if seizure_mask[idx]:
                    event_id = df.iloc[idx]['eventId']
                    norm_subtype = event_subtypes.get(event_id, 'Unknown')
                    
                    # Get multiplier for this subtype (default to 1.0)
                    multiplier = subtype_weights.get(norm_subtype, 1.0)
                    subtype_multipliers[idx] = multiplier
            
            # Apply subtype multipliers to class weights
            sample_weights = sample_weights * subtype_multipliers
            
            if debug:
                print(f"\nSubtype Weighting Configuration:")
                print(f"  Subtype weights: {subtype_weights}")
                print(f"\nSubtype Distribution in Seizures:")
                for subtype, count in sorted(subtype_counts.items()):
                    multiplier = subtype_weights.get(subtype, 1.0)
                    print(f"  {subtype}: {count} events, multiplier={multiplier}")
                
                print(f"\nFinal Sample Weights (statistics for seizure samples):")
                seizure_weights = sample_weights[seizure_mask]
                print(f"  Min: {seizure_weights.min():.4f}")
                print(f"  Max: {seizure_weights.max():.4f}")
                print(f"  Mean: {seizure_weights.mean():.4f}")
    
    # Normalize weights to [0, 1] range for WeightedRandomSampler
    if sample_weights.max() > 0:
        sample_weights = sample_weights / sample_weights.max()
    
    if debug:
        print(f"\nSampler Configuration:")
        print(f"  Total samples: {len(sample_weights)}")
        print(f"  Weight range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def get_default_subtype_weights():
    """
    Return default subtype weights for common seizure types.
    
    These are reasonable starting points; adjust based on your data.
    
    Returns:
        Dict mapping normalized subtype names to weight multipliers
    """
    return {
        'Tonic-Clonic': 2.0,    # Double weight for tonic-clonic
        'Focal': 1.0,
        'Generalized': 1.0,
        'Absence': 1.0,
        'Atonic': 1.0,
        'Myoclonic': 1.0,
        'Unknown': 1.0
    }


if __name__ == '__main__':
    # Simple test
    import sys
    
    # Create mock data
    df = pd.DataFrame({
        'eventId': ['1', '1', '2', '2', '3', '3', '4', '4'],
        'subType': ['Tonic-Clonic', 'Tonic-Clonic', 'Tonic-Clonic', 'Tonic-Clonic', 
                   'Focal', 'Focal', 'Focal', 'Focal'],
        'type': [1, 1, 1, 1, 1, 1, 0, 0]
    })
    
    y_values = [1, 1, 1, 1, 1, 1, 0, 0]
    
    # Test with no subtype weights
    print("Test 1: Class-based weighting only")
    sampler = create_subtype_weighted_sampler(df, y_values, debug=True)
    print()
    
    # Test with subtype weights
    print("Test 2: With subtype weighting (2x for Tonic-Clonic)")
    subtype_weights = {
        'Tonic-Clonic': 2.0,
        'Focal': 1.0,
        'Unknown': 1.0
    }
    sampler = create_subtype_weighted_sampler(df, y_values, subtype_weights, debug=True)
    print()
    
    print("SUCCESS: Subtype weighting module loaded correctly")
