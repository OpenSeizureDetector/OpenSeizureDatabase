#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import csv
import json
import importlib
#from tkinter import Y
import sklearn.metrics
try:
    import imblearn
except ImportError as e:
    # imblearn is only required for (over/under) sampling utilities during training.
    # Keep inference/testing tools importable even if imblearn/sklearn versions are incompatible.
    imblearn = None
    _imblearn_import_error = e
import numpy as np
import matplotlib.pyplot as plt
import gc

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.osdAlgTools
import libosd.configUtils

try:
    from user_tools.nnTraining2 import augmentData
except ImportError:
    import augmentData
try:
    from user_tools.nnTraining2.subtype_weighting import create_subtype_weighted_sampler
except ImportError:
    try:
        from subtype_weighting import create_subtype_weighted_sampler
    except ImportError:
        create_subtype_weighted_sampler = None
import nnTester

# ---------------------------------------------------------------------------
# Centralized random seed management
# Single place to control determinism for the entire training pipeline.
# Uses config["randomSeed"] (top-level) if set to an int.
# If null/None/missing -> non-deterministic (true random) sampling.
# To change behaviour, update only this section or the config file.
# ---------------------------------------------------------------------------
RANDOM_SEED_CONFIG_KEY = "randomSeed"


def get_seed_from_config(configObj):
    """Return seed int or None. None means non-deterministic."""
    if not isinstance(configObj, dict):
        return None
    if RANDOM_SEED_CONFIG_KEY not in configObj:
        return None
    seed = configObj.get(RANDOM_SEED_CONFIG_KEY)
    if seed is None:
        return None
    # JSON may store as string
    try:
        # Explicit null in JSON -> None already handled
        seed_int = int(seed)
        return seed_int
    except Exception:
        return None


def seed_all(seed, debug=False):
    """Seed all RNGs for deterministic training. If seed is None, leave random.

    Seeds: python random, numpy, torch (cpu+cuda), PYTHONHASHSEED,
           cudnn deterministic flags, torch deterministic algorithms.
    Call once at start of runSequence and at start of trainModel_pytorch.
    """
    if seed is None:
        if debug:
            print(f"seed_all: randomSeed is null/None -> non-deterministic (random) sampling")
        return None
    import random as _random
    # PYTHONHASHSEED must be set before interpreter start to fully affect hashing,
    # but setting env var still helps for child processes.
    os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # cuDNN determinism
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        # PyTorch >=1.8 deterministic algorithms (may warn on some ops)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            if debug:
                print(f"seed_all: torch.use_deterministic_algorithms not enabled: {e}")
        # For DataLoader workers (if num_workers>0)
        try:
            torch.manual_seed(seed)
        except Exception:
            pass
    except ImportError:
        pass
    except Exception as e:
        if debug:
            print(f"seed_all: torch seeding failed: {e}")
    # TensorFlow if present
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    except Exception:
        pass
    if debug:
        print(f"seed_all: seeded all RNGs with {seed}")
    return seed


def make_torch_generator(seed, debug=False):
    """Return torch.Generator seeded with seed, or None if seed is None/non-deterministic."""
    if seed is None:
        return None
    try:
        import torch
        g = torch.Generator()
        g.manual_seed(int(seed))
        return g
    except Exception as e:
        if debug:
            print(f"make_torch_generator: failed: {e}")
        return None


def get_framework_from_config(configObj):
    """
    Determine which framework to use from config.
    
    Returns: 'tensorflow' or 'pytorch'
    """
    if 'modelConfig' in configObj and 'framework' in configObj['modelConfig']:
        framework = configObj['modelConfig']['framework'].lower()
        if framework in ['pytorch', 'torch']:
            return 'pytorch'
        elif framework in ['tensorflow', 'tf', 'keras']:
            return 'tensorflow'
    
    # Check legacy 'modelType' field
    if 'modelConfig' in configObj and 'modelType' in configObj['modelConfig']:
        model_type = configObj['modelConfig']['modelType'].lower()
        if model_type in ['pytorch', 'torch']:
            return 'pytorch'
        elif model_type in ['tensorflow', 'tf', 'keras']:
            return 'tensorflow'
    
    # Default to tensorflow for backward compatibility
    return 'tensorflow'


def _log_mem(phase, extra=""):
    """Phase 0 helper: always-on memory logging (no hard dep on psutil)."""
    try:
        import psutil
        p = psutil.Process()
        rss = p.memory_info().rss / 1e9
        vms = p.memory_info().vms / 1e9
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        avail = vm.available / 1e9
        gpu_str = ""
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                gpu_str = f" gpu_free={free/1e9:.1f}GB/{total/1e9:.1f}GB"
        except Exception:
            pass
        msg = f"[MEM] {phase:30s} rss={rss:.2f}GB vms={vms:.2f}GB avail={avail:.2f}GB swap_used={swap.used/1e9:.2f}/{swap.total/1e9:.2f}GB{gpu_str}"
        if extra:
            msg += f" {extra}"
        print(msg, flush=True)
    except ImportError:
        pass
    except Exception as e:
        print(f"[MEM] {phase} error: {e}", flush=True)


def _build_optimized_read_params(csvPath, nnModel, use_float32=True):
    """
    Phase 1 helper: build dtype for augmentData.loadCsv to enable
    float32 early.

    Returns dict with keys dtype (or None) and usecols (or None).
    Minimal version for Phase 1: only dtype (float32 for accel, int32
    for type) — usecols is left as None to avoid header-read brittleness.
    dtype halves DataFrame RAM vs float64; usecols optimisation can be
    added later once proven stable.
    """
    if not use_float32 or csvPath is None or not os.path.exists(csvPath):
        return {"dtype": None, "usecols": None}
    try:
        accel_mode = str(getattr(nnModel, 'accel_input_mode', 'magnitude')).lower()
    except Exception:
        accel_mode = 'magnitude'
    use_xyz = accel_mode == 'xyz'
    dtype = {}
    for i in range(125):
        dtype[f"M{i:03d}_t-0"] = 'float32'
        dtype[f"M{i:03d}"] = 'float32'
        if use_xyz:
            for p in ('X', 'Y', 'Z'):
                dtype[f"{p}{i:03d}_t-0"] = 'float32'
                dtype[f"{p}{i:03d}"] = 'float32'
    dtype['hr'] = 'float32'
    dtype['type'] = 'int32'
    # No header read, no usecols — pandas will ignore dtype keys for
    # columns not present in the file (safe).
    return {"dtype": dtype, "usecols": None}


def ensure_subtype_column(df):
    """Return df guaranteed to carry a 'subType' column for subtype weighting.

    The flattened / feature CSVs have no dedicated subType column: flattenData.py
    writes the subtype inside typeStr as 'Type/SubType' (e.g. 'Seizure/Tonic-Clonic').
    Derive it from that column (or from an 'eventType' column if present).

    Returns:
        df unchanged if it already has subType, a copy with subType derived, or
        None if no source column is available.
    """
    if 'subType' in df.columns:
        return df
    for srcCol in ('eventType', 'typeStr'):
        if srcCol in df.columns:
            out = df.copy()
            out['subType'] = (out[srcCol].astype(str)
                              .str.split('/', n=1).str[-1]
                              .str.strip()
                              .str.strip('"').str.strip("'"))
            return out
    return None


def df2trainingData(df, nnModel, debug=False, return_row_indices=False):
    ''' Converts a pandas dataframe df into a list of data and a list of associated seizure classes
    for use by model nnModel.
    This works by taking each row in the dataframe and converting it into a dict with the same
    values as an OpenSeizureDetector datapoint.   It then calls the dp2vector method of the specified
    model to pre-process the data into the format required by the model.

    FIXME:  It uses a simple for loop to loop through the dataframe - there is probably a quicker
    way of applying a function to each row in the dataframe in turn.

    Supports both magnitude-only and 3D xyz input modes.
    '''

    cols = list(df.columns)

    def _collect_axis_cols(prefix):
        with_suffix = [
            c for c in cols
            if isinstance(c, str) and c.startswith(prefix) and c.endswith('_t-0') and c[len(prefix):-4].isdigit()
        ]
        if with_suffix:
            return sorted(with_suffix, key=lambda c: int(c[len(prefix):-4]))
        no_suffix = [
            c for c in cols
            if isinstance(c, str) and c.startswith(prefix) and len(c) == 4 and c[1:].isdigit()
        ]
        return sorted(no_suffix, key=lambda c: int(c[1:]))

    accel_input_mode = str(getattr(nnModel, 'accel_input_mode', 'magnitude')).lower()
    use_xyz = accel_input_mode == 'xyz'

    m_cols = _collect_axis_cols('M')
    x_cols = _collect_axis_cols('X')
    y_cols = _collect_axis_cols('Y')
    z_cols = _collect_axis_cols('Z')

    if use_xyz:
        if len(x_cols) == 0 or len(y_cols) == 0 or len(z_cols) == 0:
            print("cols are: ", [c for c in cols])
            raise ValueError("df2trainingData: XYZ mode requested but X/Y/Z columns not found")
    else:
        if len(m_cols) == 0:
            print("cols are: ", [c for c in cols])
            raise ValueError("df2trainingData: No magnitude (Mxxx_t-0 or Mxxx) columns found in dataframe")
        if len(x_cols) != len(y_cols) or len(x_cols) != len(z_cols):
            # Only validate xyz counts when xyz mode
            pass
        if len(x_cols) == len(y_cols) == len(z_cols) and len(x_cols) != 0 and not use_xyz:
            pass

    # Phase 2: vectorised extraction — avoid per-row Series (df.iloc[n]) which
    # copies 135 cols per row and re-inflates float32 to float64 via astype(float).
    # Use to_numpy once (flexible window via model.bufferSamples, not hardcoded 750).
    N = len(df)
    # Pre-extract essential columns as numpy arrays
    event_ids = df['eventId'].to_numpy(dtype=object)
    types_arr = df['type'].to_numpy()
    hr_arr = None
    try:
        hr_arr = df['hr'].to_numpy(dtype=np.float32)
        # hr may contain NaN for float32; keep as is for later int conversion
    except Exception:
        hr_arr = None

    if use_xyz:
        # Extract accel matrices as float32, then nan_to_num vectorised (avoids per-row tolist)
        # Using to_numpy with dtype float32 will copy but is 4x smaller than float64 list
        try:
            x_data = df[x_cols].to_numpy(dtype=np.float32)
            y_data = df[y_cols].to_numpy(dtype=np.float32)
            z_data = df[z_cols].to_numpy(dtype=np.float32)
        except Exception:
            # Fallback: let pandas infer then cast
            x_data = df[x_cols].to_numpy(dtype=np.float32)
            y_data = df[y_cols].to_numpy(dtype=np.float32)
            z_data = df[z_cols].to_numpy(dtype=np.float32)
        x_data = np.nan_to_num(x_data, nan=0.0, posinf=0.0, neginf=0.0)
        y_data = np.nan_to_num(y_data, nan=0.0, posinf=0.0, neginf=0.0)
        z_data = np.nan_to_num(z_data, nan=0.0, posinf=0.0, neginf=0.0)
        mag_data = None
    else:
        # Magnitude: single matrix (N, 125)
        try:
            mag_data = df[m_cols].to_numpy(dtype=np.float32)
        except Exception:
            mag_data = df[m_cols].to_numpy(dtype=np.float32)
        mag_data = np.nan_to_num(mag_data, nan=0.0, posinf=0.0, neginf=0.0)
        x_data = y_data = z_data = None

    outLst = []
    classLst = []
    usedRowIdxLst = []
    lastEventId = None
    print("Processing Events:")
    # Reuse single dict to reduce allocation (still need per-row rawData)
    for n in range(N):
        eventId = event_ids[n]
        if eventId != lastEventId:
            sys.stdout.write("%d/%d (%.1f %%) : %s\r" % (n, N, 100.*n/N, eventId))
            nnModel.resetAccBuf()
            lastEventId = eventId

        dpDict = {}
        if use_xyz:
            # Interleave xyz per sample: raw3d length 375 (125*3)
            # Use slice from pre-extracted matrices (already float32, nan->0)
            # Create 1D interleaved view without Python loops for raw3d extend
            xv = x_data[n]
            yv = y_data[n]
            zv = z_data[n]
            # Interleave as [x0,y0,z0, x1,y1,z1, ...] via stacking
            # Use empty and strided assignment for speed (avoids tolist)
            raw3d = np.empty(375, dtype=np.float32)
            raw3d[0::3] = xv
            raw3d[1::3] = yv
            raw3d[2::3] = zv
            dpDict['rawData3D'] = raw3d
        else:
            accArr = mag_data[n]
            if debug:
                print("accArr=", accArr, type(accArr))
            dpDict['rawData'] = accArr
        if hr_arr is not None:
            try:
                # hr_arr is float32, may be nan
                hv = hr_arr[n]
                if np.isnan(hv):
                    dpDict['hr'] = None
                else:
                    dpDict['hr'] = int(hv)
            except Exception:
                dpDict['hr'] = None
        else:
            dpDict['hr'] = None
        if debug:
            print("dpDict=", dpDict)
        dpInputData = nnModel.dp2vector(dpDict, normalise=False)
        if dpInputData is not None:
            outLst.append(dpInputData)
            classLst.append(types_arr[n])
            usedRowIdxLst.append(n)
    # Free large matrices early before np conversion (helps stay <64GB for N=1.6M)
    del mag_data, x_data, y_data, z_data, event_ids, types_arr, hr_arr
    # gc not needed here immediately; caller will del df and gc
    print(".")
    if return_row_indices:
        return(outLst, classLst, usedRowIdxLst)
    return(outLst, classLst)


def load_config_params(configObj):
    """Extract all training configuration parameters from configObj.
    
    Returns:
        dict: Dictionary containing all configuration parameters
    """
    params = {}
    
    # Data file names – respect whether feature history is actually used.
    # runSequence skips history when addFeatureHistoryLength==0 or only raw acc features;
    # nnTrainer must mirror that logic instead of always preferring the history file.
    addHistoryLength = configObj.get('dataProcessing', {}).get('addFeatureHistoryLength', 0)
    features = configObj.get('dataProcessing', {}).get('features', [])
    raw_acc_features = {
        'acc_magnitude',
        'acc_x', 'acc_y', 'acc_z',
        'accX', 'accY', 'accZ',
    }
    try:
        only_raw_acc = all(f in raw_acc_features for f in features) if features else False
    except Exception:
        only_raw_acc = False
    skip_history = (addHistoryLength == 0) or only_raw_acc

    if skip_history:
        params['trainAugCsvFname'] = libosd.configUtils.getConfigParam('trainFeaturesFileCsv', configObj['dataFileNames'])
        params['testCsvFname'] = libosd.configUtils.getConfigParam("testFeaturesFileCsv", configObj['dataFileNames'])
        # val will be resolved via features file in resolve_data_file_paths;
        # keep the explicit valDataFileCsv name here so the resolver can prefer valFeatures.csv
        params['valCsvFname'] = libosd.configUtils.getConfigParam('valDataFileCsv', configObj['dataFileNames'])
    else:
        params['trainAugCsvFname'] = libosd.configUtils.getConfigParam('trainFeaturesHistoryFileCsv', configObj['dataFileNames'])
        params['testCsvFname'] = libosd.configUtils.getConfigParam("testFeaturesHistoryFileCsv", configObj['dataFileNames'])
        params['valCsvFname'] = libosd.configUtils.getConfigParam('valDataFileCsv', configObj['dataFileNames'])
    if not isinstance(params['trainAugCsvFname'], str):
        params['trainAugCsvFname'] = None
    if not isinstance(params['valCsvFname'], str):
        params['valCsvFname'] = None
    if not isinstance(params['testCsvFname'], str):
        params['testCsvFname'] = None
    
    # Model configuration
    params['modelFnameRoot'] = libosd.configUtils.getConfigParam("modelFname", configObj['modelConfig'])
    params['epochs'] = libosd.configUtils.getConfigParam("epochs", configObj['modelConfig'])
    params['batch_size'] = libosd.configUtils.getConfigParam("batchSize", configObj['modelConfig'])
    params['nLayers'] = libosd.configUtils.getConfigParam("nLayers", configObj['modelConfig'])
    params['lrFactor'] = libosd.configUtils.getConfigParam("lrFactor", configObj['modelConfig'])
    params['lrPatience'] = libosd.configUtils.getConfigParam("lrPatience", configObj['modelConfig'])
    params['lrStart'] = libosd.configUtils.getConfigParam("lrStart", configObj['modelConfig'])
    params['lrMin'] = libosd.configUtils.getConfigParam("lrMin", configObj['modelConfig'])
    params['earlyStoppingPatience'] = libosd.configUtils.getConfigParam("earlyStoppingPatience", configObj['modelConfig'])
    params['trainingVerbosity'] = libosd.configUtils.getConfigParam("trainingVerbosity", configObj['modelConfig'])
    params['nnModelClassName'] = libosd.configUtils.getConfigParam("modelClass", configObj['modelConfig'])
    
    # Optional advanced LR schedule / AdamW settings
    params['use_lr_schedule'] = libosd.configUtils.getConfigParam("useLrSchedule", configObj['modelConfig'])
    if params['use_lr_schedule'] is None:
        params['use_lr_schedule'] = False
    params['lr_peak'] = libosd.configUtils.getConfigParam("lrPeak", configObj['modelConfig'])
    if params['lr_peak'] is None:
        params['lr_peak'] = 1e-3
        print(f"WARNING: lrPeak not found in config, using default: {params['lr_peak']}")
    else:
        print(f"INFO: Read lrPeak from config: {params['lr_peak']}")
    params['lr_main_end'] = libosd.configUtils.getConfigParam("lrMainEnd", configObj['modelConfig'])
    if params['lr_main_end'] is None:
        params['lr_main_end'] = 3e-5
        print(f"WARNING: lrMainEnd not found in config, using default: {params['lr_main_end']}")
    else:
        print(f"INFO: Read lrMainEnd from config: {params['lr_main_end']}")
    params['warmup_steps'] = libosd.configUtils.getConfigParam("warmupSteps", configObj['modelConfig'])
    if params['warmup_steps'] is None:
        params['warmup_steps'] = 2500
    params['main_steps'] = libosd.configUtils.getConfigParam("mainSteps", configObj['modelConfig'])
    if params['main_steps'] is None:
        params['main_steps'] = 45000
    params['cooldown_steps'] = libosd.configUtils.getConfigParam("cooldownSteps", configObj['modelConfig'])
    if params['cooldown_steps'] is None:
        params['cooldown_steps'] = 2500
    params['use_adamw'] = libosd.configUtils.getConfigParam("useAdamW", configObj['modelConfig'])
    if params['use_adamw'] is None:
        params['use_adamw'] = True
    params['adamw_beta1'] = libosd.configUtils.getConfigParam("adamwBeta1", configObj['modelConfig'])
    if params['adamw_beta1'] is None:
        params['adamw_beta1'] = 0.9
    params['adamw_beta2'] = libosd.configUtils.getConfigParam("adamwBeta2", configObj['modelConfig'])
    if params['adamw_beta2'] is None:
        params['adamw_beta2'] = 0.999
    params['weight_decay'] = libosd.configUtils.getConfigParam("weightDecay", configObj['modelConfig'])
    if params['weight_decay'] is None:
        params['weight_decay'] = 0.0
    params['total_training_steps'] = libosd.configUtils.getConfigParam("totalTrainingSteps", configObj['modelConfig'])
    if params['total_training_steps'] is None:
        params['total_training_steps'] = 50000
    params['eval_every_steps'] = libosd.configUtils.getConfigParam("evalEverySteps", configObj['modelConfig'])
    if params['eval_every_steps'] is None:
        params['eval_every_steps'] = 5000
    params['save_best_on_both_improvement'] = libosd.configUtils.getConfigParam("saveBestOnBothImprovement", configObj['modelConfig'])
    if params['save_best_on_both_improvement'] is None:
        params['save_best_on_both_improvement'] = True
    params['save_best_on_far_reduction'] = libosd.configUtils.getConfigParam("saveBestOnFarReduction", configObj['modelConfig'])
    if params['save_best_on_far_reduction'] is None:
        params['save_best_on_far_reduction'] = 0.10
    params['save_best_on_sensitivity_tolerance'] = libosd.configUtils.getConfigParam("saveBestOnSensitivityTolerance", configObj['modelConfig'])
    if params['save_best_on_sensitivity_tolerance'] is None:
        params['save_best_on_sensitivity_tolerance'] = 0.05
    params['use_balanced_batches'] = libosd.configUtils.getConfigParam("useBalancedBatches", configObj['modelConfig'])
    if params['use_balanced_batches'] is None:
        params['use_balanced_batches'] = False
    params['use_subtype_weighting'] = libosd.configUtils.getConfigParam("useSubtypeWeighting", configObj['modelConfig'])
    if params['use_subtype_weighting'] is None:
        params['use_subtype_weighting'] = False
    params['subtype_weights'] = libosd.configUtils.getConfigParam("subtypeWeights", configObj['modelConfig'])
    if not isinstance(params['subtype_weights'], dict):
        params['subtype_weights'] = {}
    params['save_best_min_sensitivity'] = libosd.configUtils.getConfigParam("saveBestMinSensitivity", configObj['modelConfig'])
    if params['save_best_min_sensitivity'] is None:
        params['save_best_min_sensitivity'] = 0.25
    params['save_best_max_fpr'] = libosd.configUtils.getConfigParam("saveBestMaxFpr", configObj['modelConfig'])
    if params['save_best_max_fpr'] is None:
        params['save_best_max_fpr'] = None  # No limit by default
    params['model_selection_metric'] = libosd.configUtils.getConfigParam("modelSelectionMetric", configObj['modelConfig'])
    if params['model_selection_metric'] is None:
        params['model_selection_metric'] = 'dual_improvement'  # Options: 'dual_improvement', 'f1', 'f_beta', 'youden'
    params['f_beta'] = libosd.configUtils.getConfigParam("fBeta", configObj['modelConfig'])
    if params['f_beta'] is None:
        params['f_beta'] = 2.0  # Default favors recall (sensitivity) over precision
    params['pos_weight_multiplier'] = libosd.configUtils.getConfigParam("posWeightMultiplier", configObj['modelConfig'])
    if params['pos_weight_multiplier'] is None:
        params['pos_weight_multiplier'] = None  # No class weighting by default
    
    # Data processing
    params['validationProp'] = libosd.configUtils.getConfigParam("validationProp", configObj['dataProcessing'])
    params['inputDims'] = libosd.configUtils.getConfigParam("dims", configObj)
    if params['inputDims'] is None:
        params['inputDims'] = 1
    
    # Handle validation data fallback only when validation is explicitly disabled.
    if params['validationProp'] == 0:
        print("WARNING: validationProp set to 0 - no validation data used - using test data instead")
        params['valCsvFname'] = params['testCsvFname']
    
    return params


def load_model_class(nnModelClassName, configObj, framework='tensorflow'):
    """Load and instantiate the model class.
    
    Args:
        nnModelClassName: Full module path and class name
        configObj: Configuration object
        framework: 'tensorflow' or 'pytorch'
    
    Returns:
        Instantiated model object
    """
    # Handle auto-conversion for PyTorch
    if framework == 'pytorch' and 'deepEpiCnnModel.DeepEpiCnnModel' in nnModelClassName and 'torch' not in nnModelClassName.lower():
        nnModelClassName = nnModelClassName.replace('deepEpiCnnModel.DeepEpiCnnModel', 
                                                     'deepEpiCnnModel_torch.DeepEpiCnnModelPyTorch')
        print(f"Auto-converting model class to PyTorch: {nnModelClassName}")
    
    parts = nnModelClassName.split('.')
    if len(parts) < 2:
        raise ValueError("modelClass must be a module path and class name, e.g. 'mod.submod.ClassName'")
    nnModuleId = '.'.join(parts[:-1])
    nnClassId = parts[-1]

    print(f"Importing nn Module {nnModuleId}")
    nnModule = importlib.import_module(nnModuleId)
    # instantiate the class from the module
    nnModel = getattr(nnModule, nnClassId)(configObj['modelConfig'])
    
    return nnModel


def resolve_data_file_paths(dataDir, trainAugCsvFname, valCsvFname, configObj, TAG):
    """Resolve paths to training and validation data files, preferring feature CSVs.
    
    Args:
        dataDir: Base data directory
        trainAugCsvFname: Training data CSV filename (can be None)
        valCsvFname: Validation data CSV filename (can be None)
        configObj: Configuration object
        TAG: Tag for logging
    
    Returns:
        tuple: (trainAugCsvFnamePath, valCsvFnamePath)
    """
    trainAugCsvFnamePath = None
    valCsvFnamePath = None
    
    # Debug output
    print(f"{TAG}: resolve_data_file_paths called with:")
    print(f"{TAG}:   trainAugCsvFname = {trainAugCsvFname!r} (type: {type(trainAugCsvFname).__name__})")
    print(f"{TAG}:   valCsvFname = {valCsvFname!r} (type: {type(valCsvFname).__name__})")
    print(f"{TAG}:   dataDir = {dataDir!r}")
    
    # Build initial paths if filenames are provided
    if trainAugCsvFname and isinstance(trainAugCsvFname, str):
        trainAugCsvFnamePath = os.path.join(dataDir, trainAugCsvFname)
    if valCsvFname and isinstance(valCsvFname, str):
        valCsvFnamePath = os.path.join(dataDir, valCsvFname)

    # If feature CSVs exist, prefer them
    try:
        trainFeaturesName = configObj['dataFileNames'].get('trainFeaturesFileCsv')
        valFeaturesName = configObj['dataFileNames'].get('valFeaturesFileCsv')
        testFeaturesName = configObj['dataFileNames'].get('testFeaturesFileCsv')
        # Ensure these are strings, not booleans
        if not isinstance(trainFeaturesName, str):
            trainFeaturesName = None
        if not isinstance(valFeaturesName, str):
            valFeaturesName = None
        if not isinstance(testFeaturesName, str):
            testFeaturesName = None
    except Exception:
        trainFeaturesName = None
        valFeaturesName = None
        testFeaturesName = None

    if trainFeaturesName is not None:
        candidate = os.path.join(dataDir, trainFeaturesName)
        if os.path.exists(candidate):
            print(f"{TAG}: Using train features CSV {candidate}")
            trainAugCsvFnamePath = candidate
    if valFeaturesName is not None:
        candidate = os.path.join(dataDir, valFeaturesName)
        if os.path.exists(candidate):
            print(f"{TAG}: Using validation features CSV {candidate}")
            valCsvFnamePath = candidate
    # Fallback: if validation path is still unresolved or file does not exist, try test features
    # This handles k-fold modes where no explicit valData.csv / valFeatures.csv is created;
    # the inner fold test split is used for validation instead.
    if (valCsvFnamePath is None or not os.path.exists(valCsvFnamePath)):
        if testFeaturesName is not None:
            candidate = os.path.join(dataDir, testFeaturesName)
            if os.path.exists(candidate):
                print(f"{TAG}: Validation file not found, falling back to test features CSV {candidate}")
                valCsvFnamePath = candidate
    
    if trainAugCsvFnamePath is None:
        raise ValueError(f"{TAG}: No training data file specified or found")
    if valCsvFnamePath is None:
        raise ValueError(f"{TAG}: No validation data file specified or found")
    
    return trainAugCsvFnamePath, valCsvFnamePath


def load_and_preprocess_data(trainCsvPath, valCsvPath, nnModel, inputDims, debug, TAG, return_train_df=False):
    """Load and preprocess training and validation data.

    Phase 0/1 optimisations:
      - dtype float32 + usecols (via _build_optimized_read_params) to halve
        DataFrame RAM vs float64 (nnTrainer.py:69 accel cols).
      - sequential free: train_df freed before loading val, and list freed
        immediately after np conversion, to avoid 4× duplication.
      - _log_mem markers after each heavy phase for baseline profiling.
      - np.array(..., dtype=np.float32) early to keep peak at 4B not 8B.
    
    Args:
        trainCsvPath: Path to training CSV
        valCsvPath: Path to validation CSV
        nnModel: Model instance for data preprocessing
        inputDims: Input dimensions (1 or 2)
        debug: Debug flag
        TAG: Tag for logging
    
    Returns:
        tuple: (xTrain, yTrain, xVal, yVal, nClasses)
    """
    _log_mem(f"{TAG} load_and_preprocess start")
    # ---- Load training data ----
    print(f"{TAG}: Loading training data from file {trainCsvPath}")
    if not os.path.exists(trainCsvPath):
        print(f"ERROR: File {trainCsvPath} does not exist")
        exit(-1)

    # Phase 1: build optimized dtype/usecols (float32, int32) if possible
    train_read_params = _build_optimized_read_params(trainCsvPath, nnModel, use_float32=True)
    if train_read_params["dtype"] is not None:
        uc = train_read_params["usecols"]
        print(f"{TAG}: Using optimized read (float32, usecols={len(uc) if uc else 'all'} cols)")
    train_df = augmentData.loadCsv(trainCsvPath, debug=debug, dtype=train_read_params["dtype"], usecols=train_read_params["usecols"])
    print(f"{TAG}: Loaded {len(train_df)} training datapoints")
    _log_mem(f"{TAG} after train load", f"rows={len(train_df)} cols={len(train_df.columns)}")

    print(f"{TAG}: Re-formatting training data")
    xTrain_list, yTrain_list, used_train_rows = df2trainingData(train_df, nnModel, return_row_indices=True)
    # Phase 1: avoid extra .copy() — iloc already copies; just reset_index
    train_df_used = train_df.iloc[used_train_rows].reset_index(drop=True)
    _log_mem(f"{TAG} after train df2trainingData", f"list_len={len(xTrain_list)} used={len(used_train_rows)}")

    # Phase 1: free full train_df before converting — keep only used subset if needed
    del train_df
    gc.collect()
    _log_mem(f"{TAG} after freeing train_df")

    print(f"{TAG}: Converting to np arrays")
    try:
        xTrain = np.array(xTrain_list, dtype=np.float32)
    except ValueError as e:
        print("Failed simple array conversion - trying concatenate...")
        # Concatenate with float32 to avoid float64 peak
        xTrain = np.concatenate([np.asarray(v, dtype=np.float32) for v in xTrain_list])
    # Explicitly free the Python list (was 28B/float) before proceeding
    del xTrain_list
    gc.collect()
    yTrain = np.array(yTrain_list, dtype=np.int32)
    del yTrain_list
    gc.collect()
    _log_mem(f"{TAG} after train np conversion", f"xTrain.shape={xTrain.shape} dtype={xTrain.dtype}")

    print(f"xTrain.shape={xTrain.shape}, yTrain.shape={yTrain.shape}")
    print(f"{TAG}: re-shaping array for training")

    if xTrain.ndim == 2:
        xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    elif xTrain.ndim == 3:
        # Keep channel-last tensors as-is, e.g. (batch, 750, 3) for XYZ mode.
        pass
    elif xTrain.ndim == 4 and inputDims == 2:
        xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1))
    else:
        print(f"ERROR - unsupported xTrain shape {xTrain.shape} for inputDims={inputDims}")
        exit(-1)
    _log_mem(f"{TAG} after train reshape", f"xTrain.shape={xTrain.shape}")

    # ---- Load validation data (now that train_df is freed) ----
    print(f"{TAG}: Loading validation data from file {valCsvPath}")
    if not os.path.exists(valCsvPath):
        print(f"ERROR: File {valCsvPath} does not exist")
        exit(-1)

    val_read_params = _build_optimized_read_params(valCsvPath, nnModel, use_float32=True)
    if val_read_params["dtype"] is not None:
        uc2 = val_read_params["usecols"]
        print(f"{TAG}: Using optimized read for val (float32, usecols={len(uc2) if uc2 else 'all'} cols)")
    df = augmentData.loadCsv(valCsvPath, debug=debug, dtype=val_read_params["dtype"], usecols=val_read_params["usecols"])
    print(f"{TAG}: Loaded {len(df)} validation datapoints")
    _log_mem(f"{TAG} after val load", f"rows={len(df)} cols={len(df.columns)}")

    print(f"{TAG}: Re-formatting validation data")
    xVal_list, yVal_list = df2trainingData(df, nnModel)
    _log_mem(f"{TAG} after val df2trainingData", f"list_len={len(xVal_list)}")
    # Free val DataFrame immediately
    del df
    gc.collect()
    _log_mem(f"{TAG} after freeing val df")

    print(f"{TAG}: Converting to np arrays")
    try:
        xVal = np.array(xVal_list, dtype=np.float32)
    except ValueError as e:
        print("Failed simple array conversion - trying concatenate...")
        xVal = np.concatenate([np.asarray(v, dtype=np.float32) for v in xVal_list])
    del xVal_list
    gc.collect()
    yVal = np.array(yVal_list, dtype=np.int32)
    del yVal_list
    gc.collect()
    _log_mem(f"{TAG} after val np conversion", f"xVal.shape={xVal.shape} dtype={xVal.dtype}")

    print(f"xVal.shape={xVal.shape}, yVal.shape={yVal.shape}")
    print(f"{TAG}: re-shaping array for validation")

    if xVal.ndim == 2:
        xVal = xVal.reshape((xVal.shape[0], xVal.shape[1], 1))
    elif xVal.ndim == 3:
        # Keep channel-last tensors as-is, e.g. (batch, 750, 3) for XYZ mode.
        pass
    elif xVal.ndim == 4 and inputDims == 2:
        xVal = xVal.reshape((xVal.shape[0], xVal.shape[1], xVal.shape[2], 1))
    else:
        print(f"ERROR - unsupported xVal shape {xVal.shape} for inputDims={inputDims}")
        exit(-1)
    _log_mem(f"{TAG} after val reshape", f"xVal.shape={xVal.shape}")

    nClasses = len(np.unique(yTrain))
    print(f"nClasses={nClasses}")
    print(f"Training using {np.count_nonzero(yTrain == 1)} seizure datapoints and {np.count_nonzero(yTrain == 0)} false alarm datapoints")
    _log_mem(f"{TAG} load_and_preprocess done")

    if return_train_df:
        return xTrain, yTrain, xVal, yVal, nClasses, train_df_used
    # If caller does not need train_df_used, free it now
    del train_df_used
    gc.collect()
    return xTrain, yTrain, xVal, yVal, nClasses


def plot_training_history(history, modelFnameRoot, dataDir, framework='tensorflow'):
    """Plot and save training history.
    
    Args:
        history: Training history (dict for PyTorch, History object for TensorFlow)
        modelFnameRoot: Base filename for saving plots
        dataDir: Directory to save plots
        framework: 'tensorflow' or 'pytorch'
    """
    print("Plotting training history")
    
    if framework == 'tensorflow':
        # TensorFlow history is a History object with .history dict
        val_acc = np.array(history.history['val_sparse_categorical_accuracy'])
        acc = np.array(history.history['sparse_categorical_accuracy'])
        loss = np.array(history.history['loss'])
        val_loss = np.array(history.history['val_loss'])
        metric_name = 'sparse_categorical_accuracy'
        sensitivity = None
        far = None
    else:
        # PyTorch history is already a dict
        val_acc = np.array(history['val_accuracy'])
        acc = np.array(history['accuracy'])
        loss = np.array(history['loss'])
        val_loss = np.array(history['val_loss'])
        metric_name = 'accuracy'
        sensitivity = np.array(history.get('sensitivity', []))
        far = np.array(history.get('far', []))
        youden_arr = np.array(history.get('youden', []))
        # Backwards compatibility: derive youden if not stored
        if youden_arr.size == 0 and sensitivity.size > 0 and far.size > 0 and sensitivity.size == far.size:
            youden_arr = sensitivity - far
    
    # Plot 1: Combined metrics
    plt.figure(figsize=(12, 8))
    plt.plot(val_acc, "r--", label="val_accuracy")
    plt.plot(acc, "g--", label="accuracy")
    plt.plot(loss, "y--", label="Loss")
    plt.plot(val_loss, "p-", label="val_loss")
    plt.title("Training session's progress over iterations")
    plt.legend(loc='lower left')
    plt.ylabel('Training Progress (Loss/Accuracy)')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(dataDir, f"{modelFnameRoot}_training.png"))
    plt.close()
    
    # Plot 2: Accuracy only
    plt.figure()
    if framework == 'tensorflow':
        plt.plot(history.history[metric_name])
        plt.plot(history.history["val_" + metric_name])
    else:
        plt.plot(history[metric_name])
        plt.plot(history["val_" + metric_name])
    plt.title(f"model {metric_name}")
    plt.ylabel(metric_name, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(dataDir, f"{modelFnameRoot}_training2.png"))
    plt.close()

    # Plot 3: Sensitivity (TPR) and FAR (FPR) over validation checkpoints (PyTorch only)
    if sensitivity is not None and far is not None and sensitivity.size > 0 and far.size > 0:
        fig, ax1 = plt.subplots()
        # Plot TPR on left Y-axis
        color = 'b'
        ax1.set_xlabel("validation checkpoints", fontsize="large")
        ax1.set_ylabel("TPR", color=color, fontsize="large")
        line1 = ax1.plot(sensitivity, "b-", label="TPR")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Plot FPR on right Y-axis
        ax2 = ax1.twinx()
        color = 'm'
        ax2.set_ylabel("FAR (FPR)", color=color, fontsize="large")
        line2 = ax2.plot(far, "m-", label="FAR (FPR)")
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 0.1)
        
        # Add title and combined legend
        fig.suptitle("Validation TPR/FPR over training", fontsize="large")
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")
        
        fig.tight_layout()
        plt.savefig(os.path.join(dataDir, f"{modelFnameRoot}_training_tpr_fpr.png"))
        plt.close()

    # Plot 4: Youden (TPR - FPR) over validation checkpoints (PyTorch only)
    if youden_arr is not None and youden_arr.size > 0:
        plt.figure()
        plt.plot(youden_arr, "g-", label="Youden (TPR - FPR)")
        plt.title("Validation Youden over training", fontsize="large")
        plt.ylabel("Youden (TPR - FPR)", fontsize="large")
        plt.xlabel("validation checkpoints", fontsize="large")
        plt.ylim(-1, 1)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(dataDir, f"{modelFnameRoot}_training_youden.png"))
        plt.close()

    # Note: Threshold analysis plots (event-level, production-level and
    # event-vs-production for both all seizures and tonic-clonic seizures) are
    # generated by nnTester.testModel() as:
    #   {model}_event_threshold_analysis.png
    #   {model}_production_threshold_analysis.png
    #   {model}_event_vs_production_threshold_analysis.png
    #   {model}_event_threshold_analysis_tonic_clonic.png
    #   {model}_production_threshold_analysis_tonic_clonic.png
    #   {model}_event_vs_production_threshold_analysis_tonic_clonic.png
    # The last file (event_vs_production for tonic-clonic) is required for
    # the PDF summary (event_analysis_report.pdf) – see analyzeEventResults.generate_plots
    # which includes both all-seizure and tonic-clonic event-vs-production graphs.


def _parse_datetime_safe_trainer(value):
    """Robustly parse event/datapoint timestamps into naive datetime, or None.
    Mirrors nnTester._parse_datetime_safe for use in chart generation.
    """
    if value is None:
        return None
    # Import here to avoid circular deps at module load
    from datetime import datetime as _dt
    import pandas as _pd
    import numpy as _np
    if isinstance(value, _dt):
        return value.replace(tzinfo=None) if value.tzinfo is not None else value
    if isinstance(value, _pd.Timestamp):
        if _pd.isna(value):
            return None
        try:
            py = value.to_pydatetime()
            return py.replace(tzinfo=None) if py.tzinfo is not None else py
        except Exception:
            return None
    if isinstance(value, float) and _np.isnan(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f",
                "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return _dt.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    try:
        ts = _pd.to_datetime(s, errors='coerce', utc=False)
        if _pd.isna(ts):
            return None
        ts = _pd.Timestamp(ts)
        py = ts.to_pydatetime()
        return py.replace(tzinfo=None) if py.tzinfo is not None else py
    except Exception:
        return None


def _parse_seizure_times_trainer(seizure_times):
    """Parse seizureTimes into [start, end] floats or None.
    Handles list/tuple, JSON string like \"[-95.0, 70.0]\", or None.
    """
    if seizure_times is None:
        return None
    # Import locally
    import json as _json
    if isinstance(seizure_times, str):
        s = seizure_times.strip()
        if not s or s.lower() == 'nan':
            return None
        try:
            parsed = _json.loads(s)
            seizure_times = parsed
        except Exception:
            # Try to handle single values or malformed
            return None
    if not isinstance(seizure_times, (list, tuple)):
        return None
    if len(seizure_times) < 2:
        return None
    try:
        start = float(seizure_times[0])
        end = float(seizure_times[1])
        return [start, end]
    except Exception:
        return None


def _collect_acc_cols_trainer(columns, prefix):
    """Collect accelerometer columns for prefix ('M','X','Y','Z') sorted by index.

    Handles both plain names like M000 and history names like M000_t-0.
    Mirrors df2trainingData column detection but simpler for plotting.
    """
    cols = []
    for c in columns:
        if not isinstance(c, str):
            continue
        if not c.startswith(prefix):
            continue
        rest = c[len(prefix):]
        # Strip suffix like _t-0 if present
        if '_' in rest:
            rest = rest.split('_')[0]
        if not rest:
            continue
        # Rest should be digits
        if rest.isdigit():
            try:
                idx = int(rest)
                cols.append((idx, c))
            except Exception:
                continue
        elif rest.lstrip('-').isdigit():
            # Handle negative? unlikely for accel columns
            continue
    cols.sort(key=lambda x: x[0])
    return [c for _, c in cols]


def _has_3d_data(df_group, x_cols, y_cols, z_cols):
    """Return True if 3D acceleration data appears present (non-zero, non-NaN)."""
    if not x_cols or not y_cols or not z_cols:
        return False
    try:
        import pandas as _pd
        import numpy as _np
        # Check that columns exist in df
        for col_set in (x_cols, y_cols, z_cols):
            for c in col_set:
                if c not in df_group.columns:
                    return False
        # Stack numeric values
        vals_x = _pd.to_numeric(df_group[x_cols].stack(), errors='coerce').fillna(0).to_numpy()
        vals_y = _pd.to_numeric(df_group[y_cols].stack(), errors='coerce').fillna(0).to_numpy()
        vals_z = _pd.to_numeric(df_group[z_cols].stack(), errors='coerce').fillna(0).to_numpy()
        # Consider 3D present if any axis has sum of absolute values > small epsilon
        total = float(abs(vals_x).sum() + abs(vals_y).sum() + abs(vals_z).sum())
        return total > 1e-6
    except Exception:
        return False


def plot_event_chart(eventId, typeStr, subType, desc, seizureTimes, eventDataTime,
                     df_group, probs, out_path, titlePrefix=None):
    """Create a two-panel chart for a single seizure event.

    Top panel: raw accelerometer vs time (X/Y/Z if 3D available, else magnitude)
               with seizure interval shading; heart rate on secondary y-axis.
    Bottom panel: seizure probability vs time with same seizure shading.

    Args:
        eventId: event identifier (for title/filename)
        typeStr: event type string (e.g. "Seizure")
        subType: event subtype string (e.g. "Tonic-Clonic")
        desc: event description string (subtitle, may be 'N/A')
        seizureTimes: [start_offset, end_offset] seconds from event start, or raw string/list
        eventDataTime: event reference time (string/datetime) for time base
        df_group: DataFrame rows for this event, must contain 'dataTime','hr','M000...' etc
        probs: array-like seizure probabilities aligned with df_group rows (same order before sorting)
        out_path: file path to save PNG
        titlePrefix: optional model prefix for display (not used in filename)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from datetime import timedelta

    # Parse seizure interval
    seizure_interval = _parse_seizure_times_trainer(seizureTimes)
    seizure_start = seizure_interval[0] if seizure_interval else None
    seizure_end = seizure_interval[1] if seizure_interval else None
    has_seizure_shading = seizure_interval is not None and seizure_end > seizure_start

    # Parse event reference time
    event_dt = _parse_datetime_safe_trainer(eventDataTime)

    # Prepare sorted items by datapoint time
    # Build list of (parsed_time, row_series, prob)
    probs_arr = np.asarray(probs, dtype=float) if probs is not None else np.array([])
    n_rows = len(df_group)
    # Ensure df_group is a DataFrame; if single row, handle
    if n_rows == 0:
        # Nothing to plot
        return False

    # Create aligned list
    items = []
    for i in range(n_rows):
        row = df_group.iloc[i]
        # prob for this row; if probs_arr shorter, use NaN
        p = float(probs_arr[i]) if i < len(probs_arr) and not np.isnan(probs_arr[i]) else np.nan
        dt_raw = row['dataTime'] if 'dataTime' in df_group.columns else None
        parsed = _parse_datetime_safe_trainer(dt_raw)
        items.append((parsed, row, p, i))

    # Sort by parsed time (None last), keep original order for fallback
    def _sort_key(item):
        parsed = item[0]
        idx = item[3]
        return (parsed is None, parsed if parsed is not None else idx)
    items_sorted = sorted(items, key=_sort_key)

    # Detect column sets
    all_cols = list(df_group.columns)
    m_cols = _collect_acc_cols_trainer(all_cols, 'M')
    x_cols = _collect_acc_cols_trainer(all_cols, 'X')
    y_cols = _collect_acc_cols_trainer(all_cols, 'Y')
    z_cols = _collect_acc_cols_trainer(all_cols, 'Z')
    use_3d = _has_3d_data(df_group, x_cols, y_cols, z_cols)

    # Build time series
    raw_time = []
    raw_mag = []
    x_time = []
    x_vals = []
    y_time = []
    y_vals = []
    z_time = []
    z_vals = []
    hr_time = []
    hr_vals = []
    prob_time = []
    prob_vals = []

    for rank, (parsed, row, p, orig_idx) in enumerate(items_sorted):
        if event_dt is not None and parsed is not None:
            try:
                time_sec = (parsed - event_dt).total_seconds()
            except Exception:
                time_sec = float(rank * 5)
        else:
            # Fallback: 5 seconds per datapoint from start
            time_sec = float(rank * 5)

        # HR - plotted at centre of 5-sec window (time_sec is window START as in event_editor)
        # event_editor uses time_sec for HR at window start, but for better alignment with
        # raw samples (time_sec + n/25) and seizure shading (relative to event dataTime),
        # place HR/prob at window centre (time_sec + 2.5) - matches dataSummariser centre.
        # This fixes misalignment observed in e.g. 5288 where prob/hr at left edge appeared offset.
        centre_sec = time_sec + 2.5
        try:
            hr_raw = row['hr'] if 'hr' in row.index else None
            if hr_raw is not None and not (isinstance(hr_raw, float) and np.isnan(hr_raw)):
                hr_num = float(hr_raw)
                # Treat -1 or 0 as missing? In event_editor they check hr>0
                # Keep all but will filter later
                hr_time.append(centre_sec)
                hr_vals.append(hr_num if hr_num >= 0 else 0)
            else:
                hr_time.append(centre_sec)
                hr_vals.append(0)
        except Exception:
            hr_time.append(centre_sec)
            hr_vals.append(0)

        # Probability - one per datapoint, also at window centre for alignment with seizure shading
        prob_time.append(centre_sec)
        prob_vals.append(p)

        # Accelerometer samples - expand 125 samples per datapoint
        # Each sample offset n/25 seconds from datapoint time
        if use_3d:
            try:
                x_arr = pd.to_numeric(row[x_cols], errors='coerce').fillna(0).to_numpy(dtype=float) if x_cols else np.array([])
                y_arr = pd.to_numeric(row[y_cols], errors='coerce').fillna(0).to_numpy(dtype=float) if y_cols else np.array([])
                z_arr = pd.to_numeric(row[z_cols], errors='coerce').fillna(0).to_numpy(dtype=float) if z_cols else np.array([])
            except Exception:
                x_arr = y_arr = z_arr = np.array([])
            n_samples = len(x_arr) if len(x_arr) > 0 else 0
            # If lengths mismatch, use min
            if n_samples > 0:
                for n in range(n_samples):
                    t = time_sec + n / 25.0
                    x_time.append(t); x_vals.append(float(x_arr[n]) if n < len(x_arr) else 0.0)
                    y_time.append(t); y_vals.append(float(y_arr[n]) if n < len(y_arr) else 0.0)
                    z_time.append(t); z_vals.append(float(z_arr[n]) if n < len(z_arr) else 0.0)
            else:
                # No 3D data for this row, skip
                pass
        else:
            try:
                if m_cols:
                    m_arr = pd.to_numeric(row[m_cols], errors='coerce').fillna(0).to_numpy(dtype=float)
                else:
                    m_arr = np.array([])
            except Exception:
                m_arr = np.array([])
            for n in range(len(m_arr)):
                t = time_sec + n / 25.0
                raw_time.append(t); raw_mag.append(float(m_arr[n]))

    # Convert to arrays
    hr_time = np.asarray(hr_time, dtype=float)
    hr_vals = np.asarray(hr_vals, dtype=float)
    prob_time = np.asarray(prob_time, dtype=float)
    prob_vals = np.asarray(prob_vals, dtype=float)

    # Create figure with two vertically stacked axes, sharex
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                            gridspec_kw={'height_ratios': [2, 1]})
    # Top panel: accelerometer
    if use_3d and len(x_vals) > 0:
        ax_top.plot(x_time, x_vals, label='X', color='#d62728', alpha=0.7, linewidth=0.8)
        ax_top.plot(y_time, y_vals, label='Y', color='#2ca02c', alpha=0.7, linewidth=0.8)
        ax_top.plot(z_time, z_vals, label='Z', color='#1f77b4', alpha=0.7, linewidth=0.8)
        ax_top.set_ylabel('Acceleration (milli-g)', fontsize=10)
        ax_top.legend(fontsize=8, loc='upper right')
    else:
        if len(raw_time) > 0:
            ax_top.plot(raw_time, raw_mag, color='#1f77b4', alpha=0.7, linewidth=0.8, label='Magnitude')
            ax_top.set_ylabel('Acceleration Magnitude (milli-g)', fontsize=10)
        else:
            ax_top.text(0.5, 0.5, 'No accelerometer data', transform=ax_top.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
            ax_top.set_ylabel('Acceleration (milli-g)', fontsize=10)

    ax_top.grid(True, alpha=0.3)
    ax_top.set_xlabel('Time (seconds from event start)', fontsize=10)
    # Fixed y for easier comparison across events
    ax_top.set_ylim(0, 2500)

    # Seizure shading on top
    if has_seizure_shading:
        ax_top.axvspan(seizure_start, seizure_end, alpha=0.2, color='red', label='Seizure Period')
        ax_top.axvline(x=seizure_start, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax_top.axvline(x=seizure_end, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        # Ensure legend shows shading if not already
        # Add text labels if space - use fixed ylim 2500 for consistent placement
        try:
            y_pos = 2500 * 0.95
            ax_top.text(seizure_start, y_pos, f'Start: {seizure_start:.1f}s',
                        rotation=90, va='top', ha='right', color='red', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
            ax_top.text(seizure_end, y_pos, f'End: {seizure_end:.1f}s',
                        rotation=90, va='top', ha='right', color='red', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
        except Exception:
            pass

    # Secondary y-axis for heart rate on top chart
    # Only plot if we have any hr >0 - fixed 0-200 bpm for comparison
    has_hr = len(hr_vals) > 0 and np.any(hr_vals > 0)
    if has_hr:
        ax_hr = ax_top.twinx()
        ax_hr.plot(hr_time, hr_vals, color='darkorange', marker='o', linestyle='-', linewidth=1.5, markersize=3, label='Heart Rate')
        ax_hr.set_ylabel('Heart Rate (bpm)', color='darkorange', fontsize=10)
        ax_hr.tick_params(axis='y', labelcolor='darkorange')
        ax_hr.set_ylim(0, 200)
    else:
        # Still create fixed axis for consistent scale even if no HR data
        ax_hr = ax_top.twinx()
        ax_hr.set_ylabel('Heart Rate (bpm)', color='darkorange', fontsize=10)
        ax_hr.tick_params(axis='y', labelcolor='darkorange')
        ax_hr.set_ylim(0, 200)

    # Bottom panel: seizure probability - fixed 0-1
    # Filter out NaN probabilities for plotting but keep time alignment for shading
    mask_valid = ~np.isnan(prob_vals)
    if np.any(mask_valid):
        ax_bottom.plot(prob_time[mask_valid], prob_vals[mask_valid], color='purple', marker='o', linestyle='-', linewidth=1.5, markersize=3, label='Seizure Probability')
    else:
        ax_bottom.text(0.5, 0.5, 'No probability data', transform=ax_bottom.transAxes,
                       ha='center', va='center', fontsize=10, color='gray')
    ax_bottom.set_ylabel('Seizure Probability', fontsize=10)
    ax_bottom.set_xlabel('Time (seconds from event start)', fontsize=10)
    ax_bottom.set_ylim(0, 1)
    ax_bottom.grid(True, alpha=0.3)
    # Threshold line at 0.5
    ax_bottom.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Threshold 0.5')
    if has_seizure_shading:
        ax_bottom.axvspan(seizure_start, seizure_end, alpha=0.2, color='red')
        ax_bottom.axvline(x=seizure_start, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax_bottom.axvline(x=seizure_end, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax_bottom.legend(fontsize=8, loc='upper right')

    # Title with event number, type, subtype and desc subtitle
    # Clean strings
    def _clean(val):
        if val is None:
            return 'N/A'
        s = str(val).strip()
        if not s or s.lower() in ('nan', 'none', 'n/a', 'null'):
            return 'N/A'
        return s
    type_clean = _clean(typeStr)
    sub_clean = _clean(subType)
    desc_clean = _clean(desc)
    # Title line
    title_main = f"Event {eventId}: {type_clean} / {sub_clean}"
    if titlePrefix:
        title_main = f"{titlePrefix} - {title_main}"
    fig.suptitle(title_main, fontsize=13, fontweight='bold', y=0.98)
    # Subtitle if desc is meaningful
    if desc_clean != 'N/A':
        # Truncate very long descs to ~200 chars
        desc_disp = desc_clean
        if len(desc_disp) > 500:
            desc_disp = desc_disp[:500] + '...'
        fig.text(0.5, 0.92, desc_disp, ha='center', va='top', fontsize=9, style='italic',
                 wrap=True, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    # Adjust for subtitle
    try:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig)
    return True


def _is_tonic_clonic(typeStr, subType):
    """Return True if event is tonic-clonic seizure (case-insensitive, handles 'Tonic-Clonic' etc)."""
    try:
        sub = str(subType).strip().lower() if subType is not None else ""
        typ = str(typeStr).strip().lower() if typeStr is not None else ""
        # Must be seizure type
        if typ != "seizure":
            # also handle typeStr containing subType e.g. "Seizure/Tonic-Clonic"
            if "seizure" not in typ:
                return False
        return "tonic" in sub and "clonic" in sub
    except Exception:
        return False


def _get_event_type_subtype(eventId, event_details_map, event_stats_df, df_group):
    """Helper to resolve typeStr/subType/desc for an eventId."""
    eid_str = str(eventId)
    meta = event_details_map.get(eid_str, {}) if isinstance(event_details_map, dict) else {}
    if not meta and isinstance(event_details_map, dict):
        meta = event_details_map.get(eventId, {})
    typeStr = meta.get('typeStr', meta.get('type', 'N/A'))
    if (typeStr == 'N/A' or not typeStr) and event_stats_df is not None and 'typeStr' in event_stats_df.columns:
        try:
            row_ev = event_stats_df[event_stats_df['eventId'].astype(str) == eid_str]
            if len(row_ev) > 0:
                typeStr = row_ev.iloc[0]['typeStr']
        except Exception:
            pass
    if (typeStr == 'N/A' or not typeStr) and 'typeStr' in df_group.columns:
        try:
            typeStr = str(df_group.iloc[0]['typeStr'])
        except Exception:
            pass
    subType = meta.get('subType', meta.get('subtype', 'N/A'))
    if (subType == 'N/A' or not subType) and event_stats_df is not None and 'subType' in event_stats_df.columns:
        try:
            row_ev = event_stats_df[event_stats_df['eventId'].astype(str) == eid_str]
            if len(row_ev) > 0:
                subType = row_ev.iloc[0]['subType']
        except Exception:
            pass
    desc = meta.get('desc', meta.get('Description', 'N/A'))
    if (desc == 'N/A' or not desc) and event_stats_df is not None and 'desc' in event_stats_df.columns:
        try:
            row_ev = event_stats_df[event_stats_df['eventId'].astype(str) == eid_str]
            if len(row_ev) > 0:
                desc = row_ev.iloc[0]['desc']
        except Exception:
            pass
    seizureTimes = meta.get('seizureTimes', None)
    eventDataTime = meta.get('dataTime', None)
    if eventDataTime is None:
        try:
            eventDataTime = df_group.iloc[0]['dataTime']
        except Exception:
            eventDataTime = None
    return typeStr, subType, desc, seizureTimes, eventDataTime


def _generate_charts_for_event_list(output_subdir, target_events, df_use, pSeizure, event_details_map, event_stats_df, modelFnameRoot, titlePrefix, debug, TAG):
    """Inner helper to generate charts for a list of eventIds into output_subdir. Returns count."""
    import os as _os
    import numpy as _np
    _os.makedirs(output_subdir, exist_ok=True)
    count = 0
    for eventId in target_events:
        eid_str = str(eventId)
        try:
            mask = df_use['eventId'].astype(str) == eid_str
            df_group = df_use[mask]
        except Exception:
            df_group = df_use[df_use['eventId'] == eventId]
        if len(df_group) == 0:
            if debug:
                print(f"{TAG}: Skipping event {eventId} - no datapoints in df")
            continue
        try:
            probs = pSeizure[mask.to_numpy()]
        except Exception:
            try:
                probs = pSeizure[df_group.index.to_numpy()]
            except Exception:
                probs = pSeizure[:len(df_group)]
        typeStr, subType, desc, seizureTimes, eventDataTime = _get_event_type_subtype(eventId, event_details_map, event_stats_df, df_group)
        safe_eid = str(eventId).replace('/', '_').replace('\\', '_').replace(' ', '_')
        fname = f"{safe_eid}_{str(modelFnameRoot).replace('/', '_')}.png" if modelFnameRoot else f"{safe_eid}.png"
        out_path = _os.path.join(output_subdir, fname)
        try:
            ok = plot_event_chart(eventId=eventId, typeStr=typeStr, subType=subType, desc=desc, seizureTimes=seizureTimes, eventDataTime=eventDataTime, df_group=df_group, probs=probs, out_path=out_path, titlePrefix=titlePrefix)
            if ok:
                count += 1
                if debug or count <= 3:
                    print(f"{TAG}: Saved chart for event {eventId} to {out_path}")
        except Exception as e:
            print(f"{TAG}: Error generating chart for event {eventId}: {e}")
            if debug:
                import traceback as _tb
                _tb.print_exc()
            continue
    return count


def generate_event_charts(outputDir, df, prediction_proba, event_details_map, event_stats_df=None,
                          modelFnameRoot=None, titlePrefix=None, debug=False, only_seizure=True):
    """Generate per-event chart PNGs in outputDir/eventData/{tonicClonic,allSeizures,falsePositives}.

    This is the main entry point called from nnTester.testModel.

    Three subfolders are created:
      - allSeizures: all events where type == Seizure (true_label==1 or typeStr seizure)
      - tonicClonic: subset of allSeizures where subType is Tonic-Clonic
      - falsePositives: non-seizure events (true_label==0) where model_pred==1

    Each subfolder gets the same two-panel chart (top: accel + HR secondary, bottom: prob)
    with fixed y: 0-2500 mg, 0-200 bpm, 0-1 prob, seizure shading, title with event/type/subType+desc.

    Args:
        outputDir: base output directory where eventData subfolders will be created
        df: filtered datapoint DataFrame whose row order matches prediction_proba
        prediction_proba: (n_datapoints, n_classes) array; column 1 is seizure prob
        event_details_map: dict str(eventId) -> dict with 'typeStr','subType','desc','dataTime','seizureTimes'
        event_stats_df: optional per-event DataFrame with columns 'eventId','true_label','model_pred','typeStr','subType'
        modelFnameRoot: optional model name for logging/title
        titlePrefix: optional title prefix for chart titles
        debug: bool for verbose logging
        only_seizure: kept for backward compat (ignored, all three categories are generated)

    Returns:
        int: total number of charts generated across all three subfolders
    """
    import os as _os
    import numpy as _np
    import pandas as _pd

    TAG = "nnTrainer.generate_event_charts()"
    if df is None or prediction_proba is None:
        print(f"{TAG}: df or prediction_proba is None, skipping chart generation")
        return 0

    # Prepare pSeizure and df_use
    try:
        pSeizure = _np.asarray(prediction_proba[:, 1], dtype=float)
    except Exception as e:
        print(f"{TAG}: Could not extract seizure probabilities: {e}")
        return 0
    n_df = len(df)
    n_prob = len(pSeizure)
    if n_df != n_prob:
        print(f"{TAG}: Warning - df rows ({n_df}) != prob rows ({n_prob}), using min")
        min_n = min(n_df, n_prob)
        df_use = df.iloc[:min_n].copy()
        pSeizure = pSeizure[:min_n]
    else:
        df_use = df

    available_ids = set(_pd.Series(df_use['eventId']).astype(str).tolist()) if 'eventId' in df_use.columns else set()

    # Build event-level lookup for filtering
    # Use event_stats_df if available, otherwise fall back to df grouping
    event_level_rows = {}
    if event_stats_df is not None and len(event_stats_df) > 0 and 'eventId' in event_stats_df.columns:
        for _, r in event_stats_df.iterrows():
            event_level_rows[str(r['eventId'])] = r

    # Determine target lists for three categories
    all_seizure_events = []
    tonic_clonic_events = []
    false_positive_events = []

    if event_stats_df is not None and 'true_label' in event_stats_df.columns and 'eventId' in event_stats_df.columns:
        for _, r in event_stats_df.iterrows():
            eid = r['eventId']
            eid_str = str(eid)
            if eid_str not in available_ids:
                continue
            true_label = r.get('true_label', None)
            try:
                true_label = int(true_label) if true_label is not None and str(true_label).lower() not in ('nan','') else None
            except Exception:
                true_label = None
            # need type/subType for tonic check
            typeStr = r.get('typeStr', r.get('Type', ''))
            if (typeStr is None or str(typeStr).strip().lower() in ('nan','n/a','')):
                # fallback to details map
                meta = event_details_map.get(eid_str, {}) if isinstance(event_details_map, dict) else {}
                typeStr = meta.get('typeStr', meta.get('type', ''))
            subType = r.get('subType', r.get('SubType', ''))
            if (subType is None or str(subType).strip().lower() in ('nan','n/a','')):
                meta = event_details_map.get(eid_str, {}) if isinstance(event_details_map, dict) else {}
                subType = meta.get('subType', meta.get('subtype', ''))
            is_seizure = (true_label == 1) or (str(typeStr).strip().lower() == 'seizure')
            if is_seizure:
                all_seizure_events.append(eid)
                if _is_tonic_clonic(typeStr, subType):
                    tonic_clonic_events.append(eid)
            # false positives: non-seizure predicted as seizure
            model_pred = r.get('model_pred', r.get('ModelPrediction', None))
            try:
                model_pred = int(model_pred) if model_pred is not None and str(model_pred).lower() not in ('nan','') else None
            except Exception:
                model_pred = None
            if true_label == 0 and model_pred == 1:
                false_positive_events.append(eid)
    else:
        # Fallback without event_stats_df: use df grouping
        # allSeizures from df where type==1
        if 'type' in df_use.columns and 'eventId' in df_use.columns:
            all_seizure_events = df_use[df_use['type'] == 1]['eventId'].unique().tolist()
            # tonicClonic by checking details map subType
            for eid in list(all_seizure_events):
                meta = event_details_map.get(str(eid), {}) if isinstance(event_details_map, dict) else {}
                typeStr = meta.get('typeStr', meta.get('type', 'Seizure'))
                subType = meta.get('subType', meta.get('subtype', ''))
                if not _is_tonic_clonic(typeStr, subType):
                    # keep in all but not necessarily tonic
                    pass
            # filter tonic
            tonic_clonic_events = [eid for eid in all_seizure_events if _is_tonic_clonic(event_details_map.get(str(eid), {}).get('typeStr','Seizure'), event_details_map.get(str(eid), {}).get('subType',''))]
        # falsePositives cannot be determined without predictions - leave empty
        false_positive_events = []

    # Ensure uniqueness and available
    all_seizure_events = [eid for eid in all_seizure_events if str(eid) in available_ids]
    tonic_clonic_events = [eid for eid in tonic_clonic_events if str(eid) in available_ids]
    false_positive_events = [eid for eid in false_positive_events if str(eid) in available_ids]

    # Create base and subfolders (always, even if empty, for easier comparison)
    event_data_dir = _os.path.join(outputDir, 'eventData')
    tonic_dir = _os.path.join(event_data_dir, 'tonicClonic')
    all_dir = _os.path.join(event_data_dir, 'allSeizures')
    fp_dir = _os.path.join(event_data_dir, 'falsePositives')
    for d in [event_data_dir, tonic_dir, all_dir, fp_dir]:
        _os.makedirs(d, exist_ok=True)
    print(f"{TAG}: Generating charts - allSeizures:{len(all_seizure_events)} tonicClonic:{len(tonic_clonic_events)} falsePositives:{len(false_positive_events)} in {event_data_dir}")

    total = 0
    total += _generate_charts_for_event_list(all_dir, all_seizure_events, df_use, pSeizure, event_details_map, event_stats_df, modelFnameRoot, titlePrefix, debug, TAG)
    total += _generate_charts_for_event_list(tonic_dir, tonic_clonic_events, df_use, pSeizure, event_details_map, event_stats_df, modelFnameRoot, titlePrefix, debug, TAG)
    total += _generate_charts_for_event_list(fp_dir, false_positive_events, df_use, pSeizure, event_details_map, event_stats_df, modelFnameRoot, titlePrefix, debug, TAG)

    # Keep backward compat: also keep a flat copy of allSeizures in eventData root for any legacy tooling (optional, not required)
    # We do not duplicate to root to avoid clutter; subfolders are the source of truth.

    print(f"{TAG}: Completed generating {total} event charts total ({len(all_seizure_events)} all, {len(tonic_clonic_events)} tonic, {len(false_positive_events)} fp)")
    return total


def trainModel(configObj, dataDir='.', debug=False):
    ''' Create and train a new neural network model, saving it with filename starting 
    with the modelFnameRoot parameter.
    '''
    TAG = "nnTrainer.trainmodel()"
    print("%s" % (TAG))
    
    # Detect framework
    framework = get_framework_from_config(configObj)
    print(f"{TAG}: Using framework: {framework}")
    
    if framework == 'pytorch':
        return trainModel_pytorch(configObj, dataDir, debug)
    else:
        return trainModel_tensorflow(configObj, dataDir, debug)


def trainModel_tensorflow(configObj, dataDir='.', debug=False):
    ''' Create and train a new TensorFlow/Keras neural network model.
    '''
    TAG = "nnTrainer.trainModel_tensorflow()"
    print("%s" % (TAG))
    
    # Import TensorFlow/Keras
    import tensorflow as tf
    from tensorflow import keras
    
    # Load configuration parameters
    params = load_config_params(configObj)

    # Centralized seeding for TF as well
    _seed = get_seed_from_config(configObj)
    seed_all(_seed, debug=debug)
    if _seed is None:
        print(f"{TAG}: randomSeed is null/None -> non-deterministic training")
    else:
        print(f"{TAG}: Using deterministic seed {_seed}")

    # Load model class
    nnModel = load_model_class(params['nnModelClassName'], configObj, framework='tensorflow')
    
    # Resolve data file paths
    trainCsvPath, valCsvPath = resolve_data_file_paths(
        dataDir, params['trainAugCsvFname'], params['valCsvFname'], configObj, TAG
    )
    
    # Load and preprocess data
    xTrain, yTrain, xVal, yVal, nClasses = load_and_preprocess_data(
        trainCsvPath, valCsvPath, nnModel, params['inputDims'], debug, TAG
    )
    
    # Model filename
    modelFname = f"{params['modelFnameRoot']}.keras"
    modelFnamePath = os.path.join(dataDir, modelFname)
    
    # Create or load model
    if os.path.exists(modelFnamePath):
        print(f"Model {modelFnamePath} already exists - loading existing model as starting point for training")
        model = keras.models.load_model(modelFnamePath)
        print(f"Model {modelFnamePath} loaded")
    else:
        print("Creating new Model")
        model = nnModel.makeModel(input_shape=xTrain.shape[1:], num_classes=nClasses, nLayers=params['nLayers'])
    
    keras.utils.plot_model(model, show_shapes=True)

    # Build callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            modelFnamePath, save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=params['earlyStoppingPatience'], 
            verbose=params['trainingVerbosity']),
    ]

    # Determine steps per epoch so schedules can be expressed in steps
    try:
        steps_per_epoch = int(np.ceil(xTrain.shape[0] / float(params['batch_size'])))
    except Exception:
        steps_per_epoch = None

    # Learning rate configuration: either a static lrStart or an advanced schedule
    if params['use_lr_schedule']:
        # Define a LearningRateSchedule implementing warmup -> cosine decay -> cooldown
        class PaperLrSchedule(keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, warmup, main, cooldown, lr_start, lr_peak, lr_main_end, lr_end=0.0):
                self.warmup = float(warmup)
                self.main = float(main)
                self.cooldown = float(cooldown)
                self.lr_start = float(lr_start)
                self.lr_peak = float(lr_peak)
                self.lr_main_end = float(lr_main_end)
                self.lr_end = float(lr_end)

            def __call__(self, step):
                # step will be a scalar tf.Tensor
                step_f = tf.cast(step, tf.float32)
                warmup = tf.cast(self.warmup, tf.float32)
                main = tf.cast(self.main, tf.float32)
                cooldown = tf.cast(self.cooldown, tf.float32)
                total = warmup + main + cooldown

                def _warmup():
                    progress = step_f / warmup
                    return self.lr_start + progress * (self.lr_peak - self.lr_start)

                def _main():
                    # cosine decay from lr_peak to lr_main_end
                    t = (step_f - warmup) / main
                    cosine_decay = 0.5 * (1 + tf.cos(tf.constant(np.pi) * t))
                    return self.lr_main_end + (self.lr_peak - self.lr_main_end) * cosine_decay

                def _cooldown():
                    t = (step_f - warmup - main) / cooldown
                    return self.lr_main_end + t * (self.lr_end - self.lr_main_end)

                return tf.where(step_f < warmup, _warmup(), tf.where(step_f < (warmup + main), _main(), _cooldown()))

        schedule = PaperLrSchedule(params['warmup_steps'], params['main_steps'], params['cooldown_steps'], 
                                    params['lrStart'], params['lr_peak'], params['lr_main_end'], 0.0)

        # Try to use AdamW if available, otherwise fall back to Adam
        opt = None
        if params['use_adamw']:
            try:
                opt = keras.optimizers.AdamW(
                    learning_rate=schedule,
                    beta_1=params['adamw_beta1'],
                    beta_2=params['adamw_beta2'],
                    weight_decay=params['weight_decay']
                )
                print("Using tf.keras.optimizers.AdamW")
            except Exception:
                try:
                    import tensorflow_addons as tfa
                    opt = tfa.optimizers.AdamW(
                        learning_rate=schedule,
                        beta_1=params['adamw_beta1'],
                        beta_2=params['adamw_beta2'],
                        weight_decay=params['weight_decay']
                    )
                    print("Using tensorflow_addons.optimizers.AdamW")
                except Exception:
                    print("AdamW not available, falling back to Adam with scheduled LR")
                    opt = keras.optimizers.Adam(
                        learning_rate=schedule,
                        beta_1=params['adamw_beta1'],
                        beta_2=params['adamw_beta2']
                    )
        else:
            opt = keras.optimizers.Adam(
                learning_rate=schedule,
                beta_1=params['adamw_beta1'],
                beta_2=params['adamw_beta2']
            )

        # when using schedule, do not add ReduceLROnPlateau
    else:
        # keep existing ReduceLROnPlateau behavior
        callbacks.insert(1, keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=params['lrFactor'], 
                                                                patience=params['lrPatience'], min_lr=params['lrMin']))
        opt = keras.optimizers.Adam(learning_rate=params['lrStart'])

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(
        xTrain,
        yTrain,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        callbacks=callbacks,
        validation_data=(xVal, yVal),
        verbose=params['trainingVerbosity']
    )
    
    print(f"Trained using {np.count_nonzero(yTrain == 1)} seizure datapoints and {np.count_nonzero(yTrain == 0)} false alarm datapoints")

    # Plot training history
    plot_training_history(history, params['modelFnameRoot'], dataDir, framework='tensorflow')

    print("Training Complete")


def calculate_selection_metric(sensitivity, far, metric_type='f1', beta=2.0, min_sensitivity=None):
    """
    Calculate model selection metric for comparing model checkpoints.
    
    Args:
        sensitivity: True positive rate (recall)
        far: False positive rate (false alarm rate)
        metric_type: Type of metric to calculate:
            - 'f1': F1 score (harmonic mean of precision and recall)
            - 'f_beta': F-beta score (weighted harmonic mean, favors recall when beta > 1)
            - 'youden': Youden's J statistic (TPR - FPR, optimal balance point)
            - 'min_fpr': Minimize FPR given minimum TPR (use -far as metric, higher is better)
            - 'dual_improvement': Returns None (use legacy dual improvement logic)
        beta: Beta parameter for F-beta score (default 2.0 favors recall)
        min_sensitivity: Minimum acceptable sensitivity for 'min_fpr' metric.
                        If sensitivity < min_sensitivity, returns -inf (worst possible)
    
    Returns:
        Metric value (higher is better), or None for 'dual_improvement'
    """
    if metric_type == 'dual_improvement':
        return None
    
    elif metric_type == 'min_fpr':
        # Minimize FPR while maintaining minimum TPR
        # Returns negative FPR so that lower FPR = higher metric (better)
        if min_sensitivity is not None and sensitivity < min_sensitivity:
            return -float('inf')  # Reject models below min sensitivity
        return -far  # Negative FPR: lower FPR = higher score
    
    elif metric_type == 'youden':
        # Youden's J statistic: TPR - FPR (range -1 to 1, higher is better)
        return sensitivity - far
    
    elif metric_type == 'f1' or metric_type == 'f_beta':
        # Calculate F-beta score using sensitivity and specificity
        # For medical applications, we care about:
        # - Sensitivity (catching seizures) = TPR
        # - Not having too many false alarms = low FPR = high specificity
        
        specificity = 1 - far
        
        if metric_type == 'f1':
            beta_val = 1.0
        else:
            beta_val = beta
        
        # F-beta score using sensitivity and specificity
        # Higher beta weights sensitivity more
        beta_sq = beta_val ** 2
        if sensitivity == 0 and specificity == 0:
            return 0.0
        f_beta = (1 + beta_sq) * (sensitivity * specificity) / (beta_sq * specificity + sensitivity)
        return f_beta
    
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def visualize_pytorch_model(model, input_shape, model_name="model"):
    """
    Visualize PyTorch model architecture with key parameters.
    Creates both a text summary and optionally a graphical diagram.
    
    Args:
        model: PyTorch model to visualize
        input_shape: Shape of input data (excluding batch dimension)
        model_name: Name for saved diagram file
    """
    import torch
    
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*80)
    
    # Create detailed text summary
    total_params = 0
    trainable_params = 0
    
    print(f"\n{'Layer (type)':<40} {'Output Shape':<25} {'Param #':<15} {'Details':<20}")
    print("-" * 100)
    
    # Track dropout layers and their rates
    dropout_info = []
    
    for name, module in model.named_modules():
        if name == '':  # Skip the root module
            continue
            
        # Count parameters for this module
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            total_params += params
            trainable_params += sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        
        # Get module details
        details = ""
        if isinstance(module, torch.nn.Conv1d):
            details = f"kernel={module.kernel_size[0]}, stride={module.stride[0]}, filters={module.out_channels}"
        elif isinstance(module, torch.nn.Linear):
            details = f"{module.in_features} -> {module.out_features}"
        elif isinstance(module, torch.nn.Dropout):
            dropout_info.append((name, module.p))
            details = f"p={module.p}"
        elif isinstance(module, torch.nn.BatchNorm1d):
            details = f"features={module.num_features}"
        
        # Only print layers with parameters or dropout
        if params > 0 or isinstance(module, torch.nn.Dropout):
            # Estimate output shape (simplified)
            output_shape = "Multiple"
            print(f"{name:<40} {output_shape:<25} {params:<15,} {details:<20}")
    
    print("-" * 100)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    
    # Highlight dropout configuration
    if dropout_info:
        print(f"\n{'DROPOUT CONFIGURATION':<40}")
        print("-" * 60)
        for layer_name, dropout_rate in dropout_info:
            print(f"  {layer_name:<38} p={dropout_rate}")
    
    print("\n" + "="*80)
    
    # Try to use torchinfo for detailed summary
    try:
        from torchinfo import summary
        print("\nDetailed Model Summary (via torchinfo):")
        print("-" * 80)
        # Create sample input with batch dimension
        if len(input_shape) == 2:
            sample_input = (1, input_shape[0], input_shape[1])
        elif len(input_shape) == 1:
            sample_input = (1, 1, input_shape[0])
        else:
            sample_input = tuple([1] + list(input_shape))
        summary(model, input_size=sample_input, col_names=["output_size", "num_params", "kernel_size", "mult_adds"])
    except ImportError:
        print("\nNote: Install 'torchinfo' for more detailed model summaries:")
        print("  pip install torchinfo")
    except Exception as e:
        print(f"\nCould not generate torchinfo summary: {e}")
    
    # Try to create a visual diagram using torchviz
    try:
        from torchviz import make_dot
        import torch
        
        # Set model to train mode to avoid BatchNorm issues with batch_size=1
        was_training = model.training
        model.train()
        
        # Create dummy input with batch_size > 1 for BatchNorm compatibility
        batch_size = 4
        if len(input_shape) == 2:
            dummy_input = torch.randn(batch_size, input_shape[0], input_shape[1])
        elif len(input_shape) == 1:
            dummy_input = torch.randn(batch_size, 1, input_shape[0])
        else:
            dummy_input = torch.randn(tuple([batch_size] + list(input_shape)))
        
        dummy_input = dummy_input.to(next(model.parameters()).device)
        
        # Forward pass to create computation graph
        output = model(dummy_input)
        
        # Restore original training state
        if not was_training:
            model.eval()
        
        # Create visualization
        dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        diagram_path = f"{model_name}_architecture"
        dot.render(diagram_path, format='png', cleanup=True)
        print(f"\nModel architecture diagram saved to: {diagram_path}.png")
    except ImportError:
        print("\nNote: Install 'torchviz' and 'graphviz' for visual model diagrams:")
        print("  pip install torchviz")
        print("  Graphviz is already in requirements.txt")
    except Exception as e:
        print(f"\nCould not generate visual diagram: {e}")
    
    print("="*80 + "\n")


def trainModel_pytorch(configObj, dataDir='.', debug=False):
    ''' Create and train a new PyTorch neural network model.
    '''
    TAG = "nnTrainer.trainModel_pytorch()"
    print("%s" % (TAG))
    
    # Import PyTorch
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    # Load configuration parameters
    params = load_config_params(configObj)

    # Centralized deterministic seeding (single source: config["randomSeed"])
    # If null/None/missing -> non-deterministic (random) sampling.
    _seed = get_seed_from_config(configObj)
    seed_all(_seed, debug=debug)
    _torch_gen = make_torch_generator(_seed, debug=debug)
    if _seed is None:
        print(f"{TAG}: randomSeed is null/None -> non-deterministic (random) batch sampling")
    else:
        print(f"{TAG}: Using deterministic seed {_seed} for all RNGs (torch, numpy, python, cudnn)")

    # Load model class (weights init will be deterministic if seed is set)
    nnModel = load_model_class(params['nnModelClassName'], configObj, framework='pytorch')
    
    # Resolve data file paths
    trainCsvPath, valCsvPath = resolve_data_file_paths(
        dataDir, params['trainAugCsvFname'], params['valCsvFname'], configObj, TAG
    )
    
    # Load and preprocess data
    xTrain, yTrain, xVal, yVal, nClasses, train_df_used = load_and_preprocess_data(
        trainCsvPath, valCsvPath, nnModel, params['inputDims'], debug, TAG, return_train_df=True
    )
    
    # Model filename
    modelFname = f"{params['modelFnameRoot']}.pt"
    modelFnamePath = os.path.join(dataDir, modelFname)
    
    # Create or load model
    if os.path.exists(modelFnamePath):
        print(f"Model {modelFnamePath} already exists - loading existing model as starting point for training")
        model = nnModel.makeModel(input_shape=xTrain.shape[1:], num_classes=nClasses, nLayers=params['nLayers'])
        device = nnModel.device
        
        # Load checkpoint
        # PyTorch 2.6 defaults weights_only=True; set explicitly to allow legacy checkpoints
        checkpoint = torch.load(modelFnamePath, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model weights from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Older format or direct state dict
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {modelFnamePath}")
    else:
        print("Creating new PyTorch Model")
        model = nnModel.makeModel(input_shape=xTrain.shape[1:], num_classes=nClasses, nLayers=params['nLayers'])
        device = nnModel.device
    
    # Visualize model architecture (equivalent to keras.utils.plot_model)
    visualize_pytorch_model(model, xTrain.shape[1:], model_name=os.path.join(dataDir, params['modelFnameRoot']))
    
    # Convert numpy arrays to PyTorch tensors
    # Phase 1: xTrain/xVal are already float32 (from load_and_preprocess_data),
    # so from_numpy shares memory without copy; avoid .float() duplicate.
    # yTrain/yVal are int32, need long for CrossEntropyLoss.
    if xTrain.dtype == np.float32:
        xTrain_tensor = torch.from_numpy(xTrain)
    else:
        xTrain_tensor = torch.from_numpy(xTrain.astype(np.float32))
    yTrain_tensor = torch.from_numpy(yTrain.astype(np.int64) if yTrain.dtype != np.int64 else yTrain).long()
    if xVal.dtype == np.float32:
        xVal_tensor = torch.from_numpy(xVal)
    else:
        xVal_tensor = torch.from_numpy(xVal.astype(np.float32))
    yVal_tensor = torch.from_numpy(yVal.astype(np.int64) if yVal.dtype != np.int64 else yVal).long()
    # Optionally free numpy arrays now that tensors share/own the data;
    # keep them for later shape checks but allow GC of Python list remnants
    gc.collect()
    _log_mem(f"{TAG} after tensor conversion", f"xTrain_tensor={tuple(xTrain_tensor.shape)} dtype={xTrain_tensor.dtype}")
    
    # Create data loaders
    train_dataset = TensorDataset(xTrain_tensor, yTrain_tensor)
    val_dataset = TensorDataset(xVal_tensor, yVal_tensor)
    
    # Build train loader: subtype-aware sampling -> class-balanced sampling -> standard shuffle.
    use_balanced_batches = params.get('use_balanced_batches', False)
    use_subtype_weighting = params.get('use_subtype_weighting', False) and len(params.get('subtype_weights', {})) > 0
    train_loader = None

    if use_subtype_weighting and create_subtype_weighted_sampler is not None:
        # Phase 1: avoid extra copy unless we need to add subType column
        # (subtype lives inside typeStr as 'Type/SubType' in the feature CSVs)
        train_df_for_sampling = ensure_subtype_column(train_df_used)

        if train_df_for_sampling is not None and 'eventId' in train_df_for_sampling.columns and 'subType' in train_df_for_sampling.columns:
            print(f"{TAG}: Using subtype-aware weighted sampling")
            print(f"{TAG}: Subtype weights: {params['subtype_weights']}")
            sampler = create_subtype_weighted_sampler(
                df=train_df_for_sampling,
                y_values=yTrain,
                subtype_weights=params['subtype_weights'],
                debug=debug,
                seed=_seed,
                generator=_torch_gen,
            )
            # Pass generator to DataLoader for deterministic shuffling/sampling
            if _torch_gen is not None:
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler, drop_last=True, generator=_torch_gen)
            else:
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler, drop_last=True)
            print(f"{TAG}: Subtype-aware sampling enabled (seed={_seed})")
        else:
            print(f"{TAG}: WARNING - subtype weighting requested but eventId/subType columns are unavailable; falling back")

    if train_loader is None and use_subtype_weighting and create_subtype_weighted_sampler is None:
        print(f"{TAG}: WARNING - subtype weighting requested but subtype_weighting module is unavailable; falling back")

    if train_loader is None and use_balanced_batches and params['use_lr_schedule']:
        print(f"{TAG}: Using balanced batch sampling (Spahr et al. 2025 approach)")
        # Calculate sample weights for balanced sampling
        class_counts = torch.bincount(yTrain_tensor)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[yTrain_tensor]
        
        # Create weighted random sampler for balanced batches (deterministic if seed set)
        from torch.utils.data import WeightedRandomSampler
        try:
            if _torch_gen is not None:
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,  # Allow oversampling with replacement
                    generator=_torch_gen
                )
            else:
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
        except TypeError:
            # Older torch without generator arg
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        if _torch_gen is not None:
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler, drop_last=True, generator=_torch_gen)
        else:
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler, drop_last=True)
        print(f"{TAG}: Class distribution in training data: {class_counts.tolist()}")
        print(f"{TAG}: Each batch will be approximately balanced between classes (seed={_seed})")

    if train_loader is None:
        if _torch_gen is not None:
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True, generator=_torch_gen)
        else:
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # Setup optimizer - use AdamW if configured
    if params['use_adamw']:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params['lrStart'],
            betas=(params['adamw_beta1'], params['adamw_beta2']),
            weight_decay=params['weight_decay']
        )
        print(f"{TAG}: Using AdamW optimizer with betas=({params['adamw_beta1']}, {params['adamw_beta2']}), weight_decay={params['weight_decay']}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=params['lrStart'])
        print(f"{TAG}: Using Adam optimizer")
    
    # Setup loss function with optional class weighting
    if 'pos_weight_multiplier' in params and params['pos_weight_multiplier'] is not None:
        # Weight the positive (seizure) class lower to penalize false positives more
        pos_weight = params['pos_weight_multiplier']
        neg_weight = 1.0 / pos_weight if pos_weight > 0 else 1.0
        class_weights = torch.tensor([neg_weight, pos_weight], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"{TAG}: Using weighted CrossEntropyLoss - class weights: negative={neg_weight:.2f}, positive={pos_weight:.2f}")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"{TAG}: Using standard CrossEntropyLoss")
    
    # Setup learning rate scheduler
    if params['use_lr_schedule']:
        # Three-phase learning rate schedule (Spahr et al. 2025)
        print(f"{TAG}: ===== Three-Phase LR Schedule Configuration =====")
        print(f"{TAG}:   lrStart (initial): {params['lrStart']}")
        print(f"{TAG}:   lrPeak (max): {params['lr_peak']}")
        print(f"{TAG}:   lrMainEnd (cosine end): {params['lr_main_end']}")
        print(f"{TAG}:   warmupSteps: {params['warmup_steps']}")
        print(f"{TAG}:   mainSteps: {params['main_steps']}")
        print(f"{TAG}:   cooldownSteps: {params['cooldown_steps']}")
        print(f"{TAG}:   totalTrainingSteps: {params.get('total_training_steps', 'N/A')}")
        print(f"{TAG}: ==============================================")
        
        def get_three_phase_lr(step):
            """
            Three-phase learning rate schedule:
            1. Warmup: linear increase from lrStart to lr_peak
            2. Cosine annealing: from lr_peak to lr_main_end
            3. Cooldown: linear decrease from lr_main_end to 0
            """
            warmup = params['warmup_steps']
            main = params['main_steps']
            cooldown = params['cooldown_steps']
            lr_start = params['lrStart']
            lr_peak = params['lr_peak']
            lr_main_end = params['lr_main_end']
            
            if step < warmup:
                # Warmup phase: linear increase
                progress = step / warmup
                lr = lr_start + progress * (lr_peak - lr_start)
            elif step < warmup + main:
                # Cosine annealing phase
                progress = (step - warmup) / main
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                lr = lr_main_end + (lr_peak - lr_main_end) * cosine_factor
            else:
                # Cooldown phase: linear decrease to 0
                progress = (step - warmup - main) / cooldown
                lr = lr_main_end * (1 - progress)
            
            return lr / lr_start  # LambdaLR expects a multiplier relative to initial LR
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_three_phase_lr)
        print(f"{TAG}: Using three-phase LR schedule: warmup={params['warmup_steps']}, main={params['main_steps']}, cooldown={params['cooldown_steps']}")
        print(f"{TAG}: LR progression: {params['lrStart']} -> {params['lr_peak']} -> {params['lr_main_end']} -> 0")
    else:
        # Original ReduceLROnPlateau scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=params['lrFactor'], patience=params['lrPatience'], min_lr=params['lrMin']
        )
        print(f"{TAG}: Using ReduceLROnPlateau scheduler")
    
    # Training history
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'lr': [],
        'sensitivity': [],
        'far': [],
        'youden': []
    }
    epoch_metrics_rows = []
    
    best_val_loss = float('inf')
    best_sensitivity = 0.0
    best_far = float('inf')
    best_metric = -float('inf')  # Initialize best metric for model selection
    best_metric_fallback = -float('inf')  # Track best model if criteria never met
    criteria_met_once = False  # Track if we've ever met the threshold criteria
    patience_counter = 0
    global_step = 0
    
    # Determine training mode: step-based or epoch-based
    if params['use_lr_schedule']:
        max_steps = params['total_training_steps']
        print(f"{TAG}: Starting step-based training for {max_steps} steps")
        use_step_based = True
    else:
        max_epochs = params['epochs']
        print(f"{TAG}: Starting epoch-based training for {max_epochs} epochs")
        use_step_based = False
    
    # Training loop
    epoch = 0
    training_complete = False
    
    while not training_complete:
        epoch += 1
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Update learning rate for step-based training
            if use_step_based:
                scheduler.step()
                global_step += 1
            
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Check if we've reached max steps
            if use_step_based and global_step >= max_steps:
                training_complete = True
                break
        
        train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                # Collect predictions for sensitivity/FAR calculation
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total
        
        # Calculate sensitivity and FAR for advanced checkpoint logic
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        # Sensitivity (TPR): TP / (TP + FN)
        true_positives = np.sum((val_predictions == 1) & (val_targets == 1))
        false_negatives = np.sum((val_predictions == 0) & (val_targets == 1))
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # FAR (FPR): FP / (FP + TN)
        false_positives = np.sum((val_predictions == 1) & (val_targets == 0))
        true_negatives = np.sum((val_predictions == 0) & (val_targets == 0))
        far = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0

        # Youden's J statistic: TPR - FPR
        youden = sensitivity - far
        
        # Update learning rate for epoch-based training
        if not use_step_based:
            scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['lr'].append(current_lr)
        history['sensitivity'].append(sensitivity)
        history['far'].append(far)
        history['youden'].append(youden)
        
        # Print progress
        if params['trainingVerbosity'] > 0:
            if use_step_based:
                print(f"Step {global_step}/{max_steps} (Epoch {epoch}) - "
                      f"loss: {train_loss:.3f} - acc: {train_accuracy:.3f} - "
                      f"val_loss: {val_loss:.3f} - val_acc: {val_accuracy:.3f} - "
                      f"TPR: {sensitivity:.3f} - FAR: {far:.3f} - Youden: {youden:.3f} - lr: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch}/{max_epochs} - "
                      f"loss: {train_loss:.3f} - acc: {train_accuracy:.3f} - "
                      f"val_loss: {val_loss:.3f} - val_acc: {val_accuracy:.3f} - "
                      f"TPR: {sensitivity:.3f} - FAR: {far:.3f} - Youden: {youden:.3f} - lr: {current_lr:.2e}")
        
        # Save best model with advanced checkpoint logic (Spahr et al. 2025)
        should_save = False
        save_reason = ""
        rejection_reason = ""
        row = {
            'epoch': epoch,
            'global_step': global_step,
            'loss': train_loss,
            'accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'lr': current_lr,
            'sensitivity': sensitivity,
            'far': far,
            'youden': youden,
            'checkpoint_saved': False,
            'saved_epoch': '',
            'save_reason': '',
            'rejection_reason': ''
        }
        
        # First check: enforce maximum FPR threshold if configured
        exceeds_max_fpr = False
        if params['save_best_max_fpr'] is not None and far > params['save_best_max_fpr']:
            exceeds_max_fpr = True
            rejection_reason = f"FPR={far:.3f} exceeds max threshold={params['save_best_max_fpr']:.3f}"
        
        # Second check: enforce minimum sensitivity
        below_min_sensitivity = False
        if sensitivity < params['save_best_min_sensitivity']:
            below_min_sensitivity = True
            if rejection_reason:
                rejection_reason += f", TPR={sensitivity:.3f} below min={params['save_best_min_sensitivity']:.3f}"
            else:
                rejection_reason = f"TPR={sensitivity:.3f} below min={params['save_best_min_sensitivity']:.3f}"
        
        # Only evaluate saving criteria if basic thresholds are met
        meets_criteria = not exceeds_max_fpr and not below_min_sensitivity
        
        if meets_criteria:
            criteria_met_once = True  # Mark that we've found at least one good model
            metric_type = params['model_selection_metric']
            
            if metric_type == 'dual_improvement':
                # Spahr et al. checkpoint logic: save if both sensitivity and FAR improve,
                # OR if FAR reduces >10% while sensitivity stays within tolerance
                both_improved = (sensitivity > best_sensitivity) and (far < best_far)
                far_reduction = (best_far - far) / best_far if best_far > 0 else 0
                sensitivity_tolerance = abs(sensitivity - best_sensitivity)
                
                if both_improved:
                    should_save = True
                    save_reason = f"both improved (TPR: {best_sensitivity:.3f}→{sensitivity:.3f}, FAR: {best_far:.3f}→{far:.3f}, Youden: {best_sensitivity - best_far:.3f}→{youden:.3f})"
                elif far_reduction > params['save_best_on_far_reduction'] and \
                    sensitivity_tolerance <= params['save_best_on_sensitivity_tolerance']:
                    should_save = True
                    save_reason = f"FAR reduced by {far_reduction*100:.1f}% (TPR within {params['save_best_on_sensitivity_tolerance']*100:.0f}% tolerance, Youden {youden:.3f})"
            
            else:
                # Use metric-based selection (F1, F-beta, Youden's J, min_fpr)
                current_metric = calculate_selection_metric(
                    sensitivity, far, metric_type, params['f_beta'],
                    min_sensitivity=params['save_best_min_sensitivity']
                )
                
                if current_metric > best_metric:
                    should_save = True
                    save_reason = f"{metric_type}={current_metric:.3f} improved (was {best_metric:.3f}, TPR={sensitivity:.3f}, FAR={far:.3f}, Youden={youden:.3f})"
                    best_metric = current_metric
        
        # Fallback: if criteria have never been met, save best model found so far
        elif not criteria_met_once:
            metric_type = params['model_selection_metric']
            
            if metric_type == 'dual_improvement':
                # Use dual improvement logic as fallback
                both_improved = (sensitivity > best_sensitivity) and (far < best_far)
                if both_improved:
                    should_save = True
                    save_reason = f"FALLBACK: both improved (TPR: {best_sensitivity:.3f}→{sensitivity:.3f}, FAR: {best_far:.3f}→{far:.3f}, Youden: {best_sensitivity - best_far:.3f}→{youden:.3f}) [criteria not met yet]"
            else:
                # Use metric-based selection as fallback
                current_metric = calculate_selection_metric(
                    sensitivity, far, metric_type, params['f_beta'],
                    min_sensitivity=None  # Ignore minimum sensitivity for fallback
                )
                
                if current_metric > best_metric_fallback:
                    should_save = True
                    save_reason = f"FALLBACK: {metric_type}={current_metric:.3f} improved (was {best_metric_fallback:.3f}, TPR={sensitivity:.3f}, FAR={far:.3f}, Youden={youden:.3f}) [criteria not met yet]"
                    best_metric_fallback = current_metric
        
        # Fallback: also save on best validation loss if no other criteria used (epoch-based training)
        if not should_save and not params['use_lr_schedule'] and meets_criteria:
            if val_loss < best_val_loss:
                should_save = True
                save_reason = "validation loss improved"
        
        if should_save:
            best_val_loss = val_loss
            best_sensitivity = sensitivity
            best_far = far
            patience_counter = 0
            print(f"{TAG}: ✓ Saving checkpoint to {modelFnamePath}")
            print(f"{TAG}:   Reason: {save_reason}")
            row['checkpoint_saved'] = True
            row['saved_epoch'] = epoch
            row['save_reason'] = save_reason
            
            # Extract dropout values from model if available (for .ptl conversion)
            conv_dropout = getattr(model, 'conv_dropout', 0.0)
            dense_dropout = getattr(model, 'dense_dropout', 0.025)
            
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'sensitivity': sensitivity,
                'far': far,
                'conv_dropout': conv_dropout,
                'dense_dropout': dense_dropout,
                'config': configObj
            }, modelFnamePath)
        else:
            row['rejection_reason'] = rejection_reason
            patience_counter += 1
            if rejection_reason:
                if params['trainingVerbosity'] > 0 and epoch % 10 == 0:  # Log every 10 epochs to avoid spam
                    print(f"{TAG}: ✗ Not saving: {rejection_reason}")

        epoch_metrics_rows.append(row)
        
        # Early stopping (only for epoch-based training)
        if not use_step_based and params['earlyStoppingPatience'] is not None:
            if patience_counter >= params['earlyStoppingPatience']:
                print(f"{TAG}: Early stopping triggered after {epoch} epochs")
                training_complete = True
        
        # Check termination conditions
        if not use_step_based and epoch >= max_epochs:
            training_complete = True
    
    print(f"{TAG}: Trained using {np.count_nonzero(yTrain == 1)} seizure datapoints and {np.count_nonzero(yTrain == 0)} false alarm datapoints")

    # Plot training history
    plot_training_history(history, params['modelFnameRoot'], dataDir, framework='pytorch')
    
    # Save training history as JSON for later reference
    history_json_path = os.path.join(dataDir, "training_history.json")
    try:
        # Convert numpy arrays to lists for JSON serialization
        history_for_json = {}
        for key, value in history.items():
            if isinstance(value, np.ndarray):
                history_for_json[key] = value.tolist()
            else:
                history_for_json[key] = value
        
        with open(history_json_path, 'w') as f:
            json.dump(history_for_json, f, indent=2)
        print(f"{TAG}: Saved training history to {history_json_path}")
    except Exception as e:
        print(f"{TAG}: Warning - could not save training history JSON: {e}")

    # Persist per-epoch metrics for easier threshold tuning and checkpoint review
    metrics_csv_path = os.path.join(dataDir, "epoch_metrics.csv")
    try:
        fieldnames = [
            'epoch', 'global_step', 'loss', 'accuracy', 'val_loss', 'val_accuracy',
            'lr', 'sensitivity', 'far', 'youden', 'checkpoint_saved', 'saved_epoch', 'save_reason', 'rejection_reason'
        ]
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in epoch_metrics_rows:
                csv_row = {key: row.get(key, '') for key in fieldnames}
                if csv_row['saved_epoch'] == '':
                    csv_row['saved_epoch'] = ''
                writer.writerow(csv_row)
        print(f"{TAG}: Saved epoch-level metrics to {metrics_csv_path}")
    except Exception as e:
        print(f"{TAG}: Warning - could not save epoch metrics CSV: {e}")

    # Convert .pt model to .ptl (PyTorch Lite) format for mobile deployment
    if (False):
        print(f"{TAG}: Converting model to .ptl format...")
        ptl_model_path = modelFnamePath.replace('.pt', '.ptl')
        try:
            # Import convertPt2Ptl function
            try:
                from user_tools.nnTraining2.convertPt2Ptl import convert_pt_to_ptl
            except ImportError:
                from convertPt2Ptl import convert_pt_to_ptl
            
            # Get input shape from training data
            # xTrain shape is (n_samples, length, 1) for 1D input
            # PTL inference expects (batch, channels, length) after permutation in predict_model()
            # So we need to trace with shape (1, channels, length) = (1, 1, length)
            length = xTrain.shape[1]
            channels = xTrain.shape[2] if len(xTrain.shape) > 2 else 1
            input_shape = (1, channels, length)  # (batch, channels, length)
            
            # Convert to .ptl
            success = convert_pt_to_ptl(
                input_path=modelFnamePath,
                output_path=ptl_model_path,
                input_shape=input_shape,
                num_classes=nClasses,
                verbose=True
            )
            
            if success:
                print(f"{TAG}: Successfully converted model to {ptl_model_path}")
            else:
                print(f"{TAG}: Warning - Failed to convert model to .ptl format")
        except Exception as e:
            print(f"{TAG}: Warning - Could not convert model to .ptl format: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"{TAG}: Skipping conversion to .ptl format")

    # Convert .pt model to .pte (ExecuTorch) format for edge deployment
    print(f"{TAG}: Converting model to .pte format...")
    pte_model_path = modelFnamePath.replace('.pt', '.pte')
    try:
        # Import convertPt2Pte function
        try:
            from user_tools.nnTraining2.convertPt2Pte import convert_pt_to_pte
        except ImportError:
            from convertPt2Pte import convert_pt_to_pte
        
        # Get input shape from training data
        # xTrain shape is (n_samples, length, 1) for 1D input
        # PTE inference expects (batch, channels, length) after permutation in predict_model()
        # So we need to trace with shape (1, channels, length) = (1, 1, length)
        length = xTrain.shape[1]
        channels = xTrain.shape[2] if len(xTrain.shape) > 2 else 1
        input_shape = (1, channels, length)  # (batch, channels, length)
        
        # Convert to .pte
        success = convert_pt_to_pte(
            input_path=modelFnamePath,
            output_path=pte_model_path,
            input_shape=input_shape,
            num_classes=nClasses,
            verbose=True
        )
        
        if success:
            print(f"{TAG}: Successfully converted model to {pte_model_path}")
        else:
            print(f"{TAG}: Warning - Failed to convert model to .pte format")
    except Exception as e:
        print(f"{TAG}: Warning - Could not convert model to .pte format: {e}")
        import traceback
        traceback.print_exc()

    print(f"{TAG}: Training Complete")



def main():
    print("nnTrainer_csv.main()")
    parser = argparse.ArgumentParser(description='Seizure Detection Neural Network Trainer')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing test configuration')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    parser.add_argument('--test', action="store_true",
                        help='Test existing model, do not re-train.')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)



    configObj = libosd.configUtils.loadConfig(args['config'])
    print("configObj=",configObj)
    # Load a separate OSDB Configuration file if it is included.
    if ("osdbCfg" in configObj):
        osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg",configObj)
        print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
        osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
        # Merge the contents of the OSDB Configuration file into configObj
        configObj = configObj | osdbCfgObj

    print("configObj=",configObj.keys())

    debug = configObj['debug']
    if args['debug']: debug=True

    if not args['test']:
        trainModel(configObj, dataDir='.', debug=debug)
        nnTester.testModel(configObj, debug, test_ptl=True, test_pte=True)
    else:
        nnTester.testModel(configObj, debug, test_ptl=True, test_pte=True)
        
    


if __name__ == "__main__":
    main()
