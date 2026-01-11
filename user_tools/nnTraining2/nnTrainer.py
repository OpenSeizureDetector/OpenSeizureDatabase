#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import json
import importlib
#from tkinter import Y
import sklearn.metrics
import imblearn.over_sampling
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.dpTools
import libosd.osdAlgTools
import libosd.configUtils

try:
    from user_tools.nnTraining2 import augmentData
except ImportError:
    import augmentData
import nnTester


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




def df2trainingData(df, nnModel, debug=False):
    ''' Converts a pandas dataframe df into a list of data and a list of associated seizure classes
    for use by model nnModel.
    This works by taking each row in the dataframe and converting it into a dict with the same
    values as an OpenSeizureDetector datapoint.   It then calls the dp2vector method of the specified
    model to pre-process the data into the format required by the model.

    FIXME:  It uses a simple for loop to loop through the dataframe - there is probably a quicker
    way of applying a function to each row in the dataframe in turn.

    FIXME: This only works on acceleration magnitude values at the moment - add an option to use 3d data.
    '''

    # Detect accelerometer magnitude columns dynamically (M000_t-0..Mxxx_t-0 or M000..Mxxx).
    # Supports both with and without feature history suffix (_t-0).
    # This supports different epoch lengths (e.g. 125 samples for 5s, 750 samples for 30s).
    cols = list(df.columns)
    # Try with _t-0 suffix first (feature history enabled)
    m_cols = [c for c in cols if isinstance(c, str) and c.startswith('M') and c.endswith('_t-0')]
    if len(m_cols) == 0:
        # Try without suffix (feature history disabled, addFeatureHistoryLength=0)
        m_cols = [c for c in cols if isinstance(c, str) and c.startswith('M') and len(c) == 4 and c[1:].isdigit()]
    if len(m_cols) == 0:
        print("cols are: ", [c for c in cols])
        raise ValueError("df2trainingData: No magnitude (Mxxx_t-0 or Mxxx) columns found in dataframe")
    # Find start/end indices of the M columns in the dataframe
    m_indices = [cols.index(c) for c in m_cols]
    accStartCol = min(m_indices)  # index of first Mxxx column
    accEndCol = max(m_indices) + 1  # exclusive end index

    # Other columns
    try:
        hrCol = df.columns.get_loc('hr')
    except Exception:
        hrCol = None
    typeCol = df.columns.get_loc('type')
    eventIdCol = df.columns.get_loc('eventId')

    outLst = []
    classLst = []
    lastEventId = None
    print("Processing Events:")
    for n in range(0,len(df)):
        dpDict = {}
        if (debug): print("n=%d" % n)
        rowArr = df.iloc[n]
        if (debug): print("rowArrLen=%d" % len(rowArr), type(rowArr), rowArr)

        eventId = rowArr.iloc[eventIdCol]
        if (eventId != lastEventId):
            sys.stdout.write("%d/%d (%.1f %%) : %s\r" % (n,len(df),100.*n/len(df), eventId))
            # Reset accumulation buffer when moving to a new event
            nnModel.resetAccBuf()
            lastEventId = eventId

        accArr = rowArr.iloc[accStartCol:accEndCol].values.astype(float).tolist()
        if (debug): print("accArr=", accArr, type(accArr))
        dpDict['rawData'] = accArr
        # HR may be missing in feature CSVs; handle missing hr gracefully
        if hrCol is not None:
            try:
                dpDict['hr'] = int(rowArr.iloc[hrCol])
            except Exception:
                dpDict['hr'] = None
        else:
            dpDict['hr'] = None
        if (debug): print("dpDict=",dpDict)
        dpInputData = nnModel.dp2vector(dpDict, normalise=False)
        if (dpInputData is not None):
            outLst.append(dpInputData)
            classLst.append(rowArr.iloc[typeCol])
        dpDict = None
        dpInputData = None
    print(".")
    return(outLst, classLst)


def load_config_params(configObj):
    """Extract all training configuration parameters from configObj.
    
    Returns:
        dict: Dictionary containing all configuration parameters
    """
    params = {}
    
    # Data file names
    params['trainAugCsvFname'] = libosd.configUtils.getConfigParam('trainFeaturesHistoryFileCsv', configObj['dataFileNames'])
    if not isinstance(params['trainAugCsvFname'], str):
        params['trainAugCsvFname'] = None
    params['valCsvFname'] = libosd.configUtils.getConfigParam('valDataFileCsv', configObj['dataFileNames'])
    if not isinstance(params['valCsvFname'], str):
        params['valCsvFname'] = None
    params['testCsvFname'] = libosd.configUtils.getConfigParam("testFeaturesHistoryFileCsv", configObj['dataFileNames'])
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
    
    # Data processing
    params['validationProp'] = libosd.configUtils.getConfigParam("validationProp", configObj['dataProcessing'])
    params['inputDims'] = libosd.configUtils.getConfigParam("dims", configObj)
    if params['inputDims'] is None:
        params['inputDims'] = 1
    
    # Handle validation data fallback
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
        testFeaturesName = configObj['dataFileNames'].get('testFeaturesFileCsv')
        # Ensure these are strings, not booleans
        if not isinstance(trainFeaturesName, str):
            trainFeaturesName = None
        if not isinstance(testFeaturesName, str):
            testFeaturesName = None
    except Exception:
        trainFeaturesName = None
        testFeaturesName = None

    if trainFeaturesName is not None:
        candidate = os.path.join(dataDir, trainFeaturesName)
        if os.path.exists(candidate):
            print(f"{TAG}: Using train features CSV {candidate}")
            trainAugCsvFnamePath = candidate
    if testFeaturesName is not None:
        candidate = os.path.join(dataDir, testFeaturesName)
        if os.path.exists(candidate):
            print(f"{TAG}: Using validation/test features CSV {candidate}")
            valCsvFnamePath = candidate
    
    if trainAugCsvFnamePath is None:
        raise ValueError(f"{TAG}: No training data file specified or found")
    if valCsvFnamePath is None:
        raise ValueError(f"{TAG}: No validation data file specified or found")
    
    return trainAugCsvFnamePath, valCsvFnamePath


def load_and_preprocess_data(trainCsvPath, valCsvPath, nnModel, inputDims, debug, TAG):
    """Load and preprocess training and validation data.
    
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
    # Load training data
    print(f"{TAG}: Loading training data from file {trainCsvPath}")
    if not os.path.exists(trainCsvPath):
        print(f"ERROR: File {trainCsvPath} does not exist")
        exit(-1)

    df = augmentData.loadCsv(trainCsvPath, debug=debug)
    print(f"{TAG}: Loaded {len(df)} training datapoints")

    print(f"{TAG}: Re-formatting training data")
    xTrain, yTrain = df2trainingData(df, nnModel)

    print(f"{TAG}: Converting to np arrays")
    try:
        xTrain = np.array(xTrain)
    except ValueError as e:
        print("Failed simple array conversion - trying concatenate...")
        xTrain = np.concatenate(xTrain)
    yTrain = np.array(yTrain)

    print(f"xTrain.shape={xTrain.shape}, yTrain.shape={yTrain.shape}")
    print(f"{TAG}: re-shaping array for training")

    if inputDims == 1:
        xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    elif inputDims == 2:
        xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1))
    else:
        print(f"ERROR - inputDims out of Range: {inputDims}")
        exit(-1)

    # Load validation data
    print(f"{TAG}: Loading validation data from file {valCsvPath}")
    if not os.path.exists(valCsvPath):
        print(f"ERROR: File {valCsvPath} does not exist")
        exit(-1)

    df = augmentData.loadCsv(valCsvPath, debug=debug)
    print(f"{TAG}: Loaded {len(df)} validation datapoints")

    print(f"{TAG}: Re-formatting validation data")
    xVal, yVal = df2trainingData(df, nnModel)

    print(f"{TAG}: Converting to np arrays")
    try:
        xVal = np.array(xVal)
    except ValueError as e:
        print("Failed simple array conversion - trying concatenate...")
        xVal = np.concatenate(xVal)
    yVal = np.array(yVal)

    print(f"xVal.shape={xVal.shape}, yVal.shape={yVal.shape}")
    print(f"{TAG}: re-shaping array for validation")

    if inputDims == 1:
        xVal = xVal.reshape((xVal.shape[0], xVal.shape[1], 1))
    elif inputDims == 2:
        xVal = xVal.reshape((xVal.shape[0], xVal.shape[1], xVal.shape[2], 1))
    else:
        print(f"ERROR - inputDims out of Range: {inputDims}")
        exit(-1)

    nClasses = len(np.unique(yTrain))
    print(f"nClasses={nClasses}")
    print(f"Training using {np.count_nonzero(yTrain == 1)} seizure datapoints and {np.count_nonzero(yTrain == 0)} false alarm datapoints")

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
    plt.savefig(os.path.join(dataDir, f"{modelFnameRoot}_training2.png"))
    plt.close()

    # Plot 3: Sensitivity (TPR) and FAR (FPR) over validation checkpoints (PyTorch only)
    if sensitivity is not None and far is not None and sensitivity.size > 0 and far.size > 0:
        plt.figure()
        plt.plot(sensitivity, "b-", label="sensitivity (TPR)")
        plt.plot(far, "m-", label="FAR (FPR)")
        plt.title("Validation TPR/FPR over training")
        plt.ylabel("Rate", fontsize="large")
        plt.xlabel("validation checkpoints", fontsize="large")
        plt.ylim(0, 1)
        plt.legend(loc="best")
        plt.savefig(os.path.join(dataDir, f"{modelFnameRoot}_training_tpr_fpr.png"))
        plt.close()


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


def calculate_selection_metric(sensitivity, far, metric_type='f1', beta=2.0):
    """
    Calculate model selection metric for comparing model checkpoints.
    
    Args:
        sensitivity: True positive rate (recall)
        far: False positive rate (false alarm rate)
        metric_type: Type of metric to calculate:
            - 'f1': F1 score (harmonic mean of precision and recall)
            - 'f_beta': F-beta score (weighted harmonic mean, favors recall when beta > 1)
            - 'youden': Youden's J statistic (TPR - FPR, optimal balance point)
            - 'dual_improvement': Returns None (use legacy dual improvement logic)
        beta: Beta parameter for F-beta score (default 2.0 favors recall)
    
    Returns:
        Metric value (higher is better), or None for 'dual_improvement'
    """
    if metric_type == 'dual_improvement':
        return None
    
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
    
    # Load model class
    nnModel = load_model_class(params['nnModelClassName'], configObj, framework='pytorch')
    
    # Resolve data file paths
    trainCsvPath, valCsvPath = resolve_data_file_paths(
        dataDir, params['trainAugCsvFname'], params['valCsvFname'], configObj, TAG
    )
    
    # Load and preprocess data
    xTrain, yTrain, xVal, yVal, nClasses = load_and_preprocess_data(
        trainCsvPath, valCsvPath, nnModel, params['inputDims'], debug, TAG
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
    xTrain_tensor = torch.from_numpy(xTrain).float()
    yTrain_tensor = torch.from_numpy(yTrain).long()
    xVal_tensor = torch.from_numpy(xVal).float()
    yVal_tensor = torch.from_numpy(yVal).long()
    
    # Create data loaders
    train_dataset = TensorDataset(xTrain_tensor, yTrain_tensor)
    val_dataset = TensorDataset(xVal_tensor, yVal_tensor)
    
    # Use balanced batch sampling if configured (Spahr et al. 2025)
    # This oversamples minority class to create balanced batches
    use_balanced_batches = params.get('use_balanced_batches', False)
    if use_balanced_batches and params['use_lr_schedule']:
        print(f"{TAG}: Using balanced batch sampling (Spahr et al. 2025 approach)")
        # Calculate sample weights for balanced sampling
        class_counts = torch.bincount(yTrain_tensor)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[yTrain_tensor]
        
        # Create weighted random sampler for balanced batches
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Allow oversampling with replacement
        )
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler)
        print(f"{TAG}: Class distribution in training data: {class_counts.tolist()}")
        print(f"{TAG}: Each batch will be approximately balanced between classes")
    else:
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    
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
    
    criterion = nn.CrossEntropyLoss()
    
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
        'far': []
    }
    
    best_val_loss = float('inf')
    best_sensitivity = 0.0
    best_far = float('inf')
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
        
        # Print progress
        if params['trainingVerbosity'] > 0:
            if use_step_based:
                print(f"Step {global_step}/{max_steps} (Epoch {epoch}) - "
                      f"loss: {train_loss:.4f} - acc: {train_accuracy:.4f} - "
                      f"val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f} - "
                      f"sensitivity: {sensitivity:.4f} - FAR: {far:.4f} - lr: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch}/{max_epochs} - "
                      f"loss: {train_loss:.4f} - acc: {train_accuracy:.4f} - "
                      f"val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f} - lr: {current_lr:.2e}")
        
        # Save best model with advanced checkpoint logic (Spahr et al. 2025)
        should_save = False
        save_reason = ""
        rejection_reason = ""
        
        # First check: enforce maximum FPR threshold if configured
        exceeds_max_fpr = False
        if params['save_best_max_fpr'] is not None and far > params['save_best_max_fpr']:
            exceeds_max_fpr = True
            rejection_reason = f"FPR={far:.4f} exceeds max threshold={params['save_best_max_fpr']}"
        
        # Second check: enforce minimum sensitivity
        below_min_sensitivity = False
        if sensitivity < params['save_best_min_sensitivity']:
            below_min_sensitivity = True
            if rejection_reason:
                rejection_reason += f", sensitivity={sensitivity:.4f} below min={params['save_best_min_sensitivity']}"
            else:
                rejection_reason = f"sensitivity={sensitivity:.4f} below min={params['save_best_min_sensitivity']}"
        
        # Only evaluate saving criteria if basic thresholds are met
        if not exceeds_max_fpr and not below_min_sensitivity:
            metric_type = params['model_selection_metric']
            
            if metric_type == 'dual_improvement':
                # Spahr et al. checkpoint logic: save if both sensitivity and FAR improve,
                # OR if FAR reduces >10% while sensitivity stays within tolerance
                both_improved = (sensitivity > best_sensitivity) and (far < best_far)
                far_reduction = (best_far - far) / best_far if best_far > 0 else 0
                sensitivity_tolerance = abs(sensitivity - best_sensitivity)
                
                if both_improved:
                    should_save = True
                    save_reason = f"both improved (sens: {best_sensitivity:.4f}→{sensitivity:.4f}, FAR: {best_far:.4f}→{far:.4f})"
                elif far_reduction > params['save_best_on_far_reduction'] and \
                    sensitivity_tolerance <= params['save_best_on_sensitivity_tolerance']:
                    should_save = True
                    save_reason = f"FAR reduced by {far_reduction*100:.1f}% (sens within {params['save_best_on_sensitivity_tolerance']*100:.0f}% tolerance)"
            
            else:
                # Use metric-based selection (F1, F-beta, Youden's J)
                current_metric = calculate_selection_metric(sensitivity, far, metric_type, params['f_beta'])
                
                # Initialize best metric on first epoch
                if epoch == 1:
                    best_metric = current_metric
                else:
                    if not hasattr(params, '_best_metric'):
                        params['_best_metric'] = -float('inf')
                    best_metric = params['_best_metric']
                
                if current_metric > best_metric:
                    should_save = True
                    params['_best_metric'] = current_metric
                    save_reason = f"{metric_type}={current_metric:.4f} improved (was {best_metric:.4f}, sens={sensitivity:.4f}, FAR={far:.4f})"
        
        # Fallback: also save on best validation loss if no other criteria used
        if not should_save and not params['use_lr_schedule']:
            if val_loss < best_val_loss and not exceeds_max_fpr and not below_min_sensitivity:
                should_save = True
                save_reason = "validation loss improved"
        
        if should_save:
            best_val_loss = val_loss
            best_sensitivity = sensitivity
            best_far = far
            patience_counter = 0
            print(f"{TAG}: ✓ Saving checkpoint to {modelFnamePath}")
            print(f"{TAG}:   Reason: {save_reason}")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'sensitivity': sensitivity,
                'far': far,
                'config': configObj
            }, modelFnamePath)
        else:
            patience_counter += 1
            if rejection_reason:
                if params['trainingVerbosity'] > 0 and epoch % 10 == 0:  # Log every 10 epochs to avoid spam
                    print(f"{TAG}: ✗ Not saving: {rejection_reason}")
        
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

    # Convert .pt model to .ptl (PyTorch Lite) format for mobile deployment
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
        nnTester.testModel(configObj, debug, test_ptl=True)
    else:
        nnTester.testModel(configObj, debug, test_ptl=True)
        
    


if __name__ == "__main__":
    main()
