#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
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




def df2trainingData(df, nnModel, debug=False, n_jobs=-1):
    ''' Converts a pandas dataframe df into a list of data and a list of associated seizure classes
    for use by model nnModel.
    This works by taking each row in the dataframe and converting it into a dict with the same
    values as an OpenSeizureDetector datapoint.   It then calls the dp2vector method of the specified
    model to pre-process the data into the format required by the model.

    Args:
        df: pandas dataframe with training data
        nnModel: model object with dp2vector method
        debug: enable debug output
        n_jobs: number of parallel jobs (-1 uses all CPU cores, 1 disables parallelization)

    FIXME: This only works on acceleration magnitude values at the moment - add an option to use 3d data.
    '''
    from joblib import Parallel, delayed
    
    # Detect accelerometer magnitude columns dynamically (M000..Mxxx). This supports
    # different epoch lengths (e.g. 125 samples for 5s, 750 samples for 30s).
    cols = list(df.columns)
    m_cols = [c for c in cols if isinstance(c, str) and c.startswith('M') and c.endswith('_t-0')]
    if len(m_cols) == 0:
        print("cols are: ", [c for c in cols])
        raise ValueError("df2trainingData: No magnitude (Mxxx) columns found in dataframe")
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

    def process_row(n, rowArr):
        """Process a single row of the dataframe"""
        dpDict = {}
        eventId = rowArr.iloc[eventIdCol]
        
        accArr = rowArr.iloc[accStartCol:accEndCol].values.astype(float).tolist()
        dpDict['rawData'] = accArr
        # HR may be missing in feature CSVs; handle missing hr gracefully
        if hrCol is not None:
            try:
                dpDict['hr'] = int(rowArr.iloc[hrCol])
            except Exception:
                dpDict['hr'] = None
        else:
            dpDict['hr'] = None
        
        dpInputData = nnModel.dp2vector(dpDict, normalise=True)
        if dpInputData is not None:
            return (dpInputData, rowArr.iloc[typeCol], eventId)
        return (None, None, eventId)

    print("Processing Events in parallel...")
    
    # Process rows in parallel
    results = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(
        delayed(process_row)(n, df.iloc[n]) for n in range(len(df))
    )
    
    # Filter out None results and separate data/classes
    outLst = []
    classLst = []
    for dpInputData, dpClass, eventId in results:
        if dpInputData is not None:
            outLst.append(dpInputData)
            classLst.append(dpClass)
    
    print(f"\nProcessed {len(outLst)}/{len(df)} datapoints")
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
    params['lr_main_end'] = libosd.configUtils.getConfigParam("lrMainEnd", configObj['modelConfig'])
    if params['lr_main_end'] is None:
        params['lr_main_end'] = 3e-5
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
    else:
        # PyTorch history is already a dict
        val_acc = np.array(history['val_accuracy'])
        acc = np.array(history['accuracy'])
        loss = np.array(history['loss'])
        val_loss = np.array(history['val_loss'])
        metric_name = 'accuracy'
    
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
                opt = keras.optimizers.AdamW(learning_rate=schedule)
                print("Using tf.keras.optimizers.AdamW")
            except Exception:
                try:
                    import tensorflow_addons as tfa
                    opt = tfa.optimizers.AdamW(learning_rate=schedule, weight_decay=0.0)
                    print("Using tensorflow_addons.optimizers.AdamW")
                except Exception:
                    print("AdamW not available, falling back to Adam with scheduled LR")
                    opt = keras.optimizers.Adam(learning_rate=schedule)
        else:
            opt = keras.optimizers.Adam(learning_rate=schedule)

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
    
    # Create model
    print("Creating PyTorch Model")
    model = nnModel.makeModel(input_shape=xTrain.shape[1:], num_classes=nClasses, nLayers=params['nLayers'])
    device = nnModel.device
    
    # Convert numpy arrays to PyTorch tensors
    xTrain_tensor = torch.from_numpy(xTrain).float()
    yTrain_tensor = torch.from_numpy(yTrain).long()
    xVal_tensor = torch.from_numpy(xVal).float()
    yVal_tensor = torch.from_numpy(yVal).long()
    
    # Create data loaders
    train_dataset = TensorDataset(xTrain_tensor, yTrain_tensor)
    val_dataset = TensorDataset(xVal_tensor, yVal_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=params['lrStart'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=params['lrFactor'], patience=params['lrPatience'], min_lr=params['lrMin']
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"{TAG}: Starting training for {params['epochs']} epochs")
    
    # Training loop
    for epoch in range(params['epochs']):
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
            
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Print progress
        if params['trainingVerbosity'] > 0:
            print(f"Epoch {epoch+1}/{params['epochs']} - "
                  f"loss: {train_loss:.4f} - acc: {train_accuracy:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"Saving best model to {modelFnamePath}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': configObj
            }, modelFnamePath)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= params['earlyStoppingPatience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"Trained using {np.count_nonzero(yTrain == 1)} seizure datapoints and {np.count_nonzero(yTrain == 0)} false alarm datapoints")

    # Plot training history
    plot_training_history(history, params['modelFnameRoot'], dataDir, framework='pytorch')

    print("Training Complete")



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
        nnTester.testModel(configObj, debug)
    else:
        nnTester.testModel(configObj, debug)
        
    


if __name__ == "__main__":
    main()
