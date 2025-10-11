#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import importlib
#from tkinter import Y
import sklearn.metrics
import imblearn.over_sampling
import tensorflow as tf
from tensorflow import keras
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

    # Detect accelerometer magnitude columns dynamically (M000..Mxxx). This supports
    # different epoch lengths (e.g. 125 samples for 5s, 750 samples for 30s).
    cols = list(df.columns)
    m_cols = [c for c in cols if isinstance(c, str) and c.startswith('M') and len(c) == 4 and c[1:].isdigit()]
    if len(m_cols) == 0:
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
    eventIdCol = df.columns.get_loc('id')

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
        dpInputData = nnModel.dp2vector(dpDict, normalise=True)
        if (dpInputData is not None):
            outLst.append(dpInputData)
            classLst.append(rowArr.iloc[typeCol])
        dpDict = None
        dpInputData = None
    print(".")
    return(outLst, classLst)


def trainModel(configObj, dataDir='.', debug=False):
    ''' Create and train a new neural network model, saving it with filename starting 
    with the modelFnameRoot parameter.
    '''
    TAG = "nnTrainer.trainmodel()"
    print("%s" % (TAG))
    trainAugCsvFname = libosd.configUtils.getConfigParam('trainAugmentedFileCsv', configObj['dataFileNames'])
    valCsvFname = libosd.configUtils.getConfigParam('valDataFileCsv', configObj['dataFileNames'])
    testCsvFname = libosd.configUtils.getConfigParam("testDataFileCsv", configObj['dataFileNames'])

    modelFnameRoot = libosd.configUtils.getConfigParam("modelFname", configObj['modelConfig'])
    epochs = libosd.configUtils.getConfigParam("epochs", configObj['modelConfig'])
    batch_size = libosd.configUtils.getConfigParam("batchSize", configObj['modelConfig'])
    nLayers = libosd.configUtils.getConfigParam("nLayers", configObj['modelConfig'])
    lrFactor = libosd.configUtils.getConfigParam("lrFactor", configObj['modelConfig'])
    lrPatience = libosd.configUtils.getConfigParam("lrPatience", configObj['modelConfig'])
    lrStart = libosd.configUtils.getConfigParam("lrStart", configObj['modelConfig'])
    lrMin = libosd.configUtils.getConfigParam("lrMin", configObj['modelConfig'])
    # Optional advanced LR schedule / AdamW settings (defaults follow paper values)
    use_lr_schedule = libosd.configUtils.getConfigParam("useLrSchedule", configObj['modelConfig'])
    if use_lr_schedule is None:
        use_lr_schedule = False
    lr_peak = libosd.configUtils.getConfigParam("lrPeak", configObj['modelConfig'])
    if lr_peak is None:
        lr_peak = 1e-3
    lr_main_end = libosd.configUtils.getConfigParam("lrMainEnd", configObj['modelConfig'])
    if lr_main_end is None:
        lr_main_end = 3e-5
    warmup_steps = libosd.configUtils.getConfigParam("warmupSteps", configObj['modelConfig'])
    if warmup_steps is None:
        warmup_steps = 2500
    main_steps = libosd.configUtils.getConfigParam("mainSteps", configObj['modelConfig'])
    if main_steps is None:
        main_steps = 45000
    cooldown_steps = libosd.configUtils.getConfigParam("cooldownSteps", configObj['modelConfig'])
    if cooldown_steps is None:
        cooldown_steps = 2500
    use_adamw = libosd.configUtils.getConfigParam("useAdamW", configObj['modelConfig'])
    if use_adamw is None:
        use_adamw = True
    earlyStoppingPatience = libosd.configUtils.getConfigParam("earlyStoppingPatience", configObj['modelConfig'])
    validationProp = libosd.configUtils.getConfigParam("validationProp", configObj['dataProcessing'])
    trainingVerbosity = libosd.configUtils.getConfigParam("trainingVerbosity", configObj['modelConfig'])
    nnModelClassName = libosd.configUtils.getConfigParam("modelClass", configObj['modelConfig'])

    if (validationProp == 0):
        print("WARNING: validationProp set to 0 - no validation data used - using test data instead")
        valCsvFname = testCsvFname

    inputDims = libosd.configUtils.getConfigParam("dims", configObj)
    if (inputDims is None): inputDims = 1

    # Load Model class from nnModelClassName
    modelFname = "%s.keras" % modelFnameRoot
    modelFnamePath = os.path.join(dataDir, modelFname)
    parts = nnModelClassName.split('.')
    if len(parts) < 2:
        raise ValueError("modelClass must be a module path and class name, e.g. 'mod.submod.ClassName'")
    nnModuleId = '.'.join(parts[:-1])
    nnClassId = parts[-1]

    print("%s: Importing nn Module %s" % (TAG, nnModuleId))
    nnModule = importlib.import_module(nnModuleId)
    # instantiate the class from the module
    nnModel = getattr(nnModule, nnClassId)(configObj['modelConfig'])

    # Load the training data from file
    trainAugCsvFnamePath = os.path.join(dataDir, trainAugCsvFname)
    valCsvFnamePath = os.path.join(dataDir, valCsvFname)

    # If feature CSVs exist (produced by runSequence/extractFeatures), prefer them.
    # This ensures compatibility with the runSequence toolchain which now produces
    # feature CSVs (possibly history-augmented). If these files aren't present,
    # fall back to the augmented/flattened CSVs (legacy behavior).
    try:
        trainFeaturesName = configObj['dataFileNames'].get('trainFeaturesFileCsv')
        testFeaturesName = configObj['dataFileNames'].get('testFeaturesFileCsv')
    except Exception:
        trainFeaturesName = None
        testFeaturesName = None

    if trainFeaturesName is not None:
        candidate = os.path.join(dataDir, trainFeaturesName)
        if os.path.exists(candidate):
            print("nnTrainer: Using train features CSV %s" % candidate)
            trainAugCsvFnamePath = candidate
    if testFeaturesName is not None:
        candidate = os.path.join(dataDir, testFeaturesName)
        if os.path.exists(candidate):
            print("nnTrainer: Using validation/test features CSV %s" % candidate)
            valCsvFnamePath = candidate

    print("%s: Loading training data from file %s" % (TAG, trainAugCsvFnamePath))
    if not os.path.exists(trainAugCsvFnamePath):
        print("ERROR: File %s does not exist" % trainAugCsvFnamePath)
        exit(-1)


    valCsvFnamePath = os.path.join(dataDir, valCsvFname)
    print("%s: Loading validation data from file %s" % (TAG, valCsvFnamePath))
    if not os.path.exists(valCsvFnamePath):
        print("ERROR: File %s does not exist" % valCsvFnamePath)
        exit(-1)

    df = augmentData.loadCsv(trainAugCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(df), trainAugCsvFname))
    #augmentData.analyseDf(df)


    print("%s: Re-formatting data for training" % (TAG))
    xTrain, yTrain = df2trainingData(df, nnModel)

    print("xTrain=", type(xTrain))
    print("yTrain=", type(yTrain))

    #for n in range(0,len(xTrain)):
    #    print(type(xTrain[n]))

    print("%s: Converting to np arrays" % (TAG))
    try:
        xTrain = np.array(xTrain)
    except ValueError as e:
        print("Failed simple array conversion - trying concatenate...")
        xTrain = np.concatenate(xTrain)
    yTrain = np.array(yTrain)


    print("xTrain.shape=",xTrain.shape,", yTrain.shape=",yTrain.shape)
    print("%s: re-shaping array for training" % (TAG))
    #xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1))

    if (inputDims == 1):
        xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    elif (inputDims ==2):
        xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1))
    else:
        print("ERROR - inputDims out of Range: %d" % inputDims)
        exit(-1)


    # Load and preprocess the validation data
    print("%s: Loading validation data from file %s" % (TAG, valCsvFnamePath))
    df = augmentData.loadCsv(valCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(df), valCsvFname))
    #augmentData.analyseDf(df)


    print("%s: Re-formatting data for training" % (TAG))
    xVal, yVal= df2trainingData(df, nnModel)

    print("xVal=", type(xVal))
    print("yVal=", type(yVal))

    print("%s: Converting to np arrays" % (TAG))
    try:
        xVal = np.array(xVal)
    except ValueError as e:
        print("Failed simple array conversion - trying concatenate...")
        xVal = np.concatenate(xVal)
    yVal= np.array(yVal)


    print("xVal.shape=",xVal.shape,", yVal.shape=",yVal.shape)
    print("%s: re-shaping array for training" % (TAG))

    if (inputDims == 1):
        xVal = xVal.reshape((xVal.shape[0], xVal.shape[1], 1))
    elif (inputDims ==2):
        xVal = xVal.reshape((xVal.shape[0], xVal.shape[1], xVal.shape[2], 1))
    else:
        print("ERROR - inputDims out of Range: %d" % inputDims)
        exit(-1)

    
    nClasses = len(np.unique(yTrain))
    print("nClasses=%d" % nClasses)
    print("Training using %d seizure datapoints and %d false alarm datapoints"
            % (np.count_nonzero(yTrain == 1),
            np.count_nonzero(yTrain == 0)))

   
    if (os.path.exists(modelFnamePath)):
        print("Model %s already exists - loading existing model as starting point for training" % modelFnamePath)
        model = keras.models.load_model(modelFnamePath)
        print("Model %s loaded" % modelFnamePath)
    else:
        print("Creating new Model")
        model = nnModel.makeModel(input_shape=xTrain.shape[1:], num_classes=nClasses,
            nLayers=nLayers)
    
    keras.utils.plot_model(model, show_shapes=True)


    # Build callbacks. If using a scheduled LR, we rely on the schedule instead of ReduceLROnPlateau.
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            modelFnamePath, save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=earlyStoppingPatience, 
            verbose=trainingVerbosity),
    ]

    # Determine steps per epoch so schedules can be expressed in steps
    try:
        steps_per_epoch = int(np.ceil(xTrain.shape[0] / float(batch_size)))
    except Exception:
        steps_per_epoch = None

    # Learning rate configuration: either a static lrStart or an advanced schedule
    if use_lr_schedule:
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

        schedule = PaperLrSchedule(warmup_steps, main_steps, cooldown_steps, lrStart, lr_peak, lr_main_end, 0.0)

        # Try to use AdamW if available, otherwise fall back to Adam
        opt = None
        if use_adamw:
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
        callbacks.insert(1, keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=lrFactor, patience=lrPatience, min_lr=lrMin))
        opt = keras.optimizers.Adam(learning_rate=lrStart)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    
    history = model.fit(
        xTrain,
        yTrain,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        #validation_split=validationProp,
        validation_data=(xVal, yVal),
        verbose=trainingVerbosity

    )
    print("Trained using %d seizure datapoints and %d false alarm datapoints"
        % (np.count_nonzero(yTrain == 1),
        np.count_nonzero(yTrain == 0)))

    #Train and Validation: multi-class log-Loss & accuracy plot
    print("Plotting training history")
    plt.figure(figsize=(12, 8))
    plt.plot(np.array(history.history['val_sparse_categorical_accuracy']), "r--", label = "val_sparse_categorical_accuracy")
    plt.plot(np.array(history.history['sparse_categorical_accuracy']), "g--", label = "sparse_categorical_accuracy")
    plt.plot(np.array(history.history['loss']), "y--", label = "Loss")
    plt.plot(np.array(history.history['val_loss']), "p-", label = "val_loss")
    plt.title("Training session's progress over iterations")
    plt.legend(loc='lower left')
    plt.ylabel('Training Progress (Loss/Accuracy)')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.savefig(os.path.join(dataDir,"%s_training.png" % modelFnameRoot))
    plt.close()
    
    
    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(os.path.join(dataDir,"%s_training2.png" % modelFnameRoot))
    plt.close()

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
        trainModel(configObj, debug)
        nnTester.testModel(configObj, debug)
    else:
        nnTester.testModel(configObj, debug)
        
    


if __name__ == "__main__":
    main()
