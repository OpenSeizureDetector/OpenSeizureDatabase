#!/usr/bin/env python3

import argparse
#from re import X
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.configUtils

import augmentData






def visualiseData(configObj, dataDir='.', debug=False):
    ''' Load the training data and plot some graphs....
    '''
    TAG = "visualiser.visualiseData()"
    print("%s" % (TAG))
    trainFeatCsvFname = libosd.configUtils.getConfigParam('trainFeaturesFileCsv', configObj['dataFileNames'])
    valCsvFname = libosd.configUtils.getConfigParam('valDataFileCsv', configObj['dataFileNames'])
    testCsvFname = libosd.configUtils.getConfigParam("testDataFileCsv", configObj['dataFileNames'])

    # Load the training data from file
    trainAugCsvFnamePath = os.path.join(dataDir, trainFeatCsvFname)

    print("%s: Loading training data from file %s" % (TAG, trainAugCsvFnamePath))
    if not os.path.exists(trainAugCsvFnamePath):
        print("ERROR: File %s does not exist" % trainAugCsvFnamePath)
        exit(-1)



    df = augmentData.loadCsv(trainAugCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(df), trainFeatCsvFname))
    #augmentData.analyseDf(df)

    df = df.sort_values(by='type', ascending=True)
    print(df)

    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='specPower', y='roiPower', hue='type', style='type', data=df, markers={1:'o',  0:'s'})
    #sns.kdeplot(x='specPower', y='roiPower', hue='type', data=df, fill=True, cmap='coolwarm')
    #sns.jointplot(x='specPower', y='roiPower', kind='hex', data=df, cmap='viridis', height=7) # 'kind='hex' for density plot, cmap for color scheme

    plt.xlim(0, 1e5)  # Set x-axis range from 0 to 1
    plt.ylim(0, 4e5)
    plt.xlabel('Spec Power')
    plt.ylabel('ROI Power')
    plt.title('Scatter Plot with Markers by Category (Seaborn)')
    plt.show()

    featuresLst = ['type']
    featuresLst.extend(configObj['dataProcessing']['features'])
    print("plotting features:",featuresLst)
    powerFrame = df[featuresLst].copy()
    powerFrame['type'] = powerFrame['type'].astype(str)
    plt.figure(figsize=(16, 12))
    _ = sns.pairplot(powerFrame, hue="type")
    plt.show()

    print("%s: FIXME - actually do something!" % (TAG))
    exit(-1)

    yTrain = df['y'].values

    
    nClasses = len(np.unique(yTrain))
    print("nClasses=%d" % nClasses)
    print("Training using %d seizure datapoints and %d false alarm datapoints"
            % (np.count_nonzero(yTrain == 1),
            np.count_nonzero(yTrain == 0)))

   
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
    print("visualiser.main()")
    parser = argparse.ArgumentParser(description='Seizure Detection Training Data Visualiser')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing configuration')
    parser.add_argument('--dataDir', default=".",
                        help='directory containing the output files to visualise')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
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
    dataDir = args['dataDir']
    if not os.path.exists(dataDir):
        print("ERROR - Output directory %s does not exist!" % dataDir)
        exit(-1)

    visualiseData(configObj, dataDir=dataDir, debug=debug)        
    


if __name__ == "__main__":
    main()
