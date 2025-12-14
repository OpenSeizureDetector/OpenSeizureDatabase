#!/usr/bin/env python3

import argparse
from re import X
import sys
import os
import importlib
#from tkinter import Y
import pandas as pd
import sklearn.metrics
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

from sklearn.metrics import classification_report
from sklearn import metrics

import nnTrainer


def get_model_extension(framework):
    """Get the appropriate file extension for the framework."""
    if framework == 'pytorch':
        return '.pt'
    else:
        return '.keras'


def load_model_for_testing(modelFnamePath, nnModel, framework='tensorflow'):
    """Load a trained model for testing (framework-agnostic).
    
    Args:
        modelFnamePath: Path to the model file
        nnModel: Model instance (needed for PyTorch architecture)
        framework: 'tensorflow' or 'pytorch'
    
    Returns:
        Loaded model ready for inference
    """
    if framework == 'tensorflow':
        from tensorflow import keras
        model = keras.models.load_model(modelFnamePath)
        return model
    elif framework == 'pytorch':
        import torch
        # For PyTorch, we need to recreate the model first, then load weights
        checkpoint = torch.load(modelFnamePath, map_location=nnModel.device)
        
        # Get model configuration from checkpoint
        if 'config' in checkpoint and 'modelConfig' in checkpoint['config']:
            nLayers = checkpoint['config']['modelConfig'].get('nLayers', 14)
        else:
            nLayers = 14
        
        # Infer input shape and num_classes from the checkpoint state_dict
        model_state = checkpoint['model_state_dict']
        num_classes = None
        input_shape = None
        
        # Find num_classes from final layer
        for key in sorted(model_state.keys(), reverse=True):
            if 'fc' in key and 'weight' in key:
                num_classes = model_state[key].shape[0]
                break
        
        # Find input shape from first conv layer  
        for key in sorted(model_state.keys()):
            if 'conv' in key and 'weight' in key:
                # Shape is (out_channels, in_channels, kernel_size)
                # For 1D conv: (out, in, kernel)
                in_channels = model_state[key].shape[1]
                # We'll use a placeholder sequence length, actual shape will come from data
                input_shape = (750, in_channels)  # 750 is typical sequence length
                break
        
        # Create model architecture if not already created
        if nnModel.model is None:
            nnModel.makeModel(input_shape=input_shape, num_classes=num_classes, nLayers=nLayers)
        
        nnModel.model.load_state_dict(checkpoint['model_state_dict'])
        nnModel.model.eval()
        return nnModel.model
    else:
        raise ValueError(f"Unknown framework: {framework}")


def evaluate_model(model, xTest, yTest, framework='tensorflow'):
    """Evaluate model and return loss and accuracy (framework-agnostic).
    
    Args:
        model: Trained model
        xTest: Test data
        yTest: Test labels
        framework: 'tensorflow' or 'pytorch'
    
    Returns:
        tuple: (test_loss, test_acc)
    """
    if framework == 'tensorflow':
        test_loss, test_acc = model.evaluate(xTest, yTest, verbose=0)
        return test_loss, test_acc
    elif framework == 'pytorch':
        import torch
        import torch.nn as nn
        
        model.eval()
        device = next(model.parameters()).device
        
        # Convert to tensors if needed
        if not isinstance(xTest, torch.Tensor):
            xTest_tensor = torch.from_numpy(xTest).float().to(device)
            yTest_tensor = torch.from_numpy(yTest).long().to(device)
        else:
            xTest_tensor = xTest.to(device)
            yTest_tensor = yTest.to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            outputs = model(xTest_tensor)
            loss = criterion(outputs, yTest_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == yTest_tensor).sum().item() / yTest_tensor.size(0)
        
        loss_value = loss.item()
        
        # Clean up GPU memory
        del xTest_tensor, yTest_tensor, outputs, predicted, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss_value, accuracy
    else:
        raise ValueError(f"Unknown framework: {framework}")


def predict_model(model, xTest, framework='tensorflow'):
    """Get prediction probabilities from model (framework-agnostic).
    
    Args:
        model: Trained model
        xTest: Test data
        framework: 'tensorflow' or 'pytorch'
    
    Returns:
        numpy array of prediction probabilities, shape (n_samples, n_classes)
    """
    if framework == 'tensorflow':
        return model.predict(xTest, verbose=0)
    elif framework == 'pytorch':
        import torch
        
        model.eval()
        device = next(model.parameters()).device
        
        # Convert to tensor if needed
        if not isinstance(xTest, torch.Tensor):
            xTest_tensor = torch.from_numpy(xTest).float().to(device)
        else:
            xTest_tensor = xTest.to(device)
        
        with torch.no_grad():
            outputs = model(xTest_tensor)
            probs = torch.softmax(outputs, dim=1)
            probs_numpy = probs.cpu().numpy()
        
        # Clean up GPU memory
        del xTest_tensor, outputs, probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return probs_numpy
    else:
        raise ValueError(f"Unknown framework: {framework}")


def testModel(configObj, dataDir='.', balanced=True, debug=False):
    TAG = "nnTrainer.testModel()"
    print("____%s____" % (TAG))
    
    # Detect framework
    framework = nnTrainer.get_framework_from_config(configObj)
    print(f"{TAG}: Using framework: {framework}")
    
    modelFnameRoot = libosd.configUtils.getConfigParam("modelFname", configObj['modelConfig'])
    nnModelClassName = libosd.configUtils.getConfigParam("modelClass", configObj['modelConfig'])
    #testDataFname = libosd.configUtils.getConfigParam("testDataFileCsv", configObj['dataFileNames'])
    if (balanced):
        testDataFname = libosd.configUtils.getConfigParam("testBalancedFileCsv", configObj['dataFileNames'])
    else:   
        testDataFname = libosd.configUtils.getConfigParam("testFeaturesHistoryFileCsv", configObj['dataFileNames'])
    
    if testDataFname is None:
        raise ValueError(f"{TAG}: Test data filename is None. Check config for testBalancedFileCsv or testFeaturesHistoryFileCsv")

    inputDims = libosd.configUtils.getConfigParam("dims", configObj['modelConfig'])
    if (inputDims is None): inputDims = 1

    modelExt = get_model_extension(framework)
    modelFname = f"{modelFnameRoot}{modelExt}"
    
    # Parse model class name properly
    parts = nnModelClassName.split('.')
    if len(parts) < 2:
        raise ValueError("modelClass must be a module path and class name, e.g. 'mod.submod.ClassName'")
    nnModuleId = '.'.join(parts[:-1])
    nnClassId = parts[-1]

    print("%s: Importing nn Module %s" % (TAG, nnModuleId))
    nnModule = importlib.import_module(nnModuleId)
    # Instantiate the model class with modelConfig
    if configObj.get('modelConfig') is None:
        raise ValueError(f"{TAG}: configObj['modelConfig'] is None")
    nnModel = getattr(nnModule, nnClassId)(configObj['modelConfig'])

    # Load the test data from file
    if testDataFname is None:
        raise ValueError(f"{TAG}: testDataFname is None - check your config file")
    print("%s: Loading Test Data from File %s" % (TAG, testDataFname))
    df = augmentData.loadCsv(os.path.join(dataDir, testDataFname), debug=debug)
    print("%s: Loaded %d datapoints" % (TAG, len(df)))
    #augmentData.analyseDf(df)

    print("%s: Re-formatting data for testing" % (TAG))
    xTest, yTest = nnTrainer.df2trainingData(df, nnModel)

    print("%s: Converting to np arrays" % (TAG))
    xTest = np.array(xTest)
    yTest = np.array(yTest)

    print("%s: re-shaping array for testing" % (TAG))
    #xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
    if (inputDims == 1):
        xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
    elif (inputDims ==2):
        xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], xTest.shape[2], 1))
    else:
        print("ERROR - inputDims out of Range: %d" % inputDims)
        exit(-1)


    # Load the best model back from disk and test it.
    modelFnamePath = os.path.join(dataDir, modelFname)
    model = load_model_for_testing(modelFnamePath, nnModel, framework)

    test_loss, test_acc = evaluate_model(model, xTest, yTest, framework)

   
    print("Tesing using %d seizure datapoints and %d false alarm datapoints"
        % (np.count_nonzero(yTest == 1),
        np.count_nonzero(yTest == 0)))

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

 
    calcConfusionMatrix(configObj, modelFnameRoot, xTest, yTest, dataDir=dataDir, balanced=balanced, debug=debug)

    testModel2(configObj, dataDir=dataDir, balanced=balanced, debug=debug)

    # Clean up memory
    if framework == 'pytorch':
        import torch
        del model, nnModel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"{TAG}: CUDA memory cleared")
    
    return None


def testModel2(configObj, dataDir='.', balanced=True, debug=False):
    TAG = "nnTester.testModel2()"
    print("____%s____ dataDir=%s" % (TAG, dataDir))
    
    # Detect framework
    framework = nnTrainer.get_framework_from_config(configObj)
    
    modelFnameRoot = libosd.configUtils.getConfigParam("modelFname", configObj['modelConfig'])
    nnModelClassName = libosd.configUtils.getConfigParam("modelClass", configObj['modelConfig'])
    if (balanced):
        testDataFname = libosd.configUtils.getConfigParam("testBalancedFileCsv", configObj['dataFileNames'])
    else:   
        testDataFname = libosd.configUtils.getConfigParam("testDataFileCsv", configObj['dataFileNames'])

    inputDims = libosd.configUtils.getConfigParam("dims", configObj['modelConfig'])
    if (inputDims is None): inputDims = 1

    modelExt = get_model_extension(framework)
    modelFname = f"{modelFnameRoot}{modelExt}"
    modelFnamePath = os.path.join(dataDir, modelFname)
    if (not os.path.exists(modelFnamePath)):
        print("ERROR - Model file %s does not exist" % modelFnamePath)
        exit(-1)

    # Parse model class name properly
    parts = nnModelClassName.split('.')
    if len(parts) < 2:
        raise ValueError("modelClass must be a module path and class name, e.g. 'mod.submod.ClassName'")
    nnModuleId = '.'.join(parts[:-1])
    nnClassId = parts[-1]

    print("%s: Importing nn Module %s" % (TAG, nnModuleId))
    nnModule = importlib.import_module(nnModuleId)
    # Instantiate the model class with modelConfig
    nnModel = getattr(nnModule, nnClassId)(configObj['modelConfig'])

    # Load the test data from file
    testDataFnamePath = os.path.join(dataDir, testDataFname)
    print("%s: Loading Test Data from File %s" % (TAG, testDataFnamePath))
    if (not os.path.exists(testDataFnamePath)):
        print("ERROR - Test data file %s does not exist" % testDataFnamePath)
        exit(-1)

    # Note, we are not actually doing augmentation here, this is just using a utility function - FIXME - move it to a more obvious place!
    df = augmentData.loadCsv(testDataFnamePath, debug=debug)
    print("%s: Loaded %d datapoints" % (TAG, len(df)))
    #augmentData.analyseDf(df)

    print("%s: Re-formatting data for testing" % (TAG))
    xTest, yTest = nnTrainer.df2trainingData(df, nnModel)

    print("%s: Converting to np arrays" % (TAG))
    xTest = np.array(xTest)
    yTest = np.array(yTest)

    print("%s: re-shaping array for testing" % (TAG))
    print(xTest.shape)
    #xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
    if (inputDims == 1):
        xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
    elif (inputDims ==2):
        xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], xTest.shape[2], 1))
    else:
        print("ERROR - inputDims out of Range: %d" % inputDims)
        exit(-1)


    print("Tesing using %d seizure datapoints and %d false alarm datapoints"
        % (np.count_nonzero(yTest == 1),
        np.count_nonzero(yTest == 0)))

    # Load the best model back from disk and test it.
    print("%s: Loading Model from file %s" % (TAG, modelFnamePath))
    if (not os.path.exists(modelFnamePath)):
        print("ERROR - Model file %s does not exist" % modelFnamePath)
        exit(-1)
    print("%s: Loading trained model %s" % (TAG, modelFnamePath))
    model = load_model_for_testing(modelFnamePath, nnModel, framework)

    #print("%s: Evaluating Model" % TAG)
    #test_loss, test_acc = evaluate_model(model, xTest, yTest, framework)
    #print("%s: Test accuracy = %.2f" % (TAG,test_acc))
    #print("%s: Test loss     = %.2f" % (TAG,test_loss))

    print("%s: Calculating Seizure probabilities from test data" % TAG)
    prediction_proba = predict_model(model, xTest, framework)
    if (debug): print("prediction_proba=",prediction_proba)

    # Here prediction is the index of the highest probability classification in the row,
    # so 0 = ok, 1 = seizure
    prediction=np.argmax(prediction_proba,axis=1)

    if (debug): print("prediction=", prediction)

    pSeizure = prediction_proba[:,1]
    seq = range(0,len(pSeizure))
    # Colour seizure data points red, and non-seizure data blue.
    colours = [ 'red' if seizureVal==1 else 'blue' for seizureVal in yTest]
    #print("pSeizure=", pSeizure)
    #print(colours)


    thLst = []
    nTPLst = []
    nFPLst = []
    nTNLst = []
    nFNLst = []
    TPRLst = []
    FPRLst = []

    thresholdLst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for th in thresholdLst:
        nTP, nFP, nTN, nFN = calcTotals(yTest, pSeizure, th)
        thLst.append(th)
        nTPLst.append(nTP)
        nFPLst.append(nFP)
        nTNLst.append(nTN)
        nFNLst.append(nFN)

        TPRLst.append(nTP/(nTP+nFN))
        FPRLst.append(nFP/(nFP+nTN))

    print("Stats!")
    print("th", thLst)    
    print("nTP", nTPLst)
    print("nFP", nFPLst)
    print("nTN", nTNLst)
    print("nFN", nFNLst)
    print("TPR", TPRLst)
    print("FPR", FPRLst)


    fig, ax = plt.subplots(3,1)
    ax[0].title.set_text("%s: Seizure Probabilities" % modelFnameRoot)
    ax[0].set_ylabel('Probability')
    ax[0].set_xlabel('Datapoint')
    ax[0].scatter(seq, pSeizure, s=2.0, marker='x', c=colours)

    ax[1].plot(yTest)
    fname = os.path.join(dataDir,"%s_probabilities.png" % modelFnameRoot)
    fig.savefig(fname)
    plt.close()

    # Call calcConfusionMatrix (which will clean up its own memory)
    calcConfusionMatrix(configObj, modelFnameRoot, xTest, yTest, dataDir=dataDir, balanced=balanced, debug=debug)
    
    # Additional cleanup for testModel2
    if framework == 'pytorch':
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"{TAG}: Final CUDA memory cleared")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"{TAG}: CUDA memory cleared")



def calcTotals(yTest, pSeizure, th = 0.5):
    ''' Calculate true positive (TP), True Negative (TN), False Positive (FP)
    and False Negative (FN) totals, for the data in yTest (where 0=ok and 1 = seizure)
    and pSeizure which is the probability of the event being a seizure, uisng threshold th.

    FIXME: I am sure there is a more efficient python way of doing this - this is how I 
    would have written it in C  :)

    '''
    nTP = 0
    nTN = 0
    nFP = 0
    nFN = 0

    for i in range(0,len(yTest)):
        if (yTest[i] == 1):   # Event was a seizure
            if (pSeizure[i]>th):
                nTP += 1   # True Positive
            else:
                nFN += 1   # False Negative
        elif (yTest[i] ==0):  # Event was not a seizure
            if (pSeizure[i]>th):
                nFP += 1   # False Positive
            else:
                nTN += 1   # True Negative
        else:
            print("WARNING - Unrecognised yTest Value: %d" % yTest[i])

    return(nTP, nFP, nTN, nFN)


def calcConfusionMatrix(configObj, modelFnameRoot="best_model", 
                        xTest=None, yTest=None, dataDir=".", balanced=True, debug=False):

    TAG = "nnTrainer.calcConfusionMatrix()"
    print("____%s____" % (TAG))
    
    # Detect framework
    framework = nnTrainer.get_framework_from_config(configObj)
    
    nnModelClassName = libosd.configUtils.getConfigParam("modelClass", configObj['modelConfig'])
    if (balanced):
        testDataFname = os.path.join(dataDir, libosd.configUtils.getConfigParam("testBalancedFileCsv", configObj['dataFileNames']))
    else:   
        testDataFname = os.path.join(dataDir, libosd.configUtils.getConfigParam("testDataFileCsv", configObj['dataFileNames']))

    inputDims = libosd.configUtils.getConfigParam("dims", configObj['modelConfig'])
    if (inputDims is None): inputDims = 1

    modelExt = get_model_extension(framework)
    modelFname = f"{modelFnameRoot}{modelExt}"
    
    # Parse model class name properly
    parts = nnModelClassName.split('.')
    if len(parts) < 2:
        raise ValueError("modelClass must be a module path and class name, e.g. 'mod.submod.ClassName'")
    nnModuleId = '.'.join(parts[:-1])
    nnClassId = parts[-1]

    if (debug): print("%s: Importing nn Module %s" % (TAG, nnModuleId))
    nnModule = importlib.import_module(nnModuleId)
    # Instantiate the model class with modelConfig
    nnModel = getattr(nnModule, nnClassId)(configObj['modelConfig'])

    if (xTest is None or yTest is None):
        # Load the test data from file
        print("%s: Loading Test Data from File %s" % (TAG, testDataFname))
        df = augmentData.loadCsv(testDataFname, debug=debug)
        print("%s: Loaded %d datapoints" % (TAG, len(df)))
        #augmentData.analyseDf(df)

        print("%s: Re-formatting data for testing" % (TAG))
        xTest, yTest = nnTrainer.df2trainingData(df, nnModel)

        print("%s: Converting to np arrays" % (TAG))
        xTest = np.array(xTest)
        yTest = np.array(yTest)

        print("%s: re-shaping array for testing" % (TAG))
        if (inputDims == 1):
            xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
        elif (inputDims ==2):
            xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], xTest.shape[2], 1))
        else:
            print("ERROR - inputDims out of Range: %d" % inputDims)
            exit(-1)

    nClasses = len(np.unique(yTest))
    print("nClasses=%d" % nClasses)
    # In the following, yTest == 1 returns an array that is true (1), for all elements where yTest == 1, and false (0) for other values of yTest - we then count how many of those elements are not zero to give
    # the number of elements where yTest = 1.
    # In our case we could have just done count_nonzero(yTest), but doing it this way gives us the option of expanding to more than 2 categories of event.
    print("Testing using %d seizure datapoints and %d false alarm datapoints"
          % (np.count_nonzero(yTest == 1),
             np.count_nonzero(yTest == 0)))


    # Load the trained model back from disk and test it.
    modelFname = os.path.join(dataDir, f"{modelFnameRoot}{modelExt}")
    print("Loading trained model %s" % modelFname)
    model = load_model_for_testing(modelFname, nnModel, framework)
    print("Evaluating model....")
    test_loss, test_acc = evaluate_model(model, xTest, yTest, framework)
    print("Test Loss=%.2f, Test Acc=%.2f" % (test_loss, test_acc))

   
    if (debug): print("yTest=",yTest)
    # create an array of the indices of true seizure events.
    y_true=[]
    for element in yTest:
        y_true.append(np.argmax(element))
    if (debug): print("y_true=",y_true)

    print("Calculating seizure probabilities from test data")
    prediction_proba = predict_model(model, xTest, framework)
    if (debug): print("prediction_proba=",prediction_proba)
    prediction=np.argmax(prediction_proba,axis=1)
    
       
    # Confusion Matrix
    import seaborn as sns
    LABELS = ['No-Alarm','Seizure']
    # cm = metrics.confusion_matrix(prediction, yTest)
    cm = metrics.confusion_matrix(yTest, prediction)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                linewidths = 0.1, fmt="d", cmap = 'YlGnBu');
    plt.title("%s: Confusion matrix" % modelFnameRoot, fontsize = 15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fname = os.path.join(dataDir, "%s_confusion.png" % modelFnameRoot)
    plt.savefig(fname)
    plt.close()
    print("Confusion Matrix Saved as %s." % fname)
    
    nTrue = 0
    nFalse = 0
    nTP = 0
    nFN = 0
    nTN = 0
    nFP = 0
    for n in range (0,len(yTest)):
        if (yTest[n]==1):
            nTrue += 1
        else:
            nFalse += 1
        if (yTest[n]==1):
            if (prediction[n]==1):
                nTP += 1
            else:
                nFN += 1
        else:
            if (prediction[n]==1):
                nFP += 1
            else:
                nTN += 1


    fname = os.path.join(dataDir, "%s_stats.txt" % modelFnameRoot)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    total1=sum(sum(cm))
    with open(fname,"w") as outFile:
        outFile.write("\n|====================================================================|\n")
        outFile.write("****  Open Seizure Detector Classififcation Metrics Metrics  ****\n")
        outFile.write("****  Analysis of %d seizure and non seizure events Classififcation Metrics  ****\n" % total1)
        outFile.write("|====================================================================|\n")
        outFile.write("Totals:  Seizures %d, non-Seizures %d\n" % (nTrue, nFalse))
        outFile.write("    nTP = %d,  nFN= %d\n" % (nTP, nFN))
        outFile.write("    nTN = %d,  nFP= %d\n" % (nTN, nFP))
        tpr = nTP / (nTP + nFN)
        outFile.write("    TPR = %.2f\n" % tpr)
        tnr = nTN / (nTN + nFP)
        outFile.write("    TNR = %.2f\n" % tnr)

        outFile.write("\n Stats from Confusion Matrix Calc\n")
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        #print(TPR, TPR.shape, TPR[0])
        outFile.write("Sensitivity/recall or true positive rate: %.2f  %.2f\n" % tuple(TPR))
        # Specificity or true negative rate
        TNR = TN/(TN+FP)
        #print(TNR)
        outFile.write("Specificity or true negative rate: %.2f  %.2f\n" % tuple(TNR))
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        outFile.write("Precision or positive predictive value: %.2f  %.2f\n" % tuple(PPV))
        # Negative predictive value
        NPV = TN/(TN+FN)
        outFile.write("Negative predictive value: %.2f  %.2f\n" % tuple(NPV))
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        outFile.write("Fall out or false positive rate: %.2f  %.2f\n" % tuple(FPR))
        # False negative rate
        FNR = FN/(TP+FN)
        outFile.write("False negative rate: %.2f  %.2f\n" % tuple(FNR))
        # False discovery rate
        FDR = FP/(TP+FP)
        outFile.write("False discovery rate: %.2f  %.2f\n" % tuple(FDR))
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        outFile.write("Classification Accuracy: %.2f  %.2f\n" % tuple(ACC))
        outFile.write("|====================================================================|\n")
        report = classification_report(yTest, prediction)
        outFile.write(report)
        outFile.write("\n|====================================================================|\n")
        x=keras.metrics.sparse_categorical_accuracy(xTest, yTest)

        # summarize filter shapes
        for layer in model.layers:
        # check for convolutional layer
         if 'conv' not in layer.name:
             continue

        # get filter weights
        filters, biases = layer.get_weights()
        filterStr = layer.name
        for n in filters.shape:
            filterStr="%s, %d" % (filterStr,n)
        filterStr="%s\n" % filterStr
        outFile.write(filterStr)


        # summarize feature map shapes
        for i in range(len(model.layers)):
            layer = model.layers[i]
            # check for convolutional layer
            if 'conv' not in layer.name:
                continue
            # summarize output shape
            outFile.write("%d:  %s : " % (i, layer.name))
            for n in layer.output.shape:
                if n is not None:
                    outFile.write("%d, " % n)
            outFile.write("\n")

    print("Statistics Summary saved as %s." % fname)





def main():
    print("nnTester.main()")
    parser = argparse.ArgumentParser(description='Apply the training data to calculate statistcs on a trained model (specifid in the config file)')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing model configuration')
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

    testModel2(configObj, debug)
        
    


if __name__ == "__main__":
    main()
