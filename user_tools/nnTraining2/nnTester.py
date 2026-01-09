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
import json

import nnTrainer


def fpr_score(y, y_pred, pos_label=1, neg_label=0):
    """Calculate TPR and FPR from predictions."""
    cm = sklearn.metrics.confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = 1 - tnr
    return (tpr, fpr)


def get_model_extension(framework):
    """Get the appropriate file extension for the framework."""
    if framework == 'pytorch':
        return '.pt'
    else:
        return '.keras'


def load_ptl_model_for_testing(modelFnamePath):
    """Load a PyTorch Lite (.ptl) model for testing.
    
    Args:
        modelFnamePath: Path to the .ptl model file
    
    Returns:
        Loaded PTL model ready for inference
    """
    import torch
    
    # Load the mobile-optimized model using jit.load
    # PTL models are TorchScript models optimized for mobile
    model = torch.jit.load(modelFnamePath, map_location=torch.device('cpu'))
    model.eval()
    return model


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
        
        # Check if this is a .ptl (PyTorch Lite) model
        if modelFnamePath.endswith('.ptl'):
            return load_ptl_model_for_testing(modelFnamePath)
        
        # For PyTorch, we need to recreate the model first, then load weights
        # Use weights_only=False since we trust our own checkpoint files
        checkpoint = torch.load(modelFnamePath, map_location=nnModel.device, weights_only=False)
        
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


def evaluate_model(model, xTest, yTest, framework='tensorflow', batch_size=512, is_ptl=False):
    """Evaluate model and return loss and accuracy (framework-agnostic).
    
    Args:
        model: Trained model
        xTest: Test data
        yTest: Test labels
        framework: 'tensorflow' or 'pytorch'
        batch_size: Batch size for PyTorch evaluation to avoid OOM
        is_ptl: Whether the model is a PyTorch Lite (.ptl) model
    
    Returns:
        tuple: (test_loss, test_acc)
    """
    if framework == 'tensorflow':
        test_loss, test_acc = model.evaluate(xTest, yTest, verbose=0)
        return test_loss, test_acc
    elif framework == 'pytorch':
        import torch
        import torch.nn as nn
        
        criterion = nn.CrossEntropyLoss()
        
        if is_ptl:
            # PyTorch Lite model evaluation
            # PTL models expect shape (batch, channels, length) but xTest has (batch, length, channels)
            # Process in batches for speed
            # PTL models use mobile-optimized prepacked operators that only support CPU
            device = torch.device('cpu')
            model = model.to(device)
            
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_samples = len(xTest)
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            print(f"Evaluating PTL model on CPU: {n_samples} samples in {n_batches} batches...")
            
            for batch_idx, i in enumerate(range(0, n_samples, batch_size)):
                batch_end = min(i + batch_size, n_samples)
                xTest_batch = xTest[i:batch_end]
                yTest_batch = yTest[i:batch_end]
                
                # Convert to tensors if needed
                if not isinstance(xTest_batch, torch.Tensor):
                    xTest_tensor = torch.from_numpy(xTest_batch).float().to(device)
                    yTest_tensor = torch.from_numpy(yTest_batch).long().to(device)
                else:
                    xTest_tensor = xTest_batch.to(device)
                    yTest_tensor = yTest_batch.to(device)
                
                # Transpose to match PTL model's expected input: (batch, channels, length)
                # xTest has shape (batch, length, channels), need (batch, channels, length)
                if len(xTest_tensor.shape) == 3:
                    xTest_tensor = xTest_tensor.permute(0, 2, 1)
                
                outputs = model(xTest_tensor)
                loss = criterion(outputs, yTest_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
                batch_samples = yTest_tensor.size(0)
                total_loss += loss.item() * batch_samples
                total_correct += (predicted == yTest_tensor).sum().item()
                total_samples += batch_samples
                
                # Progress update every 10% or every 10 batches, whichever is more frequent
                progress_interval = max(1, min(10, n_batches // 10))
                if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == n_batches:
                    progress_pct = 100 * (batch_idx + 1) / n_batches
                    print(f"  PTL evaluation progress: {batch_idx + 1}/{n_batches} batches ({progress_pct:.1f}%)")
                
                # Clean up memory
                del xTest_tensor, yTest_tensor, outputs, predicted, loss
            
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            
            return avg_loss, accuracy
        else:
            # Regular PyTorch model evaluation
            model.eval()
            device = next(model.parameters()).device
            
            # Process in batches to avoid OOM
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_samples = len(xTest)
            
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    xTest_batch = xTest[i:batch_end]
                    yTest_batch = yTest[i:batch_end]
                    
                    # Convert to tensors if needed
                    if not isinstance(xTest_batch, torch.Tensor):
                        xTest_tensor = torch.from_numpy(xTest_batch).float().to(device)
                        yTest_tensor = torch.from_numpy(yTest_batch).long().to(device)
                    else:
                        xTest_tensor = xTest_batch.to(device)
                        yTest_tensor = yTest_batch.to(device)
                
                outputs = model(xTest_tensor)
                loss = criterion(outputs, yTest_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
                batch_samples = yTest_tensor.size(0)
                total_loss += loss.item() * batch_samples
                total_correct += (predicted == yTest_tensor).sum().item()
                total_samples += batch_samples
                
                # Clean up GPU memory after each batch
                del xTest_tensor, yTest_tensor, outputs, predicted, loss
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    else:
        raise ValueError(f"Unknown framework: {framework}")


def predict_model(model, xTest, framework='tensorflow', batch_size=512, is_ptl=False):
    """Get prediction probabilities from model (framework-agnostic).
    
    Args:
        model: Trained model
        xTest: Test data
        framework: 'tensorflow' or 'pytorch'
        batch_size: Batch size for PyTorch inference to avoid OOM
        is_ptl: Whether the model is a PyTorch Lite (.ptl) model
    
    Returns:
        numpy array of prediction probabilities, shape (n_samples, n_classes)
    """
    if framework == 'tensorflow':
        return model.predict(xTest, verbose=0)
    elif framework == 'pytorch':
        import torch
        
        if is_ptl:
            # PyTorch Lite model - use batching for speed
            # PTL models expect shape (batch, channels, length) but xTest has (batch, length, channels)
            # PTL models use mobile-optimized prepacked operators that only support CPU
            device = torch.device('cpu')
            model = model.to(device)
            
            all_probs = []
            n_samples = len(xTest)
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            print(f"Generating PTL model predictions on CPU: {n_samples} samples in {n_batches} batches...")
            
            for batch_idx, i in enumerate(range(0, n_samples, batch_size)):
                batch_end = min(i + batch_size, n_samples)
                xTest_batch = xTest[i:batch_end]
                
                # Convert to tensor if needed
                if not isinstance(xTest_batch, torch.Tensor):
                    xTest_tensor = torch.from_numpy(xTest_batch).float().to(device)
                else:
                    xTest_tensor = xTest_batch.to(device)
                
                # Transpose to match PTL model's expected input: (batch, channels, length)
                # xTest has shape (batch, length, channels), need (batch, channels, length)
                if len(xTest_tensor.shape) == 3:
                    xTest_tensor = xTest_tensor.permute(0, 2, 1)
                
                outputs = model(xTest_tensor)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.numpy())
                
                # Progress update every 10% or every 10 batches, whichever is more frequent
                progress_interval = max(1, min(10, n_batches // 10))
                if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == n_batches:
                    progress_pct = 100 * (batch_idx + 1) / n_batches
                    print(f"  PTL prediction progress: {batch_idx + 1}/{n_batches} batches ({progress_pct:.1f}%)")
                
                del xTest_tensor, outputs, probs
            
            return np.vstack(all_probs)
        else:
            # Regular PyTorch model
            model.eval()
            device = next(model.parameters()).device
            
            # Process in batches to avoid OOM
            all_probs = []
            n_samples = len(xTest)
            
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    xTest_batch = xTest[i:batch_end]
                    
                    # Convert to tensor if needed
                    if not isinstance(xTest_batch, torch.Tensor):
                        xTest_tensor = torch.from_numpy(xTest_batch).float().to(device)
                    else:
                        xTest_tensor = xTest_batch.to(device)
                    
                    outputs = model(xTest_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy())
                    
                    # Clean up GPU memory after each batch
                    del xTest_tensor, outputs, probs
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return np.vstack(all_probs)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def testModel(configObj, dataDir='.', balanced=True, debug=False, testDataCsv=None, test_ptl=False, outputDir=None, titlePrefix=None):
    TAG = "nnTester.testModel()"
    print("____%s____" % (TAG))
    
    # Detect framework
    framework = nnTrainer.get_framework_from_config(configObj)
    print(f"{TAG}: Using framework: {framework}")
    
    modelFnameRoot = libosd.configUtils.getConfigParam("modelFname", configObj['modelConfig'])
    nnModelClassName = libosd.configUtils.getConfigParam("modelClass", configObj['modelConfig'])
    
    # If outputDir not specified, use dataDir for outputs
    if outputDir is None:
        outputDir = dataDir
    else:
        print(f"{TAG}: Using separate output directory: {outputDir}")
        os.makedirs(outputDir, exist_ok=True)
    
    # If titlePrefix not specified, use modelFnameRoot
    if titlePrefix is None:
        titlePrefix = modelFnameRoot
    else:
        print(f"{TAG}: Using title prefix: {titlePrefix}")
    
    # If testDataCsv is explicitly provided, use it directly
    if testDataCsv is not None:
        testDataPath = testDataCsv if os.path.isabs(testDataCsv) else os.path.join(dataDir, testDataCsv)
        if not os.path.exists(testDataPath):
            raise ValueError(f"{TAG}: Specified test data file not found: {testDataPath}")
        testDataFname = os.path.basename(testDataPath)
        print(f"{TAG}: Using specified test data CSV: {testDataPath}")
    else:
        # Resolve test data file path, preferring feature CSVs
        if (balanced):
            testDataFname = libosd.configUtils.getConfigParam("testBalancedFileCsv", configObj['dataFileNames'])
        else:   
            testDataFname = libosd.configUtils.getConfigParam("testFeaturesHistoryFileCsv", configObj['dataFileNames'])
        
        if testDataFname is None:
            raise ValueError(f"{TAG}: Test data filename is None. Check config for testBalancedFileCsv or testFeaturesHistoryFileCsv")

        # Build initial path
        testDataPath = os.path.join(dataDir, testDataFname) if isinstance(testDataFname, str) else None
        
        # Prefer feature CSV if it exists
        try:
            testFeaturesName = configObj['dataFileNames'].get('testFeaturesFileCsv')
            if isinstance(testFeaturesName, str):
                candidate = os.path.join(dataDir, testFeaturesName)
                if os.path.exists(candidate):
                    print(f"{TAG}: Using test features CSV {candidate}")
                    testDataPath = candidate
                    testDataFname = testFeaturesName
        except Exception:
            pass
        
        if testDataPath is None or not os.path.exists(testDataPath):
            raise ValueError(f"{TAG}: Test data file not found: {testDataPath}")

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
    print("%s: Loading Test Data from File %s" % (TAG, testDataFname))
    df_original = augmentData.loadCsv(testDataPath, debug=debug)
    print("%s: Loaded %d datapoints" % (TAG, len(df_original)))

    # Process data and track which rows are kept
    # df2trainingData may skip some rows where dp2vector returns None
    print("%s: Re-formatting data for testing" % (TAG))
    xTest_list, yTest_list, kept_indices = [], [], []
    
    # We need to replicate df2trainingData logic but track indices
    cols = list(df_original.columns)
    # Try with _t-0 suffix first (feature history enabled)
    m_cols = [c for c in cols if isinstance(c, str) and c.startswith('M') and c.endswith('_t-0')]
    if len(m_cols) == 0:
        # Try without suffix (feature history disabled, addFeatureHistoryLength=0)
        m_cols = [c for c in cols if isinstance(c, str) and c.startswith('M') and len(c) == 4 and c[1:].isdigit()]
    if len(m_cols) == 0:
        raise ValueError("No magnitude (Mxxx_t-0 or Mxxx) columns found in dataframe")
    m_indices = [cols.index(c) for c in m_cols]
    accStartCol = min(m_indices)
    accEndCol = max(m_indices) + 1
    
    try:
        hrCol = df_original.columns.get_loc('hr')
    except:
        hrCol = None
    typeCol = df_original.columns.get_loc('type')
    eventIdCol = df_original.columns.get_loc('eventId')
    
    lastEventId = None
    for idx in range(len(df_original)):
        rowArr = df_original.iloc[idx]
        
        # Reset buffer when switching to a new event
        eventId = rowArr.iloc[eventIdCol]
        if eventId != lastEventId:
            nnModel.resetAccBuf()
            lastEventId = eventId
        
        dpDict = {}
        accArr = rowArr.iloc[accStartCol:accEndCol].values.astype(float).tolist()
        dpDict['rawData'] = accArr
        if hrCol is not None:
            try:
                dpDict['hr'] = int(rowArr.iloc[hrCol])
            except:
                dpDict['hr'] = None
        else:
            dpDict['hr'] = None
        
        dpInputData = nnModel.dp2vector(dpDict, normalise=False)
        if dpInputData is not None:
            xTest_list.append(dpInputData)
            yTest_list.append(rowArr.iloc[typeCol])
            kept_indices.append(idx)
    
    # Filter dataframe to only rows that were kept (for model predictions)
    original_df_len = len(df_original)
    original_events = df_original['eventId'].nunique()
    
    # Debug: check OSD alarms before filtering
    if debug:
        print(f"{TAG}: Before filtering - OSD alarms (>=2): {(df_original['osdAlarmState'] >= 2).sum()}/{len(df_original)} datapoints")
        orig_osd_pred = (df_original['osdAlarmState'] >= 2).astype(int)
        for eid, grp in df_original.groupby('eventId'):
            evt_true = grp['type'].iloc[0]
            evt_osd_any = (grp['osdAlarmState'] >= 2).any()
            n_alarms = (grp['osdAlarmState'] >= 2).sum()
            if evt_true == 1:  # Only print seizure events
                print(f"{TAG}: Event {eid} (seizure): OSD alarms in {n_alarms}/{len(grp)} datapoints, any={evt_osd_any}")
    
    df = df_original.iloc[kept_indices].reset_index(drop=True)
    print(f"%s: Kept {len(kept_indices)} of {original_df_len} rows after filtering ({original_df_len - len(kept_indices)} removed)" % TAG)
    
    # Debug: check OSD alarms after filtering
    if debug:
        print(f"{TAG}: After filtering - OSD alarms (>=2): {(df['osdAlarmState'] >= 2).sum()}/{len(df)} datapoints")
        for eid, grp in df.groupby('eventId'):
            evt_true = grp['type'].iloc[0]
            evt_osd_any = (grp['osdAlarmState'] >= 2).any()
            n_alarms = (grp['osdAlarmState'] >= 2).sum()
            if evt_true == 1:  # Only print seizure events
                print(f"{TAG}: Event {eid} (seizure): OSD alarms in {n_alarms}/{len(grp)} datapoints after filter, any={evt_osd_any}")

    print("%s: Converting to np arrays" % (TAG))
    xTest = np.array(xTest_list)
    yTest = np.array(yTest_list)

    print("%s: re-shaping array for testing" % (TAG))
    if (inputDims == 1):
        xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
    elif (inputDims ==2):
        xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], xTest.shape[2], 1))
    else:
        print("ERROR - inputDims out of Range: %d" % inputDims)
        exit(-1)

    # Load the model once
    modelFnamePath = os.path.join(dataDir, modelFname)
    
    # For PyTorch models, check if both .pt and .ptl exist (only if test_ptl=True)
    models_to_test = [('pt', modelFnamePath, False)]  # List of (label, path, is_ptl)
    
    if test_ptl and framework == 'pytorch' and modelFnamePath.endswith('.pt'):
        ptl_model_path = modelFnamePath.replace('.pt', '.ptl')
        if os.path.exists(ptl_model_path):
            print(f"{TAG}: Found both .pt and .ptl models - will test both for comparison")
            models_to_test.append(('ptl', ptl_model_path, True))
        elif os.path.exists(modelFnamePath):
            # .pt exists but .ptl doesn't - generate .ptl for testing
            print(f"{TAG}: .ptl model not found, generating from .pt model...")
            try:
                # Import convertPt2Ptl function
                try:
                    from user_tools.nnTraining2.convertPt2Ptl import convert_pt_to_ptl
                except ImportError:
                    from convertPt2Ptl import convert_pt_to_ptl
                
                # Get input shape - use a reasonable default if we don't have xTest yet
                # We'll use the standard shape for this application
                input_shape = (1, 1, 750)  # Standard batch, channels, sequence length
                
                # Convert to .ptl
                success = convert_pt_to_ptl(
                    input_path=modelFnamePath,
                    output_path=ptl_model_path,
                    input_shape=input_shape,
                    num_classes=2,  # Binary classification
                    verbose=True
                )
                
                if success:
                    print(f"{TAG}: Successfully generated {ptl_model_path}")
                    print(f"{TAG}: Will test both .pt and .ptl models for comparison")
                    models_to_test.append(('ptl', ptl_model_path, True))
                else:
                    print(f"{TAG}: Warning - Failed to generate .ptl model, will only test .pt model")
            except Exception as e:
                print(f"{TAG}: Warning - Could not generate .ptl model: {e}")
                print(f"{TAG}: Will only test .pt model")
                import traceback
                traceback.print_exc()
    elif not test_ptl and framework == 'pytorch':
        print(f"{TAG}: Skipping .ptl model testing (test_ptl=False)")
    
    # Store results for all models tested
    all_model_results = {}
    
    for model_label, model_path, is_ptl in models_to_test:
        print(f"\n{'='*70}")
        print(f"{TAG}: Testing {model_label.upper()} model: {os.path.basename(model_path)}")
        print(f"{'='*70}\n")
        
        model = load_model_for_testing(model_path, nnModel, framework)

        # Get prediction probabilities
        print("%s: Calculating Seizure probabilities from test data for %s model" % (TAG, model_label.upper()))
        prediction_proba = predict_model(model, xTest, framework, is_ptl=is_ptl)
        if (debug): print("prediction_proba=",prediction_proba)

        # Prediction classes
        prediction = np.argmax(prediction_proba, axis=1)
        if (debug): print("prediction=", prediction)
        
        # Calculate metrics from predictions (more efficient than separate evaluation pass)
        print("Testing using %d seizure datapoints and %d false alarm datapoints"
            % (np.count_nonzero(yTest == 1),
            np.count_nonzero(yTest == 0)))
        
        # Calculate accuracy
        test_acc = np.mean(prediction == yTest)
        
        # Calculate loss (cross-entropy)
        import torch.nn.functional as F
        import torch
        if framework == 'pytorch':
            # Use PyTorch's cross-entropy on the probabilities
            proba_tensor = torch.from_numpy(prediction_proba).float()
            yTest_tensor = torch.from_numpy(yTest).long()
            test_loss = F.cross_entropy(proba_tensor, yTest_tensor).item()
        else:
            # TensorFlow: manual cross-entropy calculation
            epsilon = 1e-7
            yTest_one_hot = np.eye(prediction_proba.shape[1])[yTest]
            test_loss = -np.mean(np.sum(yTest_one_hot * np.log(prediction_proba + epsilon), axis=1))
        
        print(f"{model_label.upper()} Model - Test accuracy: {test_acc:.6f}")
        print(f"{model_label.upper()} Model - Test loss: {test_loss:.6f}")

        # Store results for this model
        all_model_results[model_label] = {
            'model': model,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'prediction_proba': prediction_proba,
            'prediction': prediction,
            'is_ptl': is_ptl
        }
    
    # Use the first model (.pt) for the rest of the analysis (maintains backward compatibility)
    model = all_model_results['pt']['model']
    test_loss = all_model_results['pt']['test_loss']
    test_acc = all_model_results['pt']['test_acc']
    prediction_proba = all_model_results['pt']['prediction_proba']
    prediction = all_model_results['pt']['prediction']

    pSeizure = prediction_proba[:,1]
    seq = range(0, len(pSeizure))
    # Colour seizure data points red, and non-seizure data blue
    colours = ['red' if seizureVal==1 else 'blue' for seizureVal in yTest]

    # Calculate statistics at different thresholds
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

    # Create probability scatter plot
    fig, ax = plt.subplots(3,1)
    ax[0].title.set_text("%s: Seizure Probabilities" % titlePrefix)
    ax[0].set_ylabel('Probability')
    ax[0].set_xlabel('Datapoint')
    ax[0].scatter(seq, pSeizure, s=2.0, marker='x', c=colours)
    ax[1].plot(yTest)
    fname = os.path.join(outputDir, "%s_probabilities.png" % modelFnameRoot)
    fig.savefig(fname)
    plt.close()

    # Calculate and save confusion matrix and detailed statistics
    calcConfusionMatrix(configObj, modelFnameRoot, xTest, yTest, dataDir=outputDir, balanced=balanced, debug=debug, titlePrefix=titlePrefix)

    # Calculate epoch-level statistics
    # Check if yTest is one-hot encoded (2D) or class indices (1D)
    if len(yTest.shape) > 1 and yTest.shape[1] > 1:
        y_true = np.argmax(yTest, axis=1)
    else:
        y_true = yTest.flatten()
    y_pred = prediction
    
    # Epoch-level confusion matrix and metrics
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    tpr, fpr = fpr_score(y_true, y_pred)
    
    # Calculate OSD algorithm predictions from dataframe
    df['pred'] = y_pred
    df['osd_pred'] = df['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)
    yPredOsd = df['osd_pred'].values
    yTestOsd = y_true
    
    tprOsd, fprOsd = fpr_score(yTestOsd, yPredOsd)
    cmOsd = sklearn.metrics.confusion_matrix(yTestOsd, yPredOsd, labels=[0, 1])
    tnOsd, fpOsd, fnOsd, tpOsd = cmOsd.ravel()
    accuracyOsd = sklearn.metrics.accuracy_score(yTestOsd, yPredOsd)
    
    # For event-level OSD statistics, use the ORIGINAL dataframe (before filtering)
    # This is important because dp2vector may filter out datapoints that contain OSD alarms
    # We still use the filtered df for model predictions since those require valid dp2vector output
    df_original['osd_pred_orig'] = df_original['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)
    
    # Event-level statistics - iterate over ALL events in original data to ensure we don't miss any
    # This is critical because some events may be completely filtered out by dp2vector
    event_stats = []
    for eventId, group_orig in df_original.groupby('eventId'):
        true_label = group_orig['type'].iloc[0]
        
        # For OSD event prediction, use the ORIGINAL unfiltered data
        osd_event_pred = 1 if (group_orig['osd_pred_orig'] == 1).any() else 0
        
        # For model predictions, use the filtered data (if this event exists in filtered df)
        group_filtered = df[df['eventId'] == eventId]
        if len(group_filtered) > 0:
            model_event_pred = 1 if (group_filtered['pred'] == 1).any() else 0
            # Calculate max seizure probability for this event using the correct row indices
            # We need to map back to the kept_indices to index into prediction_proba correctly
            max_prob = prediction_proba[group_filtered.index, 1].max()
        else:
            # This event was completely filtered out - model cannot make a prediction
            model_event_pred = 0
            max_prob = 0.0
        
        # Debug event-level OSD predictions
        if debug and true_label == 1:  # Print for seizure events
            n_osd_alarms_original = (group_orig['osd_pred_orig'] == 1).sum()
            n_osd_alarms_filtered = (group_filtered['osd_pred'] == 1).sum() if len(group_filtered) > 0 else 0
            print(f"{TAG}: Event {eventId} (seizure): model_pred={model_event_pred}, " +
                  f"osd_pred={osd_event_pred} (filtered:{n_osd_alarms_filtered}/{len(group_filtered)}, original:{n_osd_alarms_original}/{len(group_orig)})")
        
        event_stats.append({
            'eventId': eventId,
            'true_label': true_label,
            'model_pred': model_event_pred,
            'osd_pred': osd_event_pred,
            'max_seizure_prob': max_prob
        })
    event_stats_df = pd.DataFrame(event_stats)
    
    # Load event metadata from allData.json for additional details
    # allData.json is at the training output root, not in the fold subdirectory
    # Search up the directory tree to find it
    allDataFilename = configObj['dataFileNames']['allDataFileJson']
    allDataPath = None
    
    # First try in the current dataDir
    candidate_path = os.path.join(dataDir, allDataFilename)
    if os.path.exists(candidate_path):
        allDataPath = candidate_path
    else:
        # Search up the directory tree (for fold subdirectories)
        current_dir = dataDir
        for _ in range(5):  # Search up to 5 levels up
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached root directory
                break
            candidate_path = os.path.join(parent_dir, allDataFilename)
            if os.path.exists(candidate_path):
                allDataPath = candidate_path
                break
            current_dir = parent_dir
    
    event_details_map = {}
    
    if allDataPath and os.path.exists(allDataPath):
        print("%s: Loading event details from %s" % (TAG, allDataPath))
        try:
            with open(allDataPath, 'r') as f:
                allData = json.load(f)
            
            # Build a map of eventId -> event details
            print(f"{TAG}: Parsing event details from allData")
            events_list = allData if isinstance(allData, list) else allData.get('events', [])
            print(f"{TAG}: Found {len(events_list)} events in allData")
            print(f"{TAG}: Event keys are: {list(events_list[0].keys()) if len(events_list) > 0 else 'N/A'} ")
            for event in events_list:
                # Convert event ID to string for consistent type matching
                event_id_str = str(event['id'])
                event_details_map[event_id_str] = {
                    'userId': event.get('userId', 'N/A'),
                    'typeStr': event.get('type', 'N/A'),
                    'subType': event.get('subType', 'N/A'),
                    'desc': event.get('desc', 'N/A')
                }
            print(f"{TAG}: Loaded metadata for {len(event_details_map)} events from {allDataPath}")
        except Exception as e:
            print(f"{TAG}: Warning - Could not load event details from {allDataPath}: {e}")
    else:
        print(f"{TAG}: Warning - allData file not found (searched in {dataDir} and parent directories)")
    
    # Enrich event_stats_df with metadata - convert eventId to string for consistent lookup
    event_stats_df['userId'] = event_stats_df['eventId'].map(lambda eid: event_details_map.get(str(eid), {}).get('userId', 'N/A'))
    event_stats_df['typeStr'] = event_stats_df['eventId'].map(lambda eid: event_details_map.get(str(eid), {}).get('typeStr', 'N/A'))
    event_stats_df['subType'] = event_stats_df['eventId'].map(lambda eid: event_details_map.get(str(eid), {}).get('subType', 'N/A'))
    event_stats_df['desc'] = event_stats_df['eventId'].map(lambda eid: event_details_map.get(str(eid), {}).get('desc', 'N/A'))
    
    # Save detailed event results to CSV
    event_results_csv = event_stats_df[['eventId', 'userId', 'typeStr', 'subType', 'true_label', 
                                         'model_pred', 'osd_pred', 'max_seizure_prob', 'desc']].copy()
    event_results_csv.columns = ['EventID', 'UserID', 'Type', 'SubType', 'ActualLabel', 
                                   'ModelPrediction', 'OSDPrediction', 'MaxSeizureProbability', 'Description']
    
    csv_path = os.path.join(dataDir, f'{modelFnameRoot}_event_results.csv')
    event_results_csv.to_csv(csv_path, index=False)
    print(f"{TAG}: Event-level results saved to {csv_path}")
    
    # Event-level metrics
    event_tpr, event_fpr = fpr_score(event_stats_df['true_label'], event_stats_df['model_pred'])
    event_cm = sklearn.metrics.confusion_matrix(event_stats_df['true_label'], event_stats_df['model_pred'], labels=[0, 1])
    event_tn, event_fp, event_fn, event_tp = event_cm.ravel()
    
    osd_event_tpr, osd_event_fpr = fpr_score(event_stats_df['true_label'], event_stats_df['osd_pred'])
    osd_event_cm = sklearn.metrics.confusion_matrix(event_stats_df['true_label'], event_stats_df['osd_pred'], labels=[0, 1])
    osd_event_tn, osd_event_fp, osd_event_fn, osd_event_tp = osd_event_cm.ravel()
    
    # Debug: Print OSD event-level predictions summary
    if debug:
        print(f"\n{TAG}: === OSD EVENT-LEVEL DEBUG ===")
        print(f"{TAG}: Total events: {len(event_stats_df)}")
        print(f"{TAG}: Seizure events (true_label=1): {(event_stats_df['true_label']==1).sum()}")
        print(f"{TAG}: Events with OSD prediction=1: {(event_stats_df['osd_pred']==1).sum()}")
        print(f"{TAG}: OSD Event TP={osd_event_tp}, FP={osd_event_fp}, FN={osd_event_fn}, TN={osd_event_tn}")
        print(f"{TAG}: OSD Event TPR={osd_event_tpr:.3f}, FPR={osd_event_fpr:.3f}")
        if osd_event_tp == 0 and (event_stats_df['true_label']==1).sum() > 0:
            print(f"{TAG}: WARNING: OSD detected 0 seizure events despite having seizure events in test set!")
            print(f"{TAG}: This may indicate that OSD alarms were filtered out during dp2vector processing.")
    
    # Convert NumPy scalars to native Python types
    def py(v):
        return v.item() if hasattr(v, 'item') else v
    
    # Build results dictionary
    num_positive_epoch = int((y_true == 1).sum())
    num_positive_event = int((event_stats_df['true_label'] == 1).sum())
    
    foldResults = {
        'num_positive_epoch': num_positive_epoch,
        'num_positive_event': num_positive_event,
        'accuracy': py(accuracy),
        'accuracyOsd': py(accuracyOsd),
        'tpr': py(tpr),
        'fpr': py(fpr),
        'tprOsd': py(tprOsd),
        'fprOsd': py(fprOsd),
        'tn': py(tn),
        'fp': py(fp),
        'fn': py(fn),
        'tp': py(tp),
        'tnOsd': py(tnOsd),
        'fpOsd': py(fpOsd),
        'fnOsd': py(fnOsd),
        'tpOsd': py(tpOsd),
        'event_tpr': py(event_tpr),
        'event_fpr': py(event_fpr),
        'event_tp': py(event_tp),
        'event_fp': py(event_fp),
        'event_fn': py(event_fn),
        'event_tn': py(event_tn),
        'osd_event_tpr': py(osd_event_tpr),
        'osd_event_fpr': py(osd_event_fpr),
        'osd_event_tp': py(osd_event_tp),
        'osd_event_fp': py(osd_event_fp),
        'osd_event_fn': py(osd_event_fn),
        'osd_event_tn': py(osd_event_tn)
    }
    
    # Save to JSON
    json_path = os.path.join(dataDir, 'testResults.json')
    with open(json_path, 'w') as f:
        json.dump(foldResults, f, indent=2)
    print(f"nnTester: foldResults written to {json_path}")
    
    # Echo formatted results to console
    print("\n===== Formatted foldResults =====")
    print(json.dumps(foldResults, indent=2))
    print("===== End foldResults =====\n")
    
    # Print detailed event-based metrics summary
    print("\n" + "="*70)
    print("EVENT-BASED ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nTotal Events: {len(event_stats_df)}")
    print(f"  Seizure Events: {num_positive_event}")
    print(f"  Non-Seizure Events: {len(event_stats_df) - num_positive_event}")
    
    print(f"\n{'METRIC':<30} {'MODEL':<15} {'OSD ALGORITHM':<15}")
    print("-" * 70)
    print(f"{'True Positives (TP)':<30} {py(event_tp):<15} {py(osd_event_tp):<15}")
    print(f"{'False Positives (FP)':<30} {py(event_fp):<15} {py(osd_event_fp):<15}")
    print(f"{'True Negatives (TN)':<30} {py(event_tn):<15} {py(osd_event_tn):<15}")
    print(f"{'False Negatives (FN)':<30} {py(event_fn):<15} {py(osd_event_fn):<15}")
    print("-" * 70)
    print(f"{'Sensitivity (TPR)':<30} {py(event_tpr):.4f}{'':<10} {py(osd_event_tpr):.4f}{'':<10}")
    print(f"{'False Alarm Rate (FAR/FPR)':<30} {py(event_fpr):.4f}{'':<10} {py(osd_event_fpr):.4f}{'':<10}")
    
    # Calculate additional event-based metrics
    event_precision = event_tp / (event_tp + event_fp) if (event_tp + event_fp) > 0 else 0
    event_specificity = event_tn / (event_tn + event_fp) if (event_tn + event_fp) > 0 else 0
    event_f1 = 2 * event_tp / (2 * event_tp + event_fp + event_fn) if (2 * event_tp + event_fp + event_fn) > 0 else 0
    
    osd_event_precision = osd_event_tp / (osd_event_tp + osd_event_fp) if (osd_event_tp + osd_event_fp) > 0 else 0
    osd_event_specificity = osd_event_tn / (osd_event_tn + osd_event_fp) if (osd_event_tn + osd_event_fp) > 0 else 0
    osd_event_f1 = 2 * osd_event_tp / (2 * osd_event_tp + osd_event_fp + osd_event_fn) if (2 * osd_event_tp + osd_event_fp + osd_event_fn) > 0 else 0
    
    print(f"{'Precision (PPV)':<30} {event_precision:.4f}{'':<10} {osd_event_precision:.4f}{'':<10}")
    print(f"{'Specificity (TNR)':<30} {event_specificity:.4f}{'':<10} {osd_event_specificity:.4f}{'':<10}")
    print(f"{'F1 Score':<30} {event_f1:.4f}{'':<10} {osd_event_f1:.4f}{'':<10}")
    print("="*70)
    
    # Event-based threshold analysis
    print("\n" + "="*70)
    print("EVENT-BASED THRESHOLD ANALYSIS")
    print("="*70)
    
    # Calculate event-level TPR/FPR at different thresholds
    event_threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    event_tpr_list = []
    event_fpr_list = []
    event_tp_list = []
    event_fp_list = []
    event_tn_list = []
    event_fn_list = []
    
    for threshold in event_threshold_list:
        # For each event, classify as positive if max_seizure_prob >= threshold
        event_preds_at_threshold = (event_stats_df['max_seizure_prob'] >= threshold).astype(int)
        event_true_labels = event_stats_df['true_label'].values
        
        # Calculate confusion matrix for this threshold
        event_cm_th = sklearn.metrics.confusion_matrix(event_true_labels, event_preds_at_threshold, labels=[0, 1])
        event_tn_th, event_fp_th, event_fn_th, event_tp_th = event_cm_th.ravel()
        
        # Calculate TPR and FPR
        event_tpr_th = event_tp_th / (event_tp_th + event_fn_th) if (event_tp_th + event_fn_th) > 0 else 0
        event_fpr_th = event_fp_th / (event_fp_th + event_tn_th) if (event_fp_th + event_tn_th) > 0 else 0
        
        event_tpr_list.append(event_tpr_th)
        event_fpr_list.append(event_fpr_th)
        event_tp_list.append(int(event_tp_th))
        event_fp_list.append(int(event_fp_th))
        event_tn_list.append(int(event_tn_th))
        event_fn_list.append(int(event_fn_th))
    
    print(f"\n{'Threshold':<12} {'TPR':<12} {'FPR':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-" * 70)
    for i, th in enumerate(event_threshold_list):
        print(f"{th:<12.1f} {event_tpr_list[i]:<12.4f} {event_fpr_list[i]:<12.4f} "
              f"{event_tp_list[i]:<8} {event_fp_list[i]:<8} {event_tn_list[i]:<8} {event_fn_list[i]:<8}")
    
    # Create event-based threshold analysis plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: TPR and FPR vs Threshold
    axes[0].plot(event_threshold_list, event_tpr_list, 'o-', color='green', linewidth=2, markersize=8, label='TPR (Sensitivity)')
    axes[0].plot(event_threshold_list, event_fpr_list, 's-', color='red', linewidth=2, markersize=8, label='FPR (False Alarm Rate)')
    axes[0].set_xlabel('Threshold', fontsize=12)
    axes[0].set_ylabel('Rate', fontsize=12)
    axes[0].set_title('Event-Based TPR and FPR vs Threshold', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])
    
    # Add text annotations for key points
    for i, th in enumerate(event_threshold_list):
        if th in [0.3, 0.5, 0.7]:  # Annotate key thresholds
            axes[0].annotate(f'{event_tpr_list[i]:.2f}', 
                           xy=(th, event_tpr_list[i]), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=9,
                           color='green')
            axes[0].annotate(f'{event_fpr_list[i]:.2f}', 
                           xy=(th, event_fpr_list[i]), 
                           xytext=(5, -15), 
                           textcoords='offset points',
                           fontsize=9,
                           color='red')
    
    # Plot 2: ROC-style curve (FPR vs TPR)
    # Sort by FPR for proper ROC curve
    sorted_indices = np.argsort(event_fpr_list)
    sorted_fpr = [event_fpr_list[i] for i in sorted_indices]
    sorted_tpr = [event_tpr_list[i] for i in sorted_indices]
    sorted_th = [event_threshold_list[i] for i in sorted_indices]
    
    axes[1].plot(sorted_fpr, sorted_tpr, 'o-', color='blue', linewidth=2, markersize=8)
    axes[1].plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Random Classifier')
    axes[1].set_xlabel('False Positive Rate (FPR)', fontsize=12)
    axes[1].set_ylabel('True Positive Rate (TPR)', fontsize=12)
    axes[1].set_title('Event-Based ROC Curve', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    
    # Annotate points with threshold values
    for i, (fpr_val, tpr_val, th_val) in enumerate(zip(sorted_fpr, sorted_tpr, sorted_th)):
        if th_val in [0.3, 0.5, 0.7]:  # Annotate key thresholds
            axes[1].annotate(f'th={th_val}', 
                           xy=(fpr_val, tpr_val), 
                           xytext=(10, -10), 
                           textcoords='offset points',
                           fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                           arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    plt.tight_layout()
    threshold_plot_path = os.path.join(dataDir, f'{modelFnameRoot}_event_threshold_analysis.png')
    fig.savefig(threshold_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n{TAG}: Event-based threshold analysis plot saved to {threshold_plot_path}")
    
    # Save threshold analysis data to JSON
    threshold_data = {
        'thresholds': event_threshold_list,
        'tpr': event_tpr_list,
        'fpr': event_fpr_list,
        'tp': event_tp_list,
        'fp': event_fp_list,
        'tn': event_tn_list,
        'fn': event_fn_list
    }
    
    threshold_json_path = os.path.join(dataDir, f'{modelFnameRoot}_event_threshold_data.json')
    with open(threshold_json_path, 'w') as f:
        json.dump(threshold_data, f, indent=2)
    print(f"{TAG}: Event-based threshold data saved to {threshold_json_path}")
    
    print("="*70)
    
    print("\nEvent-Level Confusion Matrix (Model):")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {py(event_tn):<10}        {py(event_fp):<10}")
    print(f"Actual Positive        {py(event_fn):<10}        {py(event_tp):<10}")
    
    print("\nEvent-Level Confusion Matrix (OSD Algorithm):")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {py(osd_event_tn):<10}        {py(osd_event_fp):<10}")
    print(f"Actual Positive        {py(osd_event_fn):<10}        {py(osd_event_tp):<10}")
    print("="*70 + "\n")
    
    # Plot event-based confusion matrices
    import seaborn as sns
    LABELS = ['Non-Seizure', 'Seizure']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Model event-level confusion matrix
    sns.heatmap(event_cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                linewidths=0.1, fmt="d", cmap='YlGnBu', ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f"{titlePrefix}: Event-Level Confusion Matrix\n(Model)", fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    
    # Add metrics text below model confusion matrix
    model_text = f"Sensitivity: {py(event_tpr):.3f}  Specificity: {event_specificity:.3f}\n"
    model_text += f"Precision: {event_precision:.3f}  F1: {event_f1:.3f}"
    ax1.text(0.5, -0.15, model_text, ha='center', va='top', transform=ax1.transAxes,
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # OSD algorithm event-level confusion matrix
    sns.heatmap(osd_event_cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                linewidths=0.1, fmt="d", cmap='OrRd', ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_title(f"{titlePrefix}: Event-Level Confusion Matrix\n(OSD Algorithm)", fontsize=13, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=11)
    ax2.set_xlabel('Predicted Label', fontsize=11)
    
    # Add metrics text below OSD confusion matrix
    osd_text = f"Sensitivity: {py(osd_event_tpr):.3f}  Specificity: {osd_event_specificity:.3f}\n"
    osd_text += f"Precision: {osd_event_precision:.3f}  F1: {osd_event_f1:.3f}"
    ax2.text(0.5, -0.15, osd_text, ha='center', va='top', transform=ax2.transAxes,
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    fname_event_cm = os.path.join(outputDir, f"{modelFnameRoot}_event_confusion.png")
    plt.savefig(fname_event_cm, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Event-level confusion matrices saved as {fname_event_cm}")
    
    # If both .pt and .ptl models were tested, generate comparison plots and statistics
    if len(all_model_results) > 1 and 'ptl' in all_model_results:
        print(f"\n{'='*70}")
        print(f"{TAG}: GENERATING COMPARISON BETWEEN .PT AND .PTL MODELS")
        print(f"{'='*70}\n")
        
        # Calculate statistics for both models
        model_comparison = {}
        for model_label in ['pt', 'ptl']:
            pred_proba = all_model_results[model_label]['prediction_proba']
            pred = all_model_results[model_label]['prediction']
            
            # Epoch-level metrics
            if len(yTest.shape) > 1 and yTest.shape[1] > 1:
                y_true_model = np.argmax(yTest, axis=1)
            else:
                y_true_model = yTest.flatten()
            
            cm_model = sklearn.metrics.confusion_matrix(y_true_model, pred, labels=[0, 1])
            tn_m, fp_m, fn_m, tp_m = cm_model.ravel()
            acc_m = sklearn.metrics.accuracy_score(y_true_model, pred)
            tpr_m, fpr_m = fpr_score(y_true_model, pred)
            
            # Event-level metrics (using same event_stats_df logic)
            df_temp = df.copy()
            df_temp['pred_model'] = pred
            event_stats_model = []
            for eventId, group_orig in df_original.groupby('eventId'):
                true_label_ev = group_orig['type'].iloc[0]
                group_filtered_temp = df_temp[df_temp['eventId'] == eventId]
                if len(group_filtered_temp) > 0:
                    model_event_pred_temp = 1 if (group_filtered_temp['pred_model'] == 1).any() else 0
                    max_prob_temp = pred_proba[group_filtered_temp.index, 1].max()
                else:
                    model_event_pred_temp = 0
                    max_prob_temp = 0.0
                event_stats_model.append({
                    'eventId': eventId,
                    'true_label': true_label_ev,
                    'model_pred': model_event_pred_temp,
                    'max_seizure_prob': max_prob_temp
                })
            event_stats_df_model = pd.DataFrame(event_stats_model)
            
            event_tpr_m, event_fpr_m = fpr_score(event_stats_df_model['true_label'], event_stats_df_model['model_pred'])
            event_cm_m = sklearn.metrics.confusion_matrix(event_stats_df_model['true_label'], event_stats_df_model['model_pred'], labels=[0, 1])
            event_tn_m, event_fp_m, event_fn_m, event_tp_m = event_cm_m.ravel()
            
            model_comparison[model_label] = {
                'test_loss': all_model_results[model_label]['test_loss'],
                'test_acc': all_model_results[model_label]['test_acc'],
                'epoch_acc': acc_m,
                'epoch_tpr': tpr_m,
                'epoch_fpr': fpr_m,
                'epoch_tp': tp_m,
                'epoch_fp': fp_m,
                'epoch_tn': tn_m,
                'epoch_fn': fn_m,
                'event_tpr': event_tpr_m,
                'event_fpr': event_fpr_m,
                'event_tp': event_tp_m,
                'event_fp': event_fp_m,
                'event_tn': event_tn_m,
                'event_fn': event_fn_m,
                'event_cm': event_cm_m
            }
        
        # Print comparison table
        print("\n" + "="*80)
        print("MODEL COMPARISON: .PT vs .PTL")
        print("="*80)
        print(f"{'METRIC':<35} {'.PT MODEL':<20} {'.PTL MODEL':<20}")
        print("-" * 80)
        print(f"{'Test Loss':<35} {model_comparison['pt']['test_loss']:<20.6f} {model_comparison['ptl']['test_loss']:<20.6f}")
        print(f"{'Test Accuracy':<35} {model_comparison['pt']['test_acc']:<20.6f} {model_comparison['ptl']['test_acc']:<20.6f}")
        print("-" * 80)
        print("EPOCH-LEVEL METRICS:")
        print(f"{'  Accuracy':<35} {model_comparison['pt']['epoch_acc']:<20.6f} {model_comparison['ptl']['epoch_acc']:<20.6f}")
        print(f"{'  Sensitivity (TPR)':<35} {model_comparison['pt']['epoch_tpr']:<20.6f} {model_comparison['ptl']['epoch_tpr']:<20.6f}")
        print(f"{'  False Alarm Rate (FPR)':<35} {model_comparison['pt']['epoch_fpr']:<20.6f} {model_comparison['ptl']['epoch_fpr']:<20.6f}")
        print(f"{'  True Positives (TP)':<35} {model_comparison['pt']['epoch_tp']:<20} {model_comparison['ptl']['epoch_tp']:<20}")
        print(f"{'  False Positives (FP)':<35} {model_comparison['pt']['epoch_fp']:<20} {model_comparison['ptl']['epoch_fp']:<20}")
        print(f"{'  True Negatives (TN)':<35} {model_comparison['pt']['epoch_tn']:<20} {model_comparison['ptl']['epoch_tn']:<20}")
        print(f"{'  False Negatives (FN)':<35} {model_comparison['pt']['epoch_fn']:<20} {model_comparison['ptl']['epoch_fn']:<20}")
        print("-" * 80)
        print("EVENT-LEVEL METRICS:")
        print(f"{'  Sensitivity (TPR)':<35} {model_comparison['pt']['event_tpr']:<20.6f} {model_comparison['ptl']['event_tpr']:<20.6f}")
        print(f"{'  False Alarm Rate (FPR)':<35} {model_comparison['pt']['event_fpr']:<20.6f} {model_comparison['ptl']['event_fpr']:<20.6f}")
        print(f"{'  True Positives (TP)':<35} {model_comparison['pt']['event_tp']:<20} {model_comparison['ptl']['event_tp']:<20}")
        print(f"{'  False Positives (FP)':<35} {model_comparison['pt']['event_fp']:<20} {model_comparison['ptl']['event_fp']:<20}")
        print(f"{'  True Negatives (TN)':<35} {model_comparison['pt']['event_tn']:<20} {model_comparison['ptl']['event_tn']:<20}")
        print(f"{'  False Negatives (FN)':<35} {model_comparison['pt']['event_fn']:<20} {model_comparison['ptl']['event_fn']:<20}")
        print("="*80 + "\n")
        
        # Save comparison to JSON
        comparison_json = {
            'pt_model': {k: py(v) if hasattr(v, 'item') else (v.tolist() if isinstance(v, np.ndarray) else v) 
                        for k, v in model_comparison['pt'].items() if k != 'event_cm'},
            'ptl_model': {k: py(v) if hasattr(v, 'item') else (v.tolist() if isinstance(v, np.ndarray) else v) 
                         for k, v in model_comparison['ptl'].items() if k != 'event_cm'}
        }
        comparison_json_path = os.path.join(outputDir, f'{modelFnameRoot}_pt_vs_ptl_comparison.json')
        with open(comparison_json_path, 'w') as f:
            json.dump(comparison_json, f, indent=2)
        print(f"{TAG}: Model comparison saved to {comparison_json_path}")
        
        # Generate comparison confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # PT model event-level confusion matrix
        sns.heatmap(model_comparison['pt']['event_cm'], xticklabels=LABELS, yticklabels=LABELS, annot=True,
                    linewidths=0.1, fmt="d", cmap='YlGnBu', ax=axes[0, 0], cbar_kws={'label': 'Count'})
        axes[0, 0].set_title(f"{titlePrefix}.pt: Event-Level", fontsize=13, fontweight='bold')
        axes[0, 0].set_ylabel('True Label', fontsize=11)
        axes[0, 0].set_xlabel('Predicted Label', fontsize=11)
        pt_text = f"TPR: {model_comparison['pt']['event_tpr']:.3f}  FPR: {model_comparison['pt']['event_fpr']:.3f}"
        axes[0, 0].text(0.5, -0.12, pt_text, ha='center', va='top', transform=axes[0, 0].transAxes,
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # PTL model event-level confusion matrix
        sns.heatmap(model_comparison['ptl']['event_cm'], xticklabels=LABELS, yticklabels=LABELS, annot=True,
                    linewidths=0.1, fmt="d", cmap='YlOrRd', ax=axes[0, 1], cbar_kws={'label': 'Count'})
        axes[0, 1].set_title(f"{titlePrefix}.ptl: Event-Level", fontsize=13, fontweight='bold')
        axes[0, 1].set_ylabel('True Label', fontsize=11)
        axes[0, 1].set_xlabel('Predicted Label', fontsize=11)
        ptl_text = f"TPR: {model_comparison['ptl']['event_tpr']:.3f}  FPR: {model_comparison['ptl']['event_fpr']:.3f}"
        axes[0, 1].text(0.5, -0.12, ptl_text, ha='center', va='top', transform=axes[0, 1].transAxes,
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        # PT model epoch-level confusion matrix
        pt_epoch_cm = np.array([[model_comparison['pt']['epoch_tn'], model_comparison['pt']['epoch_fp']],
                                [model_comparison['pt']['epoch_fn'], model_comparison['pt']['epoch_tp']]])
        sns.heatmap(pt_epoch_cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                    linewidths=0.1, fmt="d", cmap='YlGnBu', ax=axes[1, 0], cbar_kws={'label': 'Count'})
        axes[1, 0].set_title(f"{titlePrefix}.pt: Epoch-Level", fontsize=13, fontweight='bold')
        axes[1, 0].set_ylabel('True Label', fontsize=11)
        axes[1, 0].set_xlabel('Predicted Label', fontsize=11)
        pt_epoch_text = f"TPR: {model_comparison['pt']['epoch_tpr']:.3f}  FPR: {model_comparison['pt']['epoch_fpr']:.3f}"
        axes[1, 0].text(0.5, -0.12, pt_epoch_text, ha='center', va='top', transform=axes[1, 0].transAxes,
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # PTL model epoch-level confusion matrix
        ptl_epoch_cm = np.array([[model_comparison['ptl']['epoch_tn'], model_comparison['ptl']['epoch_fp']],
                                 [model_comparison['ptl']['epoch_fn'], model_comparison['ptl']['epoch_tp']]])
        sns.heatmap(ptl_epoch_cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                    linewidths=0.1, fmt="d", cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Count'})
        axes[1, 1].set_title(f"{titlePrefix}.ptl: Epoch-Level", fontsize=13, fontweight='bold')
        axes[1, 1].set_ylabel('True Label', fontsize=11)
        axes[1, 1].set_xlabel('Predicted Label', fontsize=11)
        ptl_epoch_text = f"TPR: {model_comparison['ptl']['epoch_tpr']:.3f}  FPR: {model_comparison['ptl']['epoch_fpr']:.3f}"
        axes[1, 1].text(0.5, -0.12, ptl_epoch_text, ha='center', va='top', transform=axes[1, 1].transAxes,
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        comparison_cm_path = os.path.join(outputDir, f"{modelFnameRoot}_pt_vs_ptl_confusion_matrices.png")
        plt.savefig(comparison_cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"{TAG}: Comparison confusion matrices saved as {comparison_cm_path}")
        
        print(f"{'='*70}")
        print(f"{TAG}: Model comparison complete")
        print(f"{'='*70}\n")
    
    # Clean up memory
    if framework == 'pytorch':
        import torch
        del model, nnModel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"{TAG}: CUDA memory cleared")
    
    print("nnTester: Testing Complete")
    return foldResults



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
                        xTest=None, yTest=None, dataDir=".", balanced=True, debug=False, titlePrefix=None):

    TAG = "nnTrainer.calcConfusionMatrix()"
    print("____%s____" % (TAG))
    
    # If titlePrefix not specified, use modelFnameRoot
    if titlePrefix is None:
        titlePrefix = modelFnameRoot
    
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
    
    # Threshold analysis and probability plot
    pSeizure = prediction_proba[:,1]
    seq = range(0,len(pSeizure))
    # Colour seizure data points red, and non-seizure data blue.
    colours = [ 'red' if seizureVal==1 else 'blue' for seizureVal in yTest]
    
    # Calculate statistics at different thresholds
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
    
    print("Threshold Analysis:")
    print("th", thLst)    
    print("nTP", nTPLst)
    print("nFP", nFPLst)
    print("nTN", nTNLst)
    print("nFN", nFNLst)
    print("TPR", TPRLst)
    print("FPR", FPRLst)
    
    # Create probability scatter plot
    fig, ax = plt.subplots(2,1)
    ax[0].title.set_text("%s: Seizure Probabilities" % titlePrefix)
    ax[0].set_ylabel('Probability')
    ax[0].set_xlabel('Datapoint')
    ax[0].scatter(seq, pSeizure, s=2.0, marker='x', c=colours)
    ax[1].plot(yTest)
    fname_prob = os.path.join(dataDir,"%s_probabilities.png" % modelFnameRoot)
    fig.savefig(fname_prob)
    plt.close()
    print("Probability plot saved as %s" % fname_prob)
       
    # Confusion Matrix
    import seaborn as sns
    LABELS = ['No-Alarm','Seizure']
    # cm = metrics.confusion_matrix(prediction, yTest)
    cm = metrics.confusion_matrix(yTest, prediction)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                linewidths = 0.1, fmt="d", cmap = 'YlGnBu');
    plt.title("%s: Confusion matrix" % titlePrefix, fontsize = 15)
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
        
        # TensorFlow-specific model analysis
        if framework == 'tensorflow':
            from tensorflow import keras
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





def testKFold(configObj, kfold, dataDir='.', rerun=False, debug=False):
    """Test all k-fold models and aggregate results.
    
    Args:
        configObj: Configuration object
        kfold: Number of folds to test
        dataDir: Base directory containing fold subdirectories
        rerun: If True, re-run testing even if results exist
        debug: Enable debug output
    
    Returns:
        Dictionary with aggregated results across all folds
    """
    TAG = "nnTester.testKFold()"
    print("____%s____" % TAG)
    print(f"{TAG}: Testing {kfold} folds")
    
    # Verify that fold directories exist
    fold_dirs = []
    for nFold in range(kfold):
        fold_dir = os.path.join(dataDir, f"fold{nFold}")
        if not os.path.exists(fold_dir):
            raise ValueError(f"{TAG}: Fold directory not found: {fold_dir}. Expected {kfold} folds but fold{nFold} does not exist.")
        fold_dirs.append(fold_dir)
    
    print(f"{TAG}: Found all {kfold} fold directories")
    
    # Test each fold
    foldResults = []
    for nFold in range(kfold):
        fold_dir = fold_dirs[nFold]
        results_file = os.path.join(fold_dir, 'testResults.json')
        
        print(f"\n{TAG}: ========== Processing Fold {nFold} ==========")
        
        # Check if results already exist
        if os.path.exists(results_file) and not rerun:
            print(f"{TAG}: Loading existing results from {results_file}")
            with open(results_file, 'r') as f:
                fold_result = json.load(f)
            foldResults.append(fold_result)
        else:
            if rerun and os.path.exists(results_file):
                print(f"{TAG}: Re-running test for fold {nFold} (--rerun specified)")
            else:
                print(f"{TAG}: No existing results found, running test for fold {nFold}")
            
            # Run the test (skip .ptl testing for k-fold validation, test_ptl=False)
            fold_result = testModel(configObj, dataDir=fold_dir, balanced=False, debug=debug, test_ptl=False)
            foldResults.append(fold_result)
        
        print(f"{TAG}: Fold {nFold} complete")
    
    # Compute average results across folds
    print(f"\n{TAG}: ========== Computing K-Fold Statistics ==========")
    avgResults = {}
    for key in foldResults[0].keys():
        avgResults[key] = sum(foldResult[key] for foldResult in foldResults) / len(foldResults)
    
    # Calculate standard deviation for each key
    for key in foldResults[0].keys():
        avgResults[key + "_std"] = np.std([foldResult[key] for foldResult in foldResults])
    
    # Save the results to files
    kfoldSummaryPath = os.path.join(dataDir, "kfold_summary.txt")
    kfoldJsonPath = os.path.join(dataDir, "kfold_summary.json")
    
    with open(kfoldSummaryPath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("K-Fold Cross-Validation Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Number of folds: {kfold}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write("="*70 + "\n\n")
        
        f.write("EPOCH-BASED ANALYSIS:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Metric':<30} {'Mean':<15} {'Std Dev':<15}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Accuracy (Model)':<30} {avgResults['accuracy']:<15.4f} {avgResults['accuracy_std']:<15.4f}\n")
        f.write(f"{'Accuracy (OSD)':<30} {avgResults['accuracyOsd']:<15.4f} {avgResults['accuracyOsd_std']:<15.4f}\n")
        f.write(f"{'TPR/Sensitivity (Model)':<30} {avgResults['tpr']:<15.4f} {avgResults['tpr_std']:<15.4f}\n")
        f.write(f"{'TPR/Sensitivity (OSD)':<30} {avgResults['tprOsd']:<15.4f} {avgResults['tprOsd_std']:<15.4f}\n")
        f.write(f"{'FPR (Model)':<30} {avgResults['fpr']:<15.4f} {avgResults['fpr_std']:<15.4f}\n")
        f.write(f"{'FPR (OSD)':<30} {avgResults['fprOsd']:<15.4f} {avgResults['fprOsd_std']:<15.4f}\n")
        f.write("\n")
        
        f.write("EVENT-BASED ANALYSIS:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Metric':<30} {'Mean':<15} {'Std Dev':<15}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'TPR/Sensitivity (Model)':<30} {avgResults['event_tpr']:<15.4f} {avgResults['event_tpr_std']:<15.4f}\n")
        f.write(f"{'TPR/Sensitivity (OSD)':<30} {avgResults['osd_event_tpr']:<15.4f} {avgResults['osd_event_tpr_std']:<15.4f}\n")
        f.write(f"{'FPR (Model)':<30} {avgResults['event_fpr']:<15.4f} {avgResults['event_fpr_std']:<15.4f}\n")
        f.write(f"{'FPR (OSD)':<30} {avgResults['osd_event_fpr']:<15.4f} {avgResults['osd_event_fpr_std']:<15.4f}\n")
        f.write("\n")
        
        f.write("DETAILED RESULTS BY FOLD:\n")
        f.write("="*70 + "\n")
        for nFold, result in enumerate(foldResults):
            f.write(f"\nFold {nFold}:\n")
            f.write(f"  Epoch-based - Accuracy: {result['accuracy']:.4f}, TPR: {result['tpr']:.4f}, FPR: {result['fpr']:.4f}\n")
            f.write(f"  Event-based - TPR: {result['event_tpr']:.4f}, FPR: {result['event_fpr']:.4f}\n")
    
    # Save JSON summary
    summary_data = {
        'kfold': kfold,
        'timestamp': str(pd.Timestamp.now()),
        'averages': avgResults,
        'fold_results': foldResults
    }
    
    with open(kfoldJsonPath, 'w') as jf:
        json.dump(summary_data, jf, indent=2)
    
    print(f"\n{TAG}: K-Fold summary saved to {kfoldSummaryPath}")
    print(f"{TAG}: K-Fold JSON data saved to {kfoldJsonPath}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(f"Number of folds: {kfold}\n")
    print("EPOCH-BASED ANALYSIS:")
    print(f"  Model - Accuracy: {avgResults['accuracy']:.4f} ± {avgResults['accuracy_std']:.4f}")
    print(f"  Model - TPR: {avgResults['tpr']:.4f} ± {avgResults['tpr_std']:.4f}")
    print(f"  Model - FPR: {avgResults['fpr']:.4f} ± {avgResults['fpr_std']:.4f}")
    print(f"  OSD   - Accuracy: {avgResults['accuracyOsd']:.4f} ± {avgResults['accuracyOsd_std']:.4f}")
    print(f"  OSD   - TPR: {avgResults['tprOsd']:.4f} ± {avgResults['tprOsd_std']:.4f}")
    print(f"  OSD   - FPR: {avgResults['fprOsd']:.4f} ± {avgResults['fprOsd_std']:.4f}")
    print("\nEVENT-BASED ANALYSIS:")
    print(f"  Model - TPR: {avgResults['event_tpr']:.4f} ± {avgResults['event_tpr_std']:.4f}")
    print(f"  Model - FPR: {avgResults['event_fpr']:.4f} ± {avgResults['event_fpr_std']:.4f}")
    print(f"  OSD   - TPR: {avgResults['osd_event_tpr']:.4f} ± {avgResults['osd_event_tpr_std']:.4f}")
    print(f"  OSD   - FPR: {avgResults['osd_event_fpr']:.4f} ± {avgResults['osd_event_fpr_std']:.4f}")
    print("="*70)
    
    return avgResults


def main():
    print("nnTester.main()")
    parser = argparse.ArgumentParser(description='Apply the training data to calculate statistcs on a trained model (specifid in the config file)')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing model configuration')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    parser.add_argument('--test-data', default=None,
                        help='Path to test data CSV file (overrides config file setting)')
    parser.add_argument('--kfold', type=int, default=None,
                        help='Number of folds for k-fold cross-validation testing. Tests all folds and aggregates results.')
    parser.add_argument('--rerun', action="store_true",
                        help='Re-run tests even if results already exist (only used with --kfold)')
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

    # Check if k-fold testing is requested
    if args['kfold'] is not None:
        kfold = args['kfold']
        if kfold < 1:
            raise ValueError(f"kfold must be >= 1, got {kfold}")
        rerun = args['rerun']
        testKFold(configObj, kfold=kfold, dataDir='.', rerun=rerun, debug=debug)
    else:
        # Test single model with .ptl comparison enabled
        testModel(configObj, debug=debug, testDataCsv=args['test_data'], test_ptl=True)
        
    


if __name__ == "__main__":
    main()
