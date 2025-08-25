#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.configUtils
try:
    from user_tools.nnTraining2 import augmentData
except ImportError:
    import augmentData

def fpr_score(y, y_pred, pos_label=1, neg_label=0):
    cm = sklearn.metrics.confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / (tn + fp)
    tpr = tp / (tp + fn)
    fpr = 1 - tnr
    return (tpr, fpr)

def testModel(configObj, dataDir='.', debug=False):
    TAG = "skTester.testModel()"
    testCsvFname = libosd.configUtils.getConfigParam("testFeaturesFileCsv", configObj['dataFileNames'])
    modelFnameRoot = libosd.configUtils.getConfigParam("modelFname", configObj['modelConfig'])
    modelFname = "%s.sklearn" % modelFnameRoot
    modelFnamePath = os.path.join(dataDir, modelFname)
    testCsvFnamePath = os.path.join(dataDir, testCsvFname)

    import joblib
    model = joblib.load(modelFnamePath)

    print("%s: Testing model on test data" % TAG)
    testDf = augmentData.loadCsv(testCsvFnamePath, debug=debug)
    print("%s: Loaded %d datapoints from file %s" % (TAG, len(testDf), testCsvFname))
    xTest = testDf[configObj['dataProcessing']['features']]
    yTest = testDf['type']

    print(xTest, yTest)

    yPred = model.predict(xTest)
    tpr, fpr = fpr_score(yTest, yPred)
    print(f"{TAG}: True Positive Rate (TPR): {tpr:.4f}, False Positive Rate (FPR): {fpr:.4f}")

    accuracy = sklearn.metrics.accuracy_score(yTest, yPred)
    print(f'{TAG}: Model Accuracy: {accuracy:.2f}')
    print(sklearn.metrics.confusion_matrix(yTest, yPred))

    yPredOsd = testDf['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)
    yTestOsd = testDf['type']
    tprOsd, fprOsd = fpr_score(yTestOsd, yPredOsd)
    print(f"OSD Algorithm: True Positive Rate (TPR): {tprOsd:.4f}, False Positive Rate (FPR): {fprOsd:.4f}")

    print("\nOSD Algorithm Predictions:")
    print("OSD Alarm State Accuracy: %.2f" % sklearn.metrics.accuracy_score(yTestOsd, yPredOsd))
    print(sklearn.metrics.confusion_matrix(yTestOsd, yPredOsd))

    cm = sklearn.metrics.confusion_matrix(yTest, yPred, labels=[0, 1])
    cmOsd = sklearn.metrics.confusion_matrix(yTestOsd, yPredOsd, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tnOsd, fpOsd, fnOsd, tpOsd = cmOsd.ravel()
    accuracyOsd = sklearn.metrics.accuracy_score(yTestOsd, yPredOsd)


    # Now group the data by eventId because we are not really interested in the results for individual
    # data points, what matters is whether at least one datapoint within a seizure event is identified
    # as a seizure (which means the event is classified correctly), 
    # and whether one datapoint within a non-seizure event is misclassified as a seizure, which means the
    # event is classified incorrectly.

    # First add the prediction columns into the full dataFrame
    testDf['pred'] = yPred
    testDf['osd_pred'] = testDf['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)

    # Group by eventId and calculate event-level predictions for both model and OSD
    event_stats = []
    for eventId, group in testDf.groupby('eventId'):
        true_label = group['type'].iloc[0]
        model_event_pred = 1 if (group['pred'] == 1).any() else 0
        osd_event_pred = 1 if (group['osd_pred'] == 1).any() else 0
        event_stats.append({
            'eventId': eventId,
            'true_label': true_label,
            'model_pred': model_event_pred,
            'osd_pred': osd_event_pred
        })
    event_stats_df = pd.DataFrame(event_stats)

    # Model event-based stats
    event_tpr, event_fpr = fpr_score(event_stats_df['true_label'], event_stats_df['model_pred'])
    event_cm = sklearn.metrics.confusion_matrix(event_stats_df['true_label'], event_stats_df['model_pred'], labels=[0, 1])
    print(f"{TAG}: Event-level Confusion Matrix (Model):")
    print(event_cm)
    print(f"{TAG}: Event-level True Positive Rate (TPR): {event_tpr:.4f}, Event-level False Positive Rate (FPR): {event_fpr:.4f}")

    # OSD event-based stats
    osd_event_tpr, osd_event_fpr = fpr_score(event_stats_df['true_label'], event_stats_df['osd_pred'])
    osd_event_cm = sklearn.metrics.confusion_matrix(event_stats_df['true_label'], event_stats_df['osd_pred'], labels=[0, 1])
    print(f"{TAG}: Event-level Confusion Matrix (OSD Algorithm):")
    print(osd_event_cm)
    print(f"{TAG}: OSD Event-level True Positive Rate (TPR): {osd_event_tpr:.4f}, OSD Event-level False Positive Rate (FPR): {osd_event_fpr:.4f}")

    foldResults = {
        'accuracy': accuracy,
        'accuracyOsd': accuracyOsd,
        'tpr': tpr,
        'fpr': fpr,
        'tprOsd': tprOsd,
        'fprOsd': fprOsd,
        'tn': tn, 
        'fp': fp, 
        'fn': fn, 
        'tp': tp,
        'tnOsd': tnOsd, 
        'fpOsd': fpOsd, 
        'fnOsd': fnOsd, 
        'tpOsd': tpOsd,
        'event_tpr': event_tpr,
        'event_fpr': event_fpr,
        'event_cm': event_cm.tolist(),
        'osd_event_tpr': osd_event_tpr,
        'osd_event_fpr': osd_event_fpr,
        'osd_event_cm': osd_event_cm.tolist()
    }

    print("skTester: Testing Complete")
    return

def main():
    parser = argparse.ArgumentParser(description='Test a trained scikit-learn model')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing test configuration')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    args = parser.parse_args()
    configObj = libosd.configUtils.loadConfig(args.config)
    debug = configObj.get('debug', False)
    if args.debug: debug = True
    testModel(configObj, debug=debug)

if __name__ == "__main__":
    main()