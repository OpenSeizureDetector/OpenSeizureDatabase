#!/usr/bin/env python3

import argparse
import sys
import os
import sklearn.model_selection

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.configUtils



def splitData(configObj, kFold=1, nestedKfold=1, outDir=".", debug=False):
    """
    Split data into train/test folds as CSV files.
    
    Since the data has already been flattened to CSV by the runSequence pipeline,
    this function now splits the allData.csv into fold-specific CSV files rather
    than working with JSON files.
    
    This replaces the original JSON-based splitting to maintain consistency with
    the new pipeline architecture where all processing after flatten uses CSV files.
    
    Args:
        configObj: Configuration dictionary
        kFold: Number of inner folds
        nestedKfold: Number of outer folds for nested k-fold
        outDir: Directory where the output files will be saved
        debug: Enable debug output
    """
    if debug:
        print("splitData: Splitting CSV data (JSON files are no longer used)")
        print("splitData: configObj keys =", configObj.keys())
    
    # The allData.csv should already exist (created by runSequence after flatten)
    allDataCsvFname = configObj['dataFileNames'].get('allDataFileCsv', 'allData.csv')
    allDataCsvPath = os.path.join(outDir, allDataCsvFname)
    
    if not os.path.exists(allDataCsvPath):
        print("ERROR: allData.csv not found at %s" % allDataCsvPath)
        print("       splitData expects the CSV file to be created by runSequence")
        print("       after flattening allData.json")
        return
    
    # Use splitCsvData to do the actual splitting
    print("splitData: Splitting %s into fold-specific CSV files" % allDataCsvFname)
    splitCsvData(configObj, allDataCsvPath, outDir=outDir, kFold=kFold, nestedKfold=nestedKfold, debug=debug)
    print("splitData: CSV splitting complete")


def splitCsvData(configObj, csvPath, outDir=".", kFold=1, nestedKfold=1, debug=False):
    """
    Split a flattened CSV data file into fold-specific CSV files, matching the fold
    structure that was previously created by splitData (from JSON files).
    
    This function reads the flattened allData.csv and splits it based on the fold structure,
    creating CSV files for each train/test fold.
    
    Args:
        configObj: Configuration object
        csvPath: Path to the flattened CSV file (e.g., allData.csv)
        outDir: Output folder containing the fold structure (from splitData)
        kFold: Number of inner folds
        nestedKfold: Number of outer folds
        debug: Debug flag
    """
    import pandas as pd
    
    testCsvFname = configObj['dataFileNames']['testDataFileCsv']
    trainCsvFname = configObj['dataFileNames']['trainDataFileCsv']
    
    # Load the flattened CSV
    df_all = pd.read_csv(csvPath, low_memory=False)
    
    if debug:
        print("splitCsvData: Loaded CSV with %d rows and %d columns" % (len(df_all), len(df_all.columns)))
        print("splitCsvData: Columns: %s" % list(df_all.columns))
    
    # Get the eventId column - handle different possible column names
    event_id_col = None
    for possible_col in ['eventId', 'EventID', 'event_id', 'id']:
        if possible_col in df_all.columns:
            event_id_col = possible_col
            break
    
    if event_id_col is None:
        print("ERROR: Could not find eventId column in CSV. Available columns: %s" % list(df_all.columns))
        return
    
    # Get the type column to separate seizures from non-seizures for stratified splitting
    type_col = None
    for possible_col in ['type', 'Type', 'seizureType']:
        if possible_col in df_all.columns:
            type_col = possible_col
            break
    
    if debug:
        print("splitCsvData: Using eventId column: %s" % event_id_col)
        print("splitCsvData: Using type column: %s" % type_col)
    
    # Group by eventId to get unique events (since CSV will have multiple rows per event)
    unique_events = df_all.drop_duplicates(subset=[event_id_col])[[event_id_col]].reset_index(drop=True)
    event_ids = unique_events[event_id_col].tolist()
    
    # Create labels for stratification (seizure vs non-seizure)
    if type_col:
        event_labels = []
        for event_id in event_ids:
            event_type = df_all[df_all[event_id_col] == event_id][type_col].iloc[0]
            # Normalize type to 0 or 1 for stratification
            if isinstance(event_type, str):
                label = 1 if event_type.lower() == 'seizure' else 0
            else:
                label = 1 if event_type == 1 else 0
            event_labels.append(label)
    else:
        # If no type column, use dummy labels
        event_labels = [0] * len(event_ids)
    
    randomSeed = configObj.get('randomSeed', 42)
    
    # Split the CSV data according to the fold structure
    if nestedKfold > 1:
        print("splitCsvData: Splitting CSV for nested k-fold structure")
        
        skf_outer = sklearn.model_selection.StratifiedKFold(
            n_splits=nestedKfold, shuffle=True, random_state=randomSeed
        )
        
        for outer_fold, (trainval_outer_idx, test_outer_idx) in enumerate(
            skf_outer.split(event_ids, event_labels)
        ):
            outerFoldFolder = os.path.join(outDir, "outerfold%d" % outer_fold)
            if not os.path.exists(outerFoldFolder):
                os.makedirs(outerFoldFolder, exist_ok=True)
            
            # Get outer fold test event IDs
            test_outer_ids = [event_ids[i] for i in test_outer_idx]
            trainval_outer_ids = [event_ids[i] for i in trainval_outer_idx]
            
            # Save outer fold test CSV
            test_outer_csv = os.path.join(outerFoldFolder, testCsvFname.replace('.csv', '_outer.csv'))
            df_test_outer = df_all[df_all[event_id_col].isin(test_outer_ids)]
            df_test_outer.to_csv(test_outer_csv, index=False)
            if debug:
                print("splitCsvData: Saved outer fold %d test CSV to %s (%d rows)" % (outer_fold, test_outer_csv, len(df_test_outer)))
            
            # Inner k-fold
            if kFold > 1:
                skf_inner = sklearn.model_selection.StratifiedKFold(
                    n_splits=kFold, shuffle=True, random_state=randomSeed
                )
                
                trainval_outer_labels = [event_labels[i] for i in trainval_outer_idx]
                
                for inner_fold, (train_idx, test_idx) in enumerate(
                    skf_inner.split(trainval_outer_ids, trainval_outer_labels)
                ):
                    foldFolder = os.path.join(outerFoldFolder, "fold%d" % inner_fold)
                    if not os.path.exists(foldFolder):
                        os.makedirs(foldFolder, exist_ok=True)
                    
                    train_ids = [trainval_outer_ids[i] for i in train_idx]
                    test_ids = [trainval_outer_ids[i] for i in test_idx]
                    
                    # Save inner fold training CSV
                    train_csv = os.path.join(foldFolder, trainCsvFname)
                    df_train = df_all[df_all[event_id_col].isin(train_ids)]
                    df_train.to_csv(train_csv, index=False)
                    
                    # Save inner fold test CSV
                    test_csv = os.path.join(foldFolder, testCsvFname)
                    df_test = df_all[df_all[event_id_col].isin(test_ids)]
                    df_test.to_csv(test_csv, index=False)
                    
                    if debug:
                        print("splitCsvData: Saved outer fold %d, inner fold %d CSVs" % (outer_fold, inner_fold))
            else:
                # If kFold == 1, just save the outer fold data as-is
                train_csv = os.path.join(outerFoldFolder, trainCsvFname)
                test_csv = os.path.join(outerFoldFolder, testCsvFname)
                
                df_train = df_all[df_all[event_id_col].isin(trainval_outer_ids)]
                df_train.to_csv(train_csv, index=False)
                
                df_test = df_all[df_all[event_id_col].isin(test_outer_ids)]
                df_test.to_csv(test_csv, index=False)
                
                if debug:
                    print("splitCsvData: Saved outer fold %d CSVs (no inner k-fold)" % outer_fold)
    
    elif kFold > 1:
        print("splitCsvData: Splitting CSV for regular k-fold structure")
        
        skf = sklearn.model_selection.StratifiedKFold(
            n_splits=kFold, shuffle=True, random_state=randomSeed
        )
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(event_ids, event_labels)):
            foldFolder = os.path.join(outDir, "fold%d" % fold)
            if not os.path.exists(foldFolder):
                os.makedirs(foldFolder, exist_ok=True)
            
            train_ids = [event_ids[i] for i in train_idx]
            test_ids = [event_ids[i] for i in test_idx]
            
            # Save fold training CSV
            train_csv = os.path.join(foldFolder, trainCsvFname)
            df_train = df_all[df_all[event_id_col].isin(train_ids)]
            df_train.to_csv(train_csv, index=False)
            
            # Save fold test CSV
            test_csv = os.path.join(foldFolder, testCsvFname)
            df_test = df_all[df_all[event_id_col].isin(test_ids)]
            df_test.to_csv(test_csv, index=False)
            
            if debug:
                print("splitCsvData: Saved fold %d CSVs" % fold)
    
    else:
        print("splitCsvData: Splitting CSV for single train/test split")
        
        testProp = configObj['dataProcessing']['testProp']
        train_ids, test_ids = sklearn.model_selection.train_test_split(
            event_ids, test_size=testProp, random_state=randomSeed, stratify=event_labels
        )
        
        # Save training CSV
        if not os.path.exists(outDir):
            os.makedirs(outDir, exist_ok=True)
        train_csv = os.path.join(outDir, trainCsvFname)
        df_train = df_all[df_all[event_id_col].isin(train_ids)]
        df_train.to_csv(train_csv, index=False)
        
        # Save test CSV
        test_csv = os.path.join(outDir, testCsvFname)
        df_test = df_all[df_all[event_id_col].isin(test_ids)]
        df_test.to_csv(test_csv, index=False)
        
        if debug:
            print("splitCsvData: Saved train/test CSVs")
    
    print("splitCsvData: CSV splitting complete")



def main():
    print("selectData.main()")
    parser = argparse.ArgumentParser(description='Split OSDB Data into Test and Train Data sets')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing configuration')
    parser.add_argument('--kfold', default=1, type=int,
                        help='number of inner folds for cross-validation')
    parser.add_argument('--nestedKfold', default=1, type=int,
                        help='number of outer folds for nested k-fold validation (default 1 = no nesting)')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)



    configObj = libosd.configUtils.loadConfig(args['config'])
    print("configObj=",configObj.keys())
    # Load a separate OSDB Configuration file if it is included.
    if ("osdbCfg" in configObj):
        osdbCfgFname = libosd.configUtils.getConfigParam("osdbCfg",configObj)
        print("Loading separate OSDB Configuration File %s." % osdbCfgFname)
        osdbCfgObj = libosd.configUtils.loadConfig(osdbCfgFname)
        # Merge the contents of the OSDB Configuration file into configObj
        configObj = configObj | osdbCfgObj

    print("configObj=",configObj.keys())

    
    splitData(configObj, kFold=args['kfold'], nestedKfold=args['nestedKfold'], debug=args['debug'])
        
    


if __name__ == "__main__":
    main()
