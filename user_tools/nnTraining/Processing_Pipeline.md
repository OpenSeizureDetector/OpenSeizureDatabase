# Processing Pipeline

The nnTrainer.py processing sequence is:

  * Split osdb data into a test and train data file (split by event)
  * Flatten the train data file:  ./flattenOsdb.py  -o trainData.csv
  * Run nnTrainer_csv.py
