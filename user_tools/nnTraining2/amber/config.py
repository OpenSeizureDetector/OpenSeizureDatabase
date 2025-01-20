class Config:

    #Globals
    batch_size = 64
    num_classes = 2  # classes, seizure/no seizure
    epochs = 5     # Epoch iterations
    row_hidden = 128   # hidden neurons in conv layers
    col_hidden = 128   # hidden neurons in the Bi-LSTM layers
    RANDOM_SEED = 333  # random   
    N_TIME_STEPS = 125 # 50 records in each sequence
    N_FEATURES = 3     # mag,hr,FFT
    step = 100         # window overlap = 50 -10 = 40  (80% overlap)
    N_CLASSES = 2      # class labels
    learning_rate = 0.00001
    k = 5 # number of k folds
    length_time_step = 5 

    