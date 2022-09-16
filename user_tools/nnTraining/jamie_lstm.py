# importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.optimizers import SGD


# My column names incase it helps 
# eventID
# time
# spec_power
# roi_power
# roi_Ratio
# hr
# mag
# user
# outcome


# reading data from google drive /content/gdrive/MyDrive/OSD Research/data/seizures.csv
mypath = ''
#outcome column has to be your classifier outcome, 0 = no seizure, 1 = seizure
columns = ['col_1','col_2','col_3','outcome']
#If your data already has column headings then remove columns variable from the names param
df = pd.read_csv(mypath, header = None, names = columns)

df.head()


# analysis of class labels

label_dict = dict(df['outcome'].value_counts())

sns.set_style("whitegrid")
plt.figure(figsize = (8, 4))
sns.barplot(x = list(label_dict.keys()), y = list(label_dict.values()))
plt.xlabel('Activity')
plt.ylabel('Data points per activity')
plt.title('Number of samples by activities')
plt.show()

# Percentage-wise distribution of the class label yi's
print("- "* 50)
for i in label_dict.keys():
  print("Number of data points in class {0} = {1} ~ {2}%".format(
  i, label_dict[i], round((label_dict[i]*100)/sum(label_dict.values()), 2)))
print("-"*50)
print("total datapoints:", sum(label_dict.values()))

#Current shape of the dataframe
print(df.shape)

#Number of events for each participant
label_dict = dict(df['user'].value_counts()/125)
sns.set_style("whitegrid")
plt.figure(figsize = (12, 5))
sns.barplot(x = list(label_dict.keys()), y = list(label_dict.values()))
plt.xlabel('Users')
plt.ylabel('Data points per user')
plt.title('Number of samples by users')
plt.show()


#function to plot the sensor data based on position in the dataframe. Here we take the first 125 values from the dataframe and plot based on whether the outcome was 0 or 1
def plot_activity(activity):
    data = har_df[har_df['outcome'] == activity][['mag']][:125]
    axis = data["mag"].plot(subplots=True, color="b", fontsize = 12)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5), fontsize = 12)
        
#Call function
for i in np.unique(har_df['outcome'].values):
    plt.figure(figsize = (16, 6))
    print("\n")
    print(" - "*15 + str(i) + " - "*15)
    print("\n")
    plot_activity(i)
    plt.show()

# drop the records where times = 0
df = har_df[har_df['time'] != 0]
#print(df.shape)

# now arrange data in ascending order of the User and timestamp
df_sorted = df.sort_values(by = ['user', 'time'], ignore_index=True)
df_sorted


#Model Specs, Alter as you see fit
RANDOM_SEED =  28       # Random seed for num gen initalisation
N_TIME_STEPS = 125      # 125 records in each sequence
N_FEATURES = 5          # Features are the number of columns you intend to use
step = 100              # window overlap = 50 -10 = 40  (80% overlap) I use an 80% overlap between windows 
N_CLASSES = 2           # class labels , numer of classes, the model curretly uses softmax so you can add more than 2 classes if you see fit
LEARNING_RATE = 0.0025  # Hyper-parameter used to govern the pace at which an algorithm updates or learns
L2_LOSS = 0.0015        # Function to calculate the distance between the current output and the expected output 

# Array
segments = []
labels = []

#remove each of the columns unitl you have what you need
for i in range(0,  df_sorted.shape[0]-N_TIME_STEPS, step):  
    mag = df_sorted['mag'].values[i: i + 125]
    hr = df_sorted['hr'].values[i: i + 125]
    roi_Ratio = df_sorted['roi_Ratio'].values[i: i + 125]
    roi_power = df_sorted['roi_power'].values[i: i + 125]
    spec_power = df_sorted['spec_power'].values[i: i + 125]
    label = stats.mode(df_sorted['outcome'][i: i + 125])[0][0]
    segments.append([mag,hr,roi_Ratio,roi_power,spec_power])
    labels.append(label)
    
#Convert to Numpy Array
np.array(segments).shape

#Rehsape Numpy Array
reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

#Train test split dataset , test size should be between 20%-25%
X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size = 0.5, random_state = RANDOM_SEED)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Shape (number of rows, number of features, number of records in each sequence)
# For example (1000, 5, 125) we have 1000 rows, 5 features (mag,hr,roi_Ratio,roi_power,spec_power), and 125 records in each sequence

# LSTM model
epochs = 500            # Change the number of iterations for training the model unitl you are happy
batch_size =  1024      # Number of samples to propagte through the network

model = Sequential()
# RNN layer
model.add(LSTM(units = 128, input_shape= (X_train.shape[1], X_train.shape[2]))) # input_shape = 50, 3
# Dropout layer
model.add(Dropout(0.3)) 
# Dense layer with ReL
model.add(Dense(units = 128, activation='relu'))
# Softmax layer
model.add(Dense(y_train.shape[1], activation = 'softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Summary of Each Layer
model.summary()

#Train Model
# Training
history = model.fit(X_train, y_train, epochs = epochs, 
                    validation_split = 0.25, batch_size = batch_size, verbose = 1)


# Train and Validation: multi-class log-Loss & accuracy plot
plt.figure(figsize=(12, 8))
plt.plot(np.array(history.history['loss']), "r--", label = "Train loss")
plt.plot(np.array(history.history['accuracy']), "g--", label = "Train accuracy")
plt.title("Training session's progress over iterations")
plt.legend(loc='lower left')
plt.ylabel('Training Progress (Loss/Accuracy)')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.show()

# Train and Validation: multi-class log-Loss & accuracy plot
plt.figure(figsize=(12, 8))
plt.plot(np.array(history.history['loss']), "y--", label = "Train loss")
plt.plot(np.array(history.history['val_loss']), "p-", label = "Validation loss")
plt.title("Training session's progress over iterations")
plt.legend(loc='lower left')
plt.ylabel('Training Progress (Loss/Accuracy)')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.show()


# Test data loss & accuracy
loss, accuracy = model.evaluate(X_train, y_train, batch_size = batch_size, verbose = 1)
print("")
print("Test Accuracy :", accuracy)
print("Test Loss :", loss)

# Test data loss & accuracy
loss, accuracy = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
print("")
print("Test Accuracy :", accuracy)
print("Test Loss :", loss)

#convert model for test runner
import joblib
joblib.dump(model, 'file.pkl')
model_joblib = joblib.load('file.pkl')

# make and show prediction
print(model.predict(y))




