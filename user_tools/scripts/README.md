# 📊 **Data Visualisation Folder Overview**  

This folder contains a collection of Jupyter Notebooks designed for **data loading**, **preprocessing**, **feature extraction**, **visualisation** and **machine learning** of sensor data from the **Open Seizure Database (OSDB)**. The notebooks are modular, ensuring streamlined and efficient analysis. Below is an overview of the available notebooks:

---

## 📂 **1. DataLoader.ipynb**  
🔹 Loads and processes the sensor data from JSON files.  
🔹 Outputs structured **DataFrames** for further analysis.  

---

## ⚙️ **2. FeatureEngineering.ipynb**  
🔹 Extracts key statistical features from raw sensor data.  
🔹 Prepares data for input to machine learning models.  

---

## 🧠 **3. ConvolutionalNeuralNetwork.ipynb**  
🔹 Implements a **Convolutional Neural Network (CNN)** for classification tasks.  
🔹 Trains and evaluates the model on processed OSDB data.  

---

## 📈 **4. fft_plotter.ipynb**  
🔹 Computes and plots the **Fast Fourier Transform (FFT)** for raw sensor data.  
🔹 Visualises frequency domain representations to identify patterns.  

---

## 🎵 **5. frequency_band_plotter.ipynb**  
🔹 Calculates and plots the energy distribution across predefined **frequency bands**.  
🔹 Useful for identifying signal characteristics over specific frequency ranges.  

---

## 🛤️ **6. raw_Data_distance_plotter.ipynb**  
🔹 Calculates **total distance** travelled using accelerometer data.  
🔹 Plots movement over time for individual events.  

---

## 📊 **7. raw_Dataploter.ipynb**  
🔹 Plots the raw sensor data (e.g., accelerometer readings) over time.  
🔹 Visualises trends and anomalies within sensor recordings.  

---

## 🚀 **8. run_all_experiments.ipynb**  
🔹 Executes all key data loading, preprocessing, and visualisation steps.  
🔹 A single pipeline to **run all analyses** for multiple events efficiently.  

---

## 📉 **9. skewness_kurtosis_plotter.ipynb**  
🔹 Computes and visualises **skewness** and **kurtosis** of sensor data.  
🔹 Highlights data distribution characteristics for each event.  

---

## 🎛️ **10. spectogram_raw_Data_plotter.ipynb**  
🔹 Generates **spectrograms** for raw sensor data over time.  
🔹 Provides insights into frequency components within specific events.  

---

## 📊 **11. std_and_max_acceleration_plotter.ipynb**  
🔹 Computes **standard deviation** and **maximum acceleration** for events.  
🔹 Plots statistical summaries of accelerometer data.

---

## 📝 **About This Repository**  
- **Author**: Jamie Pordoy  
- **Tools Used**: Python, NumPy, Pandas, SciPy, Matplotlib, Librosa  
---