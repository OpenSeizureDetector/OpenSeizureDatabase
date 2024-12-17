# 🎨 **Data Analysis and Visualisation**  

This repository contains a **Python-based pipeline** for analysing and visualising raw sensor data from the **OpenSeizure Database**. The provided scripts enable signal processing and visualisation using techniques such as **FFT**, **spectrograms**, and **skewness/kurtosis** analysis.  
---

## 📚 **Table of Contents**

## 📂 **1. DataLoader.ipynb**  
🔹 Loads and processes the sensor data from JSON files.  
🔹 Outputs structured **DataFrames** for further analysis.  

## 📈 **2. fft_processor.py**  
🔹 Computes and plots the **Fast Fourier Transform (FFT)** for raw sensor data.  
🔹 Visualises frequency domain representations to identify patterns.  

## 📉 **3. frequency_band_energy.py**  
🔹 Calculates and plots the energy distribution across predefined **frequency bands**.  
🔹 Useful for identifying signal characteristics over specific frequency ranges.  

## 📊 **4. raw_sensor_data.py**  
🔹 Plots the raw sensor data (e.g., accelerometer readings) over time.  
🔹 Visualises trends and anomalies within sensor recordings.  

## 📉 **5. skewness_kurtosis.py**  
🔹 Computes and visualises **skewness** and **kurtosis** of sensor data.  
🔹 Highlights data distribution characteristics for each event.  

## 📉 **6. spectogram_raw_Data_plotter.py**  
🔹 Generates **spectrograms** for raw sensor data over time.  
🔹 Provides insights into frequency components within specific events.  

## 📉 **7. std_and_max_acceleration.py**  
🔹 Computes **standard deviation** and **maximum acceleration** for events.  
🔹 Plots statistical summaries of accelerometer data.

## 📉 **8. main.py**  
🔹 Executes all key data loading, preprocessing, and visualisation steps.  
🔹 A single pipeline to **run all analyses** for multiple events efficiently.  
---

## 📚 **How to Run the Data Visualiser**

### Terminal Commands:
- Navigate to your main project folder and install the required dependencies.
```bash
# Navigate to the project folder
cd path/to/your/OpenSeizureDatabase/user_tools/dataVisualiser

# Install the dependencies from requirements.txt
pip install -r requirements.txt
```

### Python Script for Configuration
- In your Python script, set the `data_url` and `eventID` by defining them as strings. Here's an example:
```python
# Import necessary libraries and classes
from data_loader import DataLoader
from raw_sensor_data import RawDataPlotter

# Path to the JSON file containing sensor data
data_url = 'https://example.com/datasets/osdb_3min_allSeizures.json'  # Modify this path if needed

# Event ID for the analysis (modify this for different events)
event_id = 407  # Modify this to your target Event ID

# Select the visualization script you want to use from the available options, # For example, choose 'SpectrogramRawDataPlotter' for generating spectrograms
visualization_script = SpectrogramRawDataPlotter  # Modify this to select your desired visualization script
```
### Terminal Commands:
- Navigate to your main project folder and install the required dependencies.
```bash
# run the main python script
python main.py
```


---

## 📝 **About This Repository**  
- **Author**: Jamie Pordoy  
- **Tools Used**: Python, NumPy, Pandas, SciPy, Matplotlib, Librosa  
---