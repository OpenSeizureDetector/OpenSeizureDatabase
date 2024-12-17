# Data Analysis and Visualisation

This repository contains a Python-based pipeline for analyzing and visualizing raw sensor data from the OpenSeizure Database. The provided scripts facilitate the processing and visualization of seizure-related data using various signal processing and plotting techniques, such as FFT, spectrograms, and skewness/kurtosis analysis.

---
## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Main Script](#running-the-main-script)
  - [User Configurable Parameters](#user-configurable-parameters)
- [Components](#components)
  - [DataLoader](#dataloader)
  - [RawSensorDataPlotter](#rawsensordataplotter)
  - [FFTProcessor](#fftprocessor)
  - [SpectrogramRawDataPlotter](#spectrogramrawdataplotter)
  - [SkewnessKurtosis](#skewnesskurtosis)
  - [FrequencyBandEnergy](#frequencybandenergy)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Data Loading**: Handles JSON-formatted raw sensor data from the OpenSeizure Database.
- **Raw Data Visualization**: Plots acceleration and other sensor data for specific events.
- **Fourier Transform Analysis**: Computes and visualizes the FFT for frequency domain analysis.
- **Spectrogram Generation**: Generates spectrograms for single and multiple events.
- **Skewness and Kurtosis Analysis**: Computes and visualizes skewness and kurtosis values for statistical insights.
- **Frequency Band Energy**: Plots energy distribution across frequency bands for a given event.

---

## Usage

### Running the Main Script
- To analyze and visualize data, run the `main.py` script:

```bash
python main.py
```

### User Configurable Parameters
Edit the following parameters in the `main.py` file to customize the analysis:

- `file_path`: Path to the JSON file containing sensor data.
- `event_id`: Target event ID for analysis.
- `event_ids`: List of event IDs for multi-event analysis.

---

## Components

### DataLoader
**File:** `data_loader.py`  

Responsible for loading and preprocessing sensor data from a JSON file. It initializes a DataFrame `df_sensordata` containing the processed data. This is a critical first step in transforming raw JSON into a format ready for analysis.

---

### RawSensorDataPlotter
**File:** `raw_sensor_data.py`  

Plots raw sensor data for a specific event. Useful for visualizing time-series data such as acceleration. This helps in identifying patterns or anomalies in sensor readings that could correlate with seizure activity, enabling visual insight into potential triggers or responses.

---

### FFTProcessor
**File:** `fft_processor.py`  

Performs Fourier Transform analysis:  

- **Process the Data:** Computes FFT to extract frequency domain information. This analysis helps by transforming time-domain data (like accelerometer signals) into the frequency domain, enabling the identification of characteristic frequencies that might indicate seizure activity.  
- **Plot the Results:** Visualizes positive frequency magnitudes. The frequency spectrum reveals dominant frequencies in sensor signals, aiding in the detection of seizure-related oscillations or rhythms.

---

### SpectrogramRawDataPlotter
**File:** `spectrogram_rawdata_plotter.py`  

Generates spectrograms for single or multiple events:  

- **Single Event:** Visualizes time-frequency domain for a single event. This representation helps in detecting changes over time in the signal's frequency content, useful for identifying sudden shifts in sensor readings associated with seizures.  
- **Multiple Events:** Plots spectrograms in a grid for comparative analysis. This comparison allows for cross-event analysis, facilitating the detection of recurring patterns or anomalies across different seizure episodes.

---

### SkewnessKurtosis
**File:** `skewness_kurtosis.py`  

Computes and plots skewness and kurtosis values for the target event:  

- **Skewness:** Measures the asymmetry of the signal distribution. In sensor data, skewness can indicate the presence of atypical fluctuations or imbalances in the data, which might point to abnormal seizure-related activities.  
- **Kurtosis:** Measures the "tailedness" of the signal distribution. High kurtosis could indicate extreme outlier events, which are crucial for detecting seizure spikes or abnormal movements in accelerometer signals.

---

### FrequencyBandEnergy
**File:** `frequency_band_energy.py`  

Calculates and plots energy distribution across predefined frequency bands for the specified event. This helps by identifying the energy concentration within different frequency ranges, assisting in the detection of seizure-specific frequency bands, such as low-frequency oscillations or high-frequency tremors.

---

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Submit a pull request.
