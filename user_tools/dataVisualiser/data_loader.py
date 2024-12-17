import pandas as pd
import json
import numpy as np
from scipy.stats import skew, kurtosis

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df_metadata = None
        self.df_sensordata = None

    def load_metadata(self):
        with open(self.file_path, 'r') as file:
            raw_json = json.load(file)

        # Flatten JSON and extract metadata
        flattened_data = []
        for attribute in raw_json:
            user_id = attribute.get('userId', None)
            seizure_times = attribute.get('seizureTimes', [])
            subtype = attribute.get('subType', None)
            stype = attribute.get('type', None)
            desc = attribute.get('desc', None)
            sample_freq = attribute.get('sampleFreq', 25)
            watch_sd_name = attribute.get('watchSdName', None)

            flattened_data.append({
                'userId': user_id,
                'subType': subtype,
                'type': stype,
                'desc': desc,
                'seizureTimes': seizure_times,
                'sampleFreq': sample_freq,
                'watchSdName': watch_sd_name,
            })

        self.df_metadata = pd.DataFrame(flattened_data)
        return self.df_metadata

    def load_sensordata(self):
        with open(self.file_path, 'r') as file:
            raw_json = json.load(file)

        # Flatten JSON and extract sensor data
        flattened_data = []
        for attribute in raw_json:
            user_id = attribute.get('userId', None)
            datapoints = attribute.get('datapoints', [])

            for point in datapoints:
                event_id = point.get('eventId', None)
                hr = point.get('hr', None)
                o2Sat = point.get('o2Sat', None)
                rawData = point.get('rawData', [])
                rawData3D = point.get('rawData3D', [])
                flattened_data.append({
                    'eventId': event_id,
                    'userId': user_id,
                    'hr': hr,
                    'o2Sat': o2Sat,
                    'rawData': rawData,
                    'rawData3D': rawData3D
                })

        self.df_sensordata = pd.DataFrame(flattened_data)
        return self.df_sensordata

    def calculate_fft_and_features(self):
        # Sampling frequency (25 Hz)
        sampling_rate = 25  # in Hz

        # Function to calculate additional features
        def calculate_additional_features(raw_data):
            # Calculate Skewness
            skewness = skew(raw_data)
            
            # Calculate Kurtosis
            kurt = kurtosis(raw_data)
            
            # Calculate Standard Deviation
            std_dev = np.std(raw_data)
            
            # Calculate Total Distance Traveled (using numerical integration of acceleration)
            time_step = 1 / sampling_rate  # time per sample in seconds
            distance_traveled = 0.5 * raw_data * time_step**2
            total_distance = np.sum(np.abs(distance_traveled))  # Sum of absolute distances
            
            return skewness, kurt, std_dev, total_distance

        # Function to calculate FFT and additional features for each row
        def calculate_fft_and_features_for_row(raw_data):
            # FFT calculation: Remove DC component (mean of the signal) and compute FFT
            raw_data = raw_data - np.mean(raw_data)
            fft_result = np.fft.fft(raw_data)
            frequencies = np.fft.fftfreq(len(raw_data), d=1/sampling_rate)
            fft_magnitude = np.abs(fft_result)
            
            # Only consider positive frequencies
            positive_frequencies = frequencies[:len(frequencies)//2]
            positive_fft_magnitude = fft_magnitude[:len(frequencies)//2]
            
            # Calculate additional features
            skewness, kurt, std_dev, total_distance = calculate_additional_features(raw_data)
            
            return positive_fft_magnitude, skewness, kurt, std_dev, total_distance

        # Process all rows and calculate FFT and additional features
        fft_results = []
        additional_features = []
        
        for _, row in self.df_sensordata.iterrows():
            raw_data = np.array(row['rawData'])
            
            # Calculate FFT and features for the current row
            fft_magnitude, skewness, kurt, std_dev, total_distance = calculate_fft_and_features_for_row(raw_data)
            
            # Store results
            fft_results.append(list(fft_magnitude))
            additional_features.append([skewness, kurt, std_dev, total_distance])
        
        # Convert the list of additional features to a DataFrame
        additional_features_df = pd.DataFrame(additional_features, columns=['Skewness', 'Kurtosis', 'StdDev', 'TotalDistance'])
        
        # Create a new DataFrame with FFT and additional features
        df_features = self.df_sensordata.copy()  # Copy original sensor data
        df_features['FFT'] = fft_results
        df_features = pd.concat([df_features, additional_features_df], axis=1)
        df_features['max_acceleration'] = df_features['rawData'].apply(lambda x: max(x))  # Calculate max value directly

        # Return the new DataFrame with features
        return df_features