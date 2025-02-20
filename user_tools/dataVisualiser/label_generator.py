import pandas as pd
import numpy as np
import json

class LabelGenerator:
    def __init__(self, file_path, sampling_rate=25):
        self.file_path = file_path  # Path to the JSON file
        self.sampling_rate = sampling_rate  # Sampling rate (Hz)
        self.df_sensordata = None  # To store the processed DataFrame
        
    def load_data(self):
        """Load and flatten the JSON data into a DataFrame."""
        with open(self.file_path, 'r') as file:
            raw_json = json.load(file)
        
        flattened_data = []
        for attribute in raw_json:
            user_id = attribute.get('userId', None)
            seizure_times = attribute.get('seizureTimes', [])
            datapoints = attribute.get('datapoints', [])
            
            for point in datapoints:
                event_id = point.get('eventId', None)
                hr = point.get('hr', [])
                o2Sat = point.get('o2Sat', [])
                rawData = point.get('rawData', [])
                rawData3D = point.get('rawData3D', [])
                alarmPhrase = point.get('alarmPhrase', None)
                flattened_data.append({
                    'eventId': event_id,
                    'userId': user_id,
                    'hr': hr,
                    'o2Sat': o2Sat,
                    'rawData': rawData,
                    'rawData3D': rawData3D,
                    'seizure_times': seizure_times,
                    'alarmPhrase': alarmPhrase
                })
        
        # Convert to DataFrame
        self.df_sensordata = pd.DataFrame(flattened_data)
        
    def calculate_fft(self, raw_data):
        """Calculate FFT for the raw data."""
        raw_data = raw_data - np.mean(raw_data)  # Remove the DC component
        fft_result = np.fft.fft(raw_data)  # Compute FFT
        frequencies = np.fft.fftfreq(len(raw_data), d=1/self.sampling_rate)  # Compute frequencies
        fft_magnitude = np.abs(fft_result)  # Compute the magnitude
        positive_frequencies = frequencies[:len(frequencies)//2]  # Only positive frequencies
        positive_fft_magnitude = fft_magnitude[:len(frequencies)//2]  # Only positive FFT magnitudes
        return positive_frequencies, positive_fft_magnitude
    
    def add_fft_column(self):
        """Add an FFT column to the DataFrame."""
        fft_results = []
        for _, row in self.df_sensordata.iterrows():
            raw_data = np.array(row['rawData'])
            _, positive_fft_magnitude = self.calculate_fft(raw_data)  # Calculate FFT for the row
            fft_results.append(list(positive_fft_magnitude))  # Append FFT result
        self.df_sensordata['FFT'] = fft_results
    
    def add_timestep_and_label(self):
        """Add timestep and label columns to the DataFrame."""
        # Add 'timestep' column in 5-second increments
        self.df_sensordata['timestep'] = self.df_sensordata.index * 5

        # Add 'label' column, initialized to 0
        self.df_sensordata['label'] = 0
    
    def label_alarm_events(self):
        """Label the data based on alarm events."""
        for idx, row in self.df_sensordata.iterrows():
            if row['alarmPhrase'] == 'ALARM':  # If alarmPhrase is ALARM
                alarm_time = row['timestep']
                seizure_times = row['seizure_times']
                
                # Process the seizure times list in seconds
                for seizure in seizure_times:
                    start_time = alarm_time + seizure  # Adjust by the seizure offset
                    
                    # Label the rows before and after the alarm (within the range of seizure_times)
                    before_idx = self.df_sensordata[(self.df_sensordata['timestep'] >= start_time) &
                                                     (self.df_sensordata['timestep'] < alarm_time)].index
                    self.df_sensordata.loc[before_idx, 'label'] = 1  # Mark as seizure (1)
                    
                    # For the positive offset (after alarm)
                    after_idx = self.df_sensordata[(self.df_sensordata['timestep'] >= alarm_time) &
                                                    (self.df_sensordata['timestep'] <= start_time)].index
                    self.df_sensordata.loc[after_idx, 'label'] = 1  # Mark as seizure (1)
    
    def process_data(self):
        """Process the data through all stages and return the final DataFrame."""
        # Step 1: Load the data
        self.load_data()

        # Step 2: Add FFT column
        self.add_fft_column()

        # Step 3: Add timestep and label columns
        self.add_timestep_and_label()

        # Step 4: Label based on alarm events
        self.label_alarm_events()

        # Step 5: Drop the 'seizure_times' column
        self.df_sensordata.drop(columns=['seizure_times'], inplace=True)

        return self.df_sensordata


# Example usage
file_path = '../../tests/testData/testDataVisualisation.json'  # Replace with your JSON file path
processor = LabelGenerator(file_path)

# Process the data and get the resulting DataFrame
df_result = processor.process_data()

# Optionally save the DataFrame to a CSV file
df_result.to_csv('generatedCsvDatasets/sensordata_labeled.csv', index=False)

# Display the first few rows of the processed DataFrame
print(df_result.head(30))
