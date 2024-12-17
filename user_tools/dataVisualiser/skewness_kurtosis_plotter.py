import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

class SkewnessKurtosis:
    def __init__(self, df_sensordata):
        if df_sensordata is None or df_sensordata.empty:
            raise ValueError("Sensor data DataFrame is empty or not loaded.")
        self.df_sensordata = df_sensordata

    def compute(self, event_id):
        # Filter rows for the given eventId
        filtered_data = self.df_sensordata[self.df_sensordata['eventId'] == event_id]
        if filtered_data.empty:
            raise ValueError(f"No data found for eventId {event_id}")

        # Lists to store skewness and kurtosis values for each row
        skewness_values = []
        kurtosis_values = []

        # Compute skewness and kurtosis for each row
        for _, row in filtered_data.iterrows():
            raw_data = row['rawData']
            if not raw_data:  # Skip if rawData is empty
                continue
            skewness_values.append(skew(raw_data))
            kurtosis_values.append(kurtosis(raw_data))

        return skewness_values, kurtosis_values

    def plot(self, event_id, skewness_values, kurtosis_values):
        if not skewness_values or not kurtosis_values:
            raise ValueError("Skewness or kurtosis values are empty. Please compute them first.")

        num_rows = len(skewness_values)
        time_axis = np.linspace(0, num_rows * 5, num_rows)  # Assuming each row covers 5 seconds

        # Plot skewness and kurtosis values
        plt.figure(figsize=(12, 6))

        # Plot skewness
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, skewness_values, label="Skewness", color='tab:blue')
        plt.title(f"Skewness and Kurtosis for EventID {event_id}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Skewness")
        plt.legend()
        plt.grid()

        # Plot kurtosis
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, kurtosis_values, label="Kurtosis", color='tab:red')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Kurtosis")
        plt.legend()
        plt.grid()

        # Display the plots
        plt.tight_layout()
        plt.show()
