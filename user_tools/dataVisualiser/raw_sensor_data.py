import matplotlib.pyplot as plt
import numpy as np

class RawDataPlotter:
    def __init__(self, event_id, data_loader):
        self.event_id = event_id
        self.data_loader = data_loader
        self.raw_data, self.raw_data3d, self.hr, self.o2_sat = self._load_data()
        self.raw_sample_rate = 125  # 125 samples per second for rawData and rawData3D
        self.hr_spo2_sample_interval = 5  # HR and SpO₂ data sampled every 5 seconds

    def _load_data(self):
        sensor_df = self.data_loader.df_sensordata
        filtered_data = sensor_df[sensor_df['eventId'] == self.event_id]

        if filtered_data.empty:
            raise ValueError(f"No data found for event ID {self.event_id}")

        # Extract rawData and rawData3D
        raw_data = np.hstack(filtered_data['rawData'].dropna().values) if 'rawData' in filtered_data else np.array([])
        raw_data3d = (
            np.vstack(filtered_data['rawData3D'].dropna().values)
            if 'rawData3D' in filtered_data and not filtered_data['rawData3D'].dropna().empty
            else np.array([])
        )

        # Extract HR and O2Sat data
        hr = filtered_data['hr'].dropna().tolist()
        o2_sat = filtered_data['o2Sat'].dropna().tolist()

        return raw_data, raw_data3d, hr, o2_sat

    def plot(self):
        # Determine layout based on available data
        if self.raw_data3d.size > 0:  # If rawData3D is available
            subplot_ratios = [1, 1, 1]  # 33% each
            nrows = 3
        else:  # Only rawData and HR/SpO₂
            subplot_ratios = [1, 1]  # 50% each
            nrows = 2

        # Total number of rows for the event (each row is 5 seconds of data)
        sensor_df = self.data_loader.df_sensordata
        filtered_data = sensor_df[sensor_df['eventId'] == self.event_id]
        num_rows = len(filtered_data)

        if num_rows == 0:
            raise ValueError(f"No data to plot for event ID {self.event_id}")

        # Calculate the total duration for the event
        total_duration = num_rows * 5  # Each row represents 5 seconds

        # Generate the x-axis points
        time_raw_data = np.linspace(0, total_duration, len(self.raw_data))
        time_raw_data3d = np.linspace(0, total_duration, len(self.raw_data3d))
        time_hr_spo2 = np.arange(0, len(self.hr) * self.hr_spo2_sample_interval, self.hr_spo2_sample_interval)

        # Create figure with dynamic subplot ratios and add hspace for whitespace
        fig, axs = plt.subplots(
            nrows,
            1,
            figsize=(14, 10),
            gridspec_kw={'height_ratios': subplot_ratios, 'hspace': 0.5}  # Add space between plots
        )
        fig.suptitle(f"Sensor Data Overview for Event ID {self.event_id}", fontsize=12, fontweight='bold')

        # Plot rawData on the first subplot
        axs[0].plot(time_raw_data, self.raw_data, label="Raw Data (1D)", color='tab:blue', alpha=0.7)
        axs[0].set_title("Raw Data (1D)")
        axs[0].set_xlabel("Time (seconds)")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend()
        axs[0].grid()

        if self.raw_data3d.size > 0:  # Plot rawData3D on the second subplot if available
            axs[1].plot(time_raw_data3d, self.raw_data3d[:, 0], label="X-axis (RawData3D)", color='tab:orange', alpha=0.7)
            axs[1].plot(time_raw_data3d, self.raw_data3d[:, 1], label="Y-axis (RawData3D)", color='tab:green', alpha=0.7)
            axs[1].plot(time_raw_data3d, self.raw_data3d[:, 2], label="Z-axis (RawData3D)", color='tab:red', alpha=0.7)
            axs[1].set_title("Raw Data (3D)")
            axs[1].set_xlabel("Time (seconds)")
            axs[1].set_ylabel("Amplitude")
            axs[1].legend()
            axs[1].grid()

        # Plot HR and SpO₂ on the last subplot
        hr_spo2_ax_index = 2 if self.raw_data3d.size > 0 else 1
        if self.hr and self.o2_sat:
            axs[hr_spo2_ax_index].plot(time_hr_spo2, self.hr, label="Heart Rate (HR)", color='tab:purple', linewidth=2)
            axs[hr_spo2_ax_index].plot(time_hr_spo2, self.o2_sat, label="Oxygen Saturation (SpO₂)", color='tab:cyan', linewidth=2)
        axs[hr_spo2_ax_index].set_title("Heart Rate and Oxygen Saturation")
        axs[hr_spo2_ax_index].set_xlabel("Time (seconds)")
        axs[hr_spo2_ax_index].set_ylabel("Value")
        axs[hr_spo2_ax_index].legend()
        axs[hr_spo2_ax_index].grid()

        # Adjust layout and show the plot
        plt.show()
