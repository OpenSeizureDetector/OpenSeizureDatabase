import numpy as np
import matplotlib.pyplot as plt

class StdAndMaxAcceleration:
    def __init__(self, data_loader, event_id):
        self.data_loader = data_loader
        self.event_id = event_id
        self.df_sensordata = None
        self.filtered_data = None
        self.raw_data = None
        self.time_axis = None
        self.std_values = []
        self.max_acceleration = []

    def load_data(self):
        # Load the sensor data using the DataLoader class
        self.df_sensordata = self.data_loader.load_sensordata()
        
        # Filter the data by eventId
        self.filtered_data = self.df_sensordata[self.df_sensordata['eventId'] == self.event_id]

        # Extract rawData (1D array) and ensure it's not empty
        self.raw_data = np.hstack(self.filtered_data['rawData'].dropna().values) if 'rawData' in self.filtered_data else np.array([])

        # Ensure there is data for rawData
        if self.raw_data.size == 0:
            raise ValueError(f"No rawData found for the given eventId {self.event_id}")

        # Number of timesteps (assuming 30 timesteps)
        num_timesteps = len(self.filtered_data)

        # Time axis (assuming 5 seconds per timestep)
        self.time_axis = np.arange(0, num_timesteps * 5, 5)

        # Calculate standard deviation and max acceleration per timestep
        self.calculate_features(num_timesteps)

    def calculate_features(self, num_timesteps):
        # Assuming each timestep has a fixed number of data points
        timesteps_length = len(self.raw_data) // num_timesteps

        for i in range(num_timesteps):
            # Get the data for the current timestep
            timestep_data = self.raw_data[i * timesteps_length:(i + 1) * timesteps_length]
            
            # Calculate the standard deviation for this timestep
            self.std_values.append(np.std(timestep_data))
            
            # Calculate the maximum acceleration for this timestep
            self.max_acceleration.append(np.max(timestep_data))

    def plot_features(self):
        # Plot the STD and max acceleration in two subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot the standard deviation for each timestep
        axs[0].plot(self.time_axis, self.std_values, label="Standard Deviation", color='tab:blue')
        axs[0].set_ylabel("Standard Deviation")
        axs[0].set_title("Standard Deviation per Timestep")
        axs[0].grid(True)
        axs[0].legend()

        # Plot the maximum acceleration for each timestep
        axs[1].plot(self.time_axis, self.max_acceleration, label="Max Acceleration", color='tab:red')
        axs[1].set_xlabel("Time (seconds)")
        axs[1].set_ylabel("Max Acceleration")
        axs[1].set_title("Maximum Acceleration per Timestep")
        axs[1].grid(True)
        axs[1].legend()

        # Adjust the layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()

