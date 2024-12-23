import numpy as np
import matplotlib.pyplot as plt

class FrequencyBandEnergy:
    def __init__(self, dataframe):
        """
        Initialize the FrequencyBandEnergy class with a DataFrame.
        :param dataframe: Pandas DataFrame containing the sensor data.
        """
        self.dataframe = dataframe

    def compute_frequency_bands(self, signal, fs=125):
        """
        Compute energy in frequency bands using FFT.
        :param signal: 1D array representing raw sensor data.
        :param fs: Sampling frequency (default is 125 Hz).
        :return: Tuple containing energy in low, mid, and high frequency bands.
        """
        # Compute FFT of the signal
        fft_values = np.fft.fft(signal)
        fft_freqs = np.fft.fftfreq(len(signal), 1 / fs)
        
        # Get the magnitude of the FFT
        fft_magnitude = np.abs(fft_values)
        
        # Define frequency bands (in Hz)
        low_freq_band = (0, 2)     # Low frequency: 0-2 Hz
        mid_freq_band = (2, 10)    # Mid frequency: 2-10 Hz
        high_freq_band = (10, 25)  # High frequency: 10-25 Hz
        
        # Compute energy in each frequency band
        low_energy = np.sum(fft_magnitude[(fft_freqs >= low_freq_band[0]) & (fft_freqs <= low_freq_band[1])])
        mid_energy = np.sum(fft_magnitude[(fft_freqs >= mid_freq_band[0]) & (fft_freqs <= mid_freq_band[1])])
        high_energy = np.sum(fft_magnitude[(fft_freqs >= high_freq_band[0]) & (fft_freqs <= high_freq_band[1])])
        
        return low_energy, mid_energy, high_energy

    def compute_energy_for_event(self, event_id):
        """
        Compute energy for each 5-second window for a specific event ID.
        :param event_id: The event ID for which to compute the energy.
        :return: Lists of low, mid, and high energy values.
        """
        # Filter rows for the specific event ID
        filtered_data = self.dataframe[self.dataframe['eventId'] == event_id]
        num_rows = len(filtered_data)

        # Initialize lists to store energy values
        low_energy_list = []
        mid_energy_list = []
        high_energy_list = []

        # Compute energy for each row
        for i in range(num_rows):
            row_data = filtered_data.iloc[i]['rawData']
            low_energy, mid_energy, high_energy = self.compute_frequency_bands(row_data)
            low_energy_list.append(low_energy)
            mid_energy_list.append(mid_energy)
            high_energy_list.append(high_energy)

        return low_energy_list, mid_energy_list, high_energy_list

    def plot_energy(self, event_id):
        """
        Plot the relative energy of the frequency bands over time for a specific event ID.
        :param event_id: The event ID for which to plot the energy.
        """
        low_energy_list, mid_energy_list, high_energy_list = self.compute_energy_for_event(event_id)
        num_rows = len(low_energy_list)
        time_axis = np.arange(num_rows) * 5  # Each row represents a 5-second interval

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, low_energy_list, label="Low Frequency (0-2 Hz)", color='tab:blue')
        plt.plot(time_axis, mid_energy_list, label="Mid Frequency (2-10 Hz)", color='tab:orange')
        plt.plot(time_axis, high_energy_list, label="High Frequency (10-25 Hz)", color='tab:green')

        # Add labels and title
        plt.title(f"Frequency Band Energy for EventID {event_id}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Energy")
        plt.legend()
        plt.grid()

        # Display the plot
        plt.show()
