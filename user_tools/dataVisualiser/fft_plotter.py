import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader

class FFTProcessor:
    def __init__(self, event_id, sampling_rate=25, roi=(3, 8)):
        """
        Initialize the FFTProcessor class.

        :param event_id: The EventID to filter the data.
        :param sampling_rate: Sampling rate of the signal in Hz.
        :param roi: Tuple specifying the region of interest (start and end frequencies in Hz).
        """
        self.event_id = event_id
        self.sampling_rate = sampling_rate
        self.roi_start, self.roi_end = roi

    def process_event(self, df_sensordata):
        """
        Process the FFT for the specified EventID.

        :param df_sensordata: DataFrame containing the sensor data.
        :return: Tuple containing positive frequencies and FFT magnitude.
        """
        # Filter rows for the specified eventId
        filtered_data = df_sensordata[df_sensordata['eventId'] == self.event_id]

        # Flatten the rawData column for all rows
        raw_data = np.hstack(filtered_data['rawData'].values)

        # Remove the DC component
        raw_data = raw_data - np.mean(raw_data)

        # Compute the Fourier Transform (FFT) for the entire signal
        fft_result = np.fft.fft(raw_data)

        # Compute the frequencies corresponding to the FFT result
        frequencies = np.fft.fftfreq(len(raw_data), d=1 / self.sampling_rate)

        # Compute the magnitude of the FFT (absolute value)
        fft_magnitude = np.abs(fft_result)

        # Only consider the positive frequencies (the FFT is symmetric)
        positive_frequencies = frequencies[:len(frequencies) // 2]
        positive_fft_magnitude = fft_magnitude[:len(frequencies) // 2]

        return positive_frequencies, positive_fft_magnitude

    def plot_fft(self, positive_frequencies, positive_fft_magnitude):
        """
        Plot the FFT results.

        :param positive_frequencies: Array of positive frequencies.
        :param positive_fft_magnitude: Array of FFT magnitudes for positive frequencies.
        """
        # Find indices of the ROI frequencies
        roi_indices = np.where(
            (positive_frequencies >= self.roi_start) & (positive_frequencies <= self.roi_end)
        )[0]

        # Plot the FFT for the full spectrum and the 3-8 Hz ROI
        plt.figure(figsize=(12, 6))

        # Full spectrum plot
        plt.subplot(2, 1, 1)
        plt.plot(positive_frequencies, positive_fft_magnitude, color='tab:blue')
        plt.title(f"Fourier Transform of EventID {self.event_id} - Full Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid()

        # Zoomed-in plot (3-8 Hz region)
        plt.subplot(2, 1, 2)
        plt.plot(
            positive_frequencies[roi_indices],
            positive_fft_magnitude[roi_indices],
            color='tab:red',
        )
        plt.title(f"Fourier Transform of EventID {self.event_id} - 3-8 Hz Region of Interest")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid()

        plt.tight_layout()
        plt.show()


# Main method
if __name__ == "__main__":
    # Load data using the DataLoader
    file_path = 'Scripts/osdb_3min_allSeizures.json'  # Replace with the actual file path
    data_loader = DataLoader(file_path=file_path)
    df_sensordata = data_loader.load_sensordata()

    # Initialize and use the FFTProcessor class
    event_id = 11591
    fft_processor = FFTProcessor(event_id=event_id)

    # Process the data
    positive_frequencies, positive_fft_magnitude = fft_processor.process_event(df_sensordata)

    # Plot the results
    fft_processor.plot_fft(positive_frequencies, positive_fft_magnitude)
