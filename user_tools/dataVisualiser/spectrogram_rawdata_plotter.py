import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa.display
import pandas as pd
import json

class SpectrogramRawDataPlotter:
    def __init__(self, df_sensordata, sampling_rate=125):
        """
        Initializes the SpectrogramRawDataPlotter.

        Args:
            df_sensordata (pd.DataFrame): Sensor data DataFrame.
            sampling_rate (int): Sampling rate of the data in Hz.
        """
        self.df_sensordata = df_sensordata
        self.sampling_rate = sampling_rate

    def plot_for_event(self, event_id, ax=None, reduce_font=False):
        """
        Plots the Mel spectrogram and raw data for a specific event ID.

        Args:
            event_id (int): The event ID to filter and plot.
            ax (matplotlib.axes._subplots.AxesSubplot): Optional. Axis to plot on.
            reduce_font (bool): Reduce font size for multiple plots.
        """
        # Filter rows for the specified event ID
        filtered_data = self.df_sensordata[self.df_sensordata['eventId'] == event_id]

        if filtered_data.empty:
            print(f"No data found for eventId {event_id}")
            return

        # Calculate total time (in seconds)
        num_rows = len(filtered_data)
        total_time = num_rows * 5  # Each row represents 5 seconds

        # Generate x-axis points for each rawData point
        time_points = np.linspace(0, total_time, num_rows * 125)  # 125 samples per row

        # Flatten the rawData column for plotting and Mel Spectrogram
        raw_data = np.hstack(filtered_data['rawData'].values)

        # Compute Short-Time Fourier Transform (STFT)
        _, _, Zxx = signal.stft(raw_data, fs=self.sampling_rate, nperseg=256)

        # Apply Mel filter bank and convert to dB scale
        mel_spectrogram = librosa.feature.melspectrogram(S=np.abs(Zxx), sr=self.sampling_rate, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Normalize the x-axis of the spectrogram to match total_time
        spectrogram_time = np.linspace(0, total_time, log_mel_spectrogram.shape[1])

        # If no axis is provided, create a new plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        # Plot the Mel Spectrogram
        mappable = ax.pcolormesh(
            spectrogram_time,
            np.linspace(0, mel_spectrogram.shape[0], mel_spectrogram.shape[0]),
            log_mel_spectrogram,
            shading='auto',
            cmap='viridis'
        )
        ax.set_title(f'EventID {event_id}', fontsize=8 if reduce_font else 12)
        ax.set_xlabel("Time (seconds)", fontsize=8 if reduce_font else 12)
        ax.set_ylabel("Mel Freq (Hz)", fontsize=8 if reduce_font else 12)
        ax.tick_params(axis='both', labelsize=6 if reduce_font else 10)

        # Create a second y-axis for the raw acceleration data
        ax2 = ax.twinx()
        ax2.plot(time_points, raw_data, color='tab:red', alpha=0.6, label='Raw Data')
        ax2.set_ylabel("Raw Acceleration ~ milli-g", fontsize=8 if reduce_font else 12)
        ax2.tick_params(axis='both', labelsize=6 if reduce_font else 10)

    def plot_multiple_events(self, event_ids, rows=1):
        """
        Plots multiple Mel spectrograms and raw data for a list of event IDs.

        Args:
            event_ids (list of int): List of event IDs to plot.
            rows (int): Number of rows of plots (3 per row).
        """
        # Number of plots per row
        cols = 3

        # Calculate the number of plots
        total_plots = len(event_ids)
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
        axes = axes.flatten()  # Flatten the axes array to make indexing easier

        for i, event_id in enumerate(event_ids):
            if i < len(axes):
                self.plot_for_event(event_id, ax=axes[i], reduce_font=True)
            else:
                break

        # Hide any extra subplots if they exist
        for j in range(len(event_ids), len(axes)):
            axes[j].axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
