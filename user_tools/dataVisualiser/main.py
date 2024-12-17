# Import necessary libraries and classes
from data_loader import DataLoader
from spectrogram_rawdata_plotter import SpectrogramRawDataPlotter
from skewness_kurtosis import SkewnessKurtosis
from frequency_band_energy import FrequencyBandEnergy
from raw_sensor_data import RawDataPlotter
from fft_processor import FFTProcessor  # Import the FFTProcessor class
from std_and_max_acceleration import StdAndMaxAcceleration


# -------------- USER CONFIGURABLE PARAMETERS --------------
# Path to the JSON file containing sensor data
file_path = 'datasets/osdb_3min_allSeizures.json'  # Modify this path if needed

# Event ID for the analysis (modify this for different events)
event_id = 407  # Modify this to your target Event ID

# You can also add a list of event IDs for multi-event analysis below
event_ids = [407, 764, 4924, 5483, 5486, 9401]  # Modify this list as needed for multiple events
# ----------------------------------------------------------

# Load the data using DataLoader
try:
    print(f"Loading data from {file_path}...")
    data_loader = DataLoader(file_path=file_path)
    data_loader.load_sensordata()
    print("Sensor data loaded successfully.")
except Exception as e:
    print(f"Error loading data from {file_path}: {e}")
    raise

# ----------------------------------------------------------
# 1. Plot Raw Sensor Data
try:
    print(f"Plotting raw sensor data for Event ID {event_id}...")
    data_plotter = RawDataPlotter(event_id=event_id, data_loader=data_loader)
    data_plotter.plot()
except Exception as e:
    print(f"Error plotting raw sensor data for Event ID {event_id}: {e}")

# ----------------------------------------------------------
try:
    print(f"Processing FFT using FFTProcessor for Event ID {event_id}...")
    # Initialize FFTProcessor
    fft_processor = FFTProcessor(event_id=event_id)

    # Process the data
    positive_frequencies, positive_fft_magnitude = fft_processor.process_event(data_loader.df_sensordata)

    # Plot the results
    fft_processor.plot_fft(positive_frequencies, positive_fft_magnitude)
except Exception as e:
    print(f"Error processing FFT using FFTProcessor for Event ID {event_id}: {e}")

# ----------------------------------------------------------
# 4. Plot Spectrogram for Raw Data (Single Event)
try:
    print(f"Plotting spectrogram for Event ID {event_id}...")
    mel_plotter = SpectrogramRawDataPlotter(data_loader.df_sensordata)
    mel_plotter.plot_for_event(event_id=event_id)
except Exception as e:
    print(f"Error plotting spectrogram for Event ID {event_id}: {e}")

# ----------------------------------------------------------
# 5. Plot Spectrogram for Multiple Events
try:
    if event_ids:  # Only plot for multiple events if event_ids is not empty
        print(f"Plotting spectrograms for multiple Event IDs: {event_ids}...")
        mel_plotter.plot_multiple_events(event_ids, rows=2)  # Modify rows if necessary
except Exception as e:
    print(f"Error plotting multiple spectrograms: {e}")

# ----------------------------------------------------------
# 6. Compute and Plot Skewness and Kurtosis
try:
    print(f"Computing skewness and kurtosis for Event ID {event_id}...")
    skew_kurt = SkewnessKurtosis(data_loader.df_sensordata)
    skewness_values, kurtosis_values = skew_kurt.compute(event_id)
    skew_kurt.plot(event_id, skewness_values, kurtosis_values)
except ValueError as e:
    print(f"Value error computing skewness/kurtosis for Event ID {event_id}: {e}")
except Exception as e:
    print(f"Error computing skewness/kurtosis for Event ID {event_id}: {e}")

# ----------------------------------------------------------
# 7. Plot Frequency Band Energy for Event ID
try:
    print(f"Plotting frequency band energy for Event ID {event_id}...")
    freq_energy = FrequencyBandEnergy(data_loader.df_sensordata)
    freq_energy.plot_energy(event_id)
except Exception as e:
    print(f"Error plotting frequency band energy for Event ID {event_id}: {e}")
    
# ----------------------------------------------------------
# 8. Plot STD and Max Acceleration
try:
    print(f"Plotting frequency band energy for Event ID {event_id}...")
    std_plotter = StdAndMaxAcceleration(data_loader, event_id)
    # Load and process the data
    std_plotter.load_data()
    # Plot the features (standard deviation and max acceleration)
    std_plotter.plot_features()
except Exception as e:
    print(f"Error plotting frequency band energy for Event ID {event_id}: {e}") 

# ----------------------------------------------------------
# Final Completion Message
print("All tasks completed.")
