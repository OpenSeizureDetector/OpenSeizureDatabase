import os
import pandas as pd
import numpy as np
import logging

class ActivityDataOSDBProcessor:
    def __init__(self, data_dir, output_csv):
        self.data_dir = data_dir
        self.output_csv = output_csv
        logging.info("ActivityDataOSDBProcessor initialized.")
    
    def decode_accelerometer(self, coded_val):
        return -1.5 + (coded_val / 63) * 3.0
    
    def extract_user_id(self, file_name):
        parts = file_name.split('-')
        return parts[-1].split('.')[0]
    
    def process_files(self):
        all_data = []
        event_id = 0
        txt_files = []

        for root, dirs, files in os.walk(self.data_dir):
            dirs[:] = [d for d in dirs if not d.endswith('_MODEL')]
            txt_files.extend(
                [(os.path.join(root, f), os.path.basename(root), self.extract_user_id(f)) 
                 for f in files if f.endswith('.txt') and f not in ['README.txt', 'MANUAL.txt'] and not f.endswith('_MODEL.txt')]
            )
        
        logging.info(f"Found {len(txt_files)} human activities to process.")

        for file_path, activity_label, user_id in txt_files:
            event_id += 1
            
            try:
                data = pd.read_csv(file_path, sep='\\s+', header=None, names=['x', 'y', 'z'])
            except (pd.errors.ParserError, UnicodeDecodeError) as e:
                logging.error(f"Error reading {file_path}: {e}")
                continue
            
            original_length = len(data)  # Original row count before downsampling
            
            data[['x', 'y', 'z']] = data[['x', 'y', 'z']].map(self.decode_accelerometer)
            time_index = pd.date_range(start='2023-01-01', periods=original_length, freq='31.25ms')
            data.index = time_index
            
            # Downsample to 25Hz (40ms interval)
            data_resampled = data.resample('40ms').mean()
            downsampled_length = len(data_resampled)  # Row count after downsampling

            data_resampled['eventId'] = int(f"99999{event_id}")  # Ensure eventId remains an integer
            data_resampled['Id'] = np.arange(1, downsampled_length + 1)
            data_resampled['userId'] = user_id
            data_resampled[['x', 'y', 'z']] *= 1000
            
            data_resampled['magnitude'] = np.sqrt(
                data_resampled['x']**2 + data_resampled['y']**2 + data_resampled['z']**2
            )
            
            data_resampled.reset_index(drop=True, inplace=True)
            data_resampled['label'] = activity_label
            
            
            # Ensure each event is a multiple of 125 rows
            truncated_length = downsampled_length  # Default to downsampled length
            if downsampled_length % 125 != 0:
                truncated_length = downsampled_length - (downsampled_length % 125)
                data_resampled = data_resampled.iloc[:truncated_length]
            
            all_data.append(data_resampled)
            
            # Print event summary in a single row
            print(f"Event {event_id} | {original_length} rows @32Hz | Downsampled: {downsampled_length} rows @ 25Hz | Truncated: {truncated_length} rows | Timesteps: {truncated_length/125}")
        
        logging.info("All files successfully converted from text to DataFrame.")
        
        if not all_data:
            logging.warning("No valid data processed. Exiting.")
            return None
        
        final_df = pd.concat(all_data, ignore_index=True)
        logging.info("OSDB Downsampling and transformations completed.")
        
        # Reorder the columns to match the required order
        final_df = final_df[['eventId', 'Id', 'userId', 'x', 'y', 'z', 'magnitude', 'label']]
        
        final_df.to_csv(self.output_csv, index=False)
        logging.info(f"Saved processed dataframe to {self.output_csv}")
        
        return final_df
