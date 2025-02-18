import pandas as pd
import numpy as np
import logging

class DataRestructurer:
    def __init__(self, input_file, output_file, timestep=125):
        self.input_file = input_file
        self.output_file = output_file
        self.timestep = timestep
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger()

    def restructure_data(self):
        self.logger.info("Starting data restructuring process.")
        
        # Load the data
        df = pd.read_csv(self.input_file)
        self.logger.info(f"Loaded data from {self.input_file} with {len(df)} rows.")
        
        # Check if required columns exist
        required_columns = {'eventId', 'userId', 'label', 'x', 'y', 'z', 'magnitude'}
        if not required_columns.issubset(df.columns):
            missing_columns = required_columns - set(df.columns)
            self.logger.error(f"Missing required columns in input CSV: {missing_columns}")
            raise ValueError(f"Missing required columns in input CSV: {missing_columns}")
        
        # Initialize lists to store transformed data
        new_data = []
        new_id = 1  # Start new ID sequence
        
        # Iterate over the dataframe in steps of TIMESTEP rows
        for i in range(0, len(df), self.timestep):
            chunk = df.iloc[i:i+self.timestep]  # Get TIMESTEP-row chunk
            
            if len(chunk) < self.timestep:
                self.logger.info(f"Skipping incomplete chunk at index {i}, size {len(chunk)}.")
                break  # Ignore incomplete timesteps at the end
            
            # Extract values
            event_id = chunk.iloc[0]['eventId']
            user_id = chunk.iloc[0]['userId']
            label = chunk.iloc[0]['label']
            
            # Process magnitude values
            magnitude_values = chunk['magnitude'].tolist()
            
            # Process x, y, z values
            raw_data_3d = np.array(chunk[['x', 'y', 'z']]).flatten().tolist()
            
            # Append new row to transformed data
            new_data.append([event_id, user_id, label, magnitude_values, raw_data_3d, new_id])
            new_id += 1  # Increment ID
            
            #self.logger.info(f"Processed chunk {new_id - 1} for eventId {event_id}.")
        
        # Create new dataframe
        columns = ['eventId', 'userId', 'label', 'magnitude', 'rawData3d', 'Id']
        df_transformed = pd.DataFrame(new_data, columns=columns)
        
        # Save to CSV, converting lists to strings
        df_transformed.to_csv(self.output_file, index=False)
        self.logger.info(f"Data saved to {self.output_file}.")
    
    def run(self):
        try:
            self.restructure_data()
            self.logger.info("Data restructuring process completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during restructuring process: {e}")
