import pandas as pd
import os
import logging

class DatasetGenerator:
    def __init__(self, data_dir_1, data_dir_2):
        self.data_dir_1 = data_dir_1
        self.data_dir_2 = data_dir_2
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def generate_dataset(self):
        # Check if files exist before loading
        if not os.path.exists(self.data_dir_1) or not os.path.exists(self.data_dir_2):
            raise FileNotFoundError(f"One or both directories do not exist: {self.data_dir_1}, {self.data_dir_2}")

        # Load the first dataframe (sample_osdb_data)
        logging.info("Loading seizure sample data from %s", self.data_dir_1)
        df1 = pd.read_csv(self.data_dir_1)
        logging.info("Seizure sample data loaded successfully. Shape: %s", df1.shape)

        # Load the second dataframe (activity_dataset)
        logging.info("Loading activity dataset from %s", self.data_dir_2)
        df2 = pd.read_csv(self.data_dir_2)
        logging.info("Activity dataset loaded successfully. Shape: %s", df2.shape)

        # Step 1: Drop the 'Id' column from both dataframes
        logging.info("Dropping 'Id' column from both datasets")
        df1 = df1.drop(columns=['Id'])
        df2 = df2.drop(columns=['Id'])

        # Step 2: Reset the index for both dataframes to avoid any index conflicts
        logging.info("Resetting index for both datasets")
        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)

        # Step 3: Concatenate the dataframes with the index reset
        logging.info("Concatenating seizure and activity datasets")
        df_combined = pd.concat([df1, df2], ignore_index=True)

        # Step 4: Add the 'Id' column back (starting from 1)
        logging.info("Re-adding 'Id' column")
        df_combined['Id'] = df_combined.index + 1

        # Step 5: Reorder the columns as required
        logging.info("Reordering columns to match required format")
        df = df_combined[['eventId', 'Id', 'userId', 'x', 'y', 'z', 'magnitude', 'label']]

        # Return the generated dataframe
        return df

    def save_combined_dataset(self, output_path):
        # Generate the dataset
        logging.info("Generating combined seizure and activity dataset")
        df = self.generate_dataset()

        # Save the dataset to CSV
        logging.info("Saving combined dataset to %s", output_path)
        df.to_csv(output_path, index=False)
        logging.info("Activity seizure data pipeline complete. Dataset saved to %s", output_path)
