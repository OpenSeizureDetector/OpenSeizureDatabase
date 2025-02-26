import os
import pandas as pd
import numpy as np
import logging
import argparse
from activity_data_osdb_processor import ActivityDataOSDBProcessor
from data_analyser import DataAnalyser
from data_restructurer import DataRestructurer
from dataset_generator import DatasetGenerator  # Import the new class

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, 'Data', 'Dataset')
DEFAULT_OUTPUT_CSV = os.path.join(BASE_DIR, 'Datasets', 'activity_dataset.csv')
DEFAULT_INPUT_CSV = os.path.join(BASE_DIR, 'Datasets', 'activity_dataset.csv')
DEFAULT_SAMPLE_OSDB_CSV = os.path.join(BASE_DIR, 'Datasets', 'sample_osdb_data.csv')
DEFAULT_RESTRUCTURED_CSV = os.path.join(BASE_DIR, 'Datasets', 'restructured_osdb_data.csv')
DEFAULT_OUTPUT_COMBINED_CSV = os.path.join(BASE_DIR, 'Datasets', 'seizure_activity_combined.csv')


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_activity_data(data_dir, output_csv):
    processor = ActivityDataOSDBProcessor(data_dir, output_csv)
    downsampled_df = processor.process_files()
    print(downsampled_df.head())

def analyze_data(input_csv):
    analyser = DataAnalyser(input_csv)
    analyser.analyze_data()

def restructure_data(input_csv, output_csv, timestep):
    restructurer = DataRestructurer(input_csv, output_csv, timestep)
    restructurer.run()

def generate_seizure_activity_dataset(data_dir_1, data_dir_2, output_csv):
    generator = DatasetGenerator(data_dir_1, data_dir_2)
    generator.save_combined_dataset(output_csv)


def main():
    parser = argparse.ArgumentParser(description="OSDB Data Processing Pipeline")
    parser.add_argument("--process_data", action="store_true", help="Process activity data")
    parser.add_argument("--analyze_data", action="store_true", help="Analyze processed data")
    parser.add_argument("--restructure_data", action="store_true", help="Restructure timeseries data")
    parser.add_argument("--generate_data", action="store_true", help="Generate combined seizure and activity dataset")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing input data")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Output CSV file path")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV, help="Input CSV file path for analysis/restructuring")
    parser.add_argument("--sample_osdb_data_csv", type=str, default=DEFAULT_SAMPLE_OSDB_CSV, help="Sample OSDB dataset")
    parser.add_argument("--restructured_csv", type=str, default=DEFAULT_RESTRUCTURED_CSV, help="Output CSV file for restructured data")
    parser.add_argument("--timestep", type=int, default=125, help="Number of rows per timestep in restructuring")
    parser.add_argument("--output_combined_csv", type=str, default=DEFAULT_OUTPUT_COMBINED_CSV, help="Output CSV file path for the combined dataset")
    
    args = parser.parse_args()
    
    # Ensure only one action is triggered
    if args.process_data:
        process_activity_data(args.data_dir, args.output_csv)
    elif args.analyze_data:
        analyze_data(args.input_csv)
    elif args.restructure_data:
        restructure_data(args.input_csv, args.restructured_csv, args.timestep)
    elif args.generate_data:
        # Ensure to use the correct directories for seizure and activity data
        if os.path.exists(args.data_dir) and os.path.isdir(args.data_dir):
            generate_seizure_activity_dataset(args.sample_osdb_data_csv, args.input_csv, args.output_combined_csv)
        else:
            print(f"Error: The directory {args.data_dir} does not exist.")
    else:
        print("Error: You must specify one of the actions (--process_data, --analyze_data, --restructure_data, or --generate_data).")

if __name__ == "__main__":
    main()
