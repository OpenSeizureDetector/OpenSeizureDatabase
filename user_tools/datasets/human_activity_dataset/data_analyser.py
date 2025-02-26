import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import random

class DataAnalyser:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        logging.info("DataAnalyser initialized with dataset.")

    def analyze_data(self):
        """Performs basic data analysis and prints key metrics."""
        logging.info("Performing dataset analysis...")

        print("\nDataset Info:")
        print(self.df.info())

        print("\nDataset Description:")
        print(self.df.describe())

        unique_events = self.df['eventId'].nunique()
        unique_labels = self.df['label'].nunique()
        label_counts = self.df['label'].value_counts()

        print(f"\nNumber of Unique Events: {unique_events}")
        print(f"Number of Unique Labels: {unique_labels}")
        print("\nLabel Distribution:\n", label_counts)

        logging.info(f"Dataset contains {unique_events} unique events and {unique_labels} unique labels.")
        logging.info("Analysis complete.")
