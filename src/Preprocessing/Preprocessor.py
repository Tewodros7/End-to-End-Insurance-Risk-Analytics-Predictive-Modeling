"""
DatasetLoader Class

This module defines a DatasetLoader class designed for importing and conducting
basic inspection and quality checks on a dataset.
"""

import pandas as pd

class DatasetLoader:
    def __init__(self, path):
        self.path = path
        self.data = None

    def import_dataset(self):
        """Reads the dataset from the given file path."""
        self.data = pd.read_csv(self.path, sep="|", engine="python")

    def explore_basic_info(self):
        """Display basic structure and contents of the dataset."""
        print("Dataset Dimensions:", self.data.shape)
        print("Column Names:", self.data.columns.tolist())
        print("Preview of Data:\n", self.data.head())

    def check_data_quality(self):
        """Identify missing values, data types, and remove duplicates."""
        print("\nNull Value Report:")
        print(self.data.isnull().sum())

        print("\nColumn Data Types:")
        print(self.data.dtypes)

        # Eliminate duplicate records, if present
        self.data.drop_duplicates(inplace=True)

    def summary_statistics(self):
        """Show summary statistics for selected numerical features."""
        selected_numeric_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
        print("\nStatistical Summary:")
        print(self.data[selected_numeric_cols].describe())
