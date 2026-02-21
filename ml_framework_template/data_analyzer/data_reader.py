import pandas as pd
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataReader:
    """Class to handle reading tabular data from various formats."""

    @staticmethod
    def read_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """
        Reads a CSV file into a Pandas DataFrame.

        Args:
            file_path (str): The path to the CSV file.
            **kwargs: Additional keyword arguments to pass to pd.read_csv.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed.
        """
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            df = pd.read_csv(file_path, **kwargs)
            logging.info(f"Successfully loaded CSV from {file_path}. Shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error reading CSV {file_path}: {e}")
            raise ValueError(f"Error reading CSV: {e}")

    @staticmethod
    def read_excel(file_path: str, **kwargs) -> pd.DataFrame:
        """
        Reads an Excel file into a Pandas DataFrame.
        """
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            df = pd.read_excel(file_path, **kwargs)
            logging.info(
                f"Successfully loaded Excel from {file_path}. Shape: {df.shape}"
            )
            return df
        except Exception as e:
            logging.error(f"Error reading Excel {file_path}: {e}")
            raise ValueError(f"Error reading Excel: {e}")

    @staticmethod
    def split_features_target(df: pd.DataFrame, target_column: str):
        """
        Splits a DataFrame into features (X) and target (y).

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the target column.

        Returns:
            tuple: (X, y)
        """
        if target_column not in df.columns:
            logging.error(f"Target column '{target_column}' not found in DataFrame.")
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        logging.info(f"Split data into X (shape: {X.shape}) and y (shape: {y.shape})")
        return X, y
