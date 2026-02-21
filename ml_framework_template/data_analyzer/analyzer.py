import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Analyzer:
    """Class to handle Exploratory Data Analysis (EDA)."""

    @staticmethod
    def dataset_summary(df: pd.DataFrame):
        """
        Prints and returns the statistical summary of the DataFrame.
        """
        logging.info("Generating dataset summary.")
        print(f"Dataset Shape: {df.shape}")
        print("\nDataset Info:")
        df.info()
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nStatistical Description:")
        description = df.describe(include="all")
        print(description)
        return description

    @staticmethod
    def get_column_correlations(
        df: pd.DataFrame, method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Calculates and returns the correlation matrix for numerical columns.
        """
        # Select only numerical features to prevent correlation errors
        numeric_df = df.select_dtypes(include=["float64", "int64"])
        logging.info(
            f"Calculating {method} correlation matrix for {numeric_df.shape[1]} numerical columns."
        )
        return numeric_df.corr(method=method)


