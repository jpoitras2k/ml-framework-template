import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataPreprocessor:
    """Class containing utility functions for generic data preprocessing."""

    @staticmethod
    def perform_train_test_split(
        X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
    ):
        """
        Splits features and target into training and testing sets.
        """
        logging.info(
            f"Performing train-test split (test_size={test_size}, random_state={random_state})"
        )
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    @staticmethod
    def handle_missing_values(X: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Impute missing values using the specified strategy ('mean', 'median', 'most_frequent', 'constant').
        """
        logging.info(f"Handling missing values using strategy: '{strategy}'")

        # We need to distinguish between numerical and non-numerical columns depending on strategy
        try:
            imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)

            # SimpleImputer returns a numpy array, we need to convert back to DataFrame
            imputed_array = imputer.fit_transform(X)
            X_imputed = pd.DataFrame(imputed_array, columns=X.columns, index=X.index)
            logging.info("Successfully imputed missing values.")
            return X_imputed
        except Exception as e:
            logging.error(f"Error during missing value imputation: {e}")
            raise ValueError(f"Error handling missing values: {e}")

    @staticmethod
    def shuffle_data(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
        """
        Shuffles the rows of a DataFrame.
        """
        logging.info(f"Shuffling DataFrame with random_state={random_state}")
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
