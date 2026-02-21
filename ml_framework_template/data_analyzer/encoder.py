import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Encoder:
    """Class unifying common categorical encoding strategies."""

    def __init__(self):
        self.label_encoders = {}
        self.one_hot_encoder = None

    def perform_label_encoding(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Performs label encoding on specified columns, replacing categories with integers.
        """
        logging.info(f"Performing label encoding on columns: {columns}")
        df_encoded = df.copy()
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
            else:
                logging.warning(
                    f"Column '{col}' not found in DataFrame for label encoding."
                )
        return df_encoded

    def perform_one_hot_encoding(
        self, df: pd.DataFrame, columns: list, drop_first: bool = True
    ) -> pd.DataFrame:
        """
        Performs one-hot encoding using pandas get_dummies.
        """
        logging.info(
            f"Performing one-hot encoding on columns: {columns} (drop_first={drop_first})"
        )

        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            logging.warning(
                f"The following columns were not found for OHE: {missing_cols}"
            )
            columns = [c for c in columns if c in df.columns]

        if not columns:
            return df

        return pd.get_dummies(df, columns=columns, drop_first=drop_first)
