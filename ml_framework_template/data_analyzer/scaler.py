import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Scaler:
    """Class abstracting common numerical scaling strategies."""

    def __init__(self):
        self.scaler = None
        self.scaler_type = None

    def fit_transform(
        self, df: pd.DataFrame, columns: list, scaling_type: str = "standard"
    ) -> pd.DataFrame:
        """
        Fits and transforms numerical columns.

        Args:
           scaling_type (str): 'standard' for StandardScaler or 'minmax' for MinMaxScaler
        """
        logging.info(f"Performing {scaling_type} scaling on columns: {columns}")

        if scaling_type.lower() == "standard":
            self.scaler = StandardScaler()
            self.scaler_type = "standard"
        elif scaling_type.lower() == "minmax":
            self.scaler = MinMaxScaler()
            self.scaler_type = "minmax"
        else:
            logging.error(f"Unknown scaling type: {scaling_type}")
            raise ValueError(f"Unknown scaling type: {scaling_type}")

        df_scaled = df.copy()

        # Only scale columns that exist
        valid_cols = [c for c in columns if c in df_scaled.columns]
        if not valid_cols:
            logging.warning("No valid columns found to scale.")
            return df_scaled

        df_scaled[valid_cols] = self.scaler.fit_transform(df_scaled[valid_cols])
        return df_scaled

    def transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Transforms numerical columns assuming the scaler has already been fit.
        """
        if self.scaler is None:
            logging.error("Scaler has not been fitted yet. Call fit_transform first.")
            raise ValueError(
                "Scaler has not been fitted yet. Call fit_transform first."
            )

        df_scaled = df.copy()
        valid_cols = [c for c in columns if c in df_scaled.columns]
        if valid_cols:
            df_scaled[valid_cols] = self.scaler.transform(df_scaled[valid_cols])

        return df_scaled
