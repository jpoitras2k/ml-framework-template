import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Regressor:
    """Class unifying common regression algorithms."""

    def __init__(self, model_type: str = "rf", **kwargs):
        """
        Initializes the regressor.

        Args:
           model_type (str): The type of model to initialize. Options are:
              'lr': Linear Regression
              'ridge': Ridge Regression
              'lasso': Lasso Regression
              'rf': Random Forest Regressor
              'svr': Support Vector Regressor
           **kwargs: Additional hyperparameters.
        """
        self.model_type = model_type.lower()
        self.model = self._initialize_model(self.model_type, **kwargs)
        self.is_fitted = False

    def _initialize_model(self, model_type: str, **kwargs):
        logging.info(f"Initializing {model_type.upper()} regressor.")
        if model_type == "lr":
            return LinearRegression(**kwargs)
        elif model_type == "ridge":
            return Ridge(**kwargs)
        elif model_type == "lasso":
            return Lasso(**kwargs)
        elif model_type == "rf":
            return RandomForestRegressor(**kwargs)
        elif model_type == "svr":
            return SVR(**kwargs)
        else:
            logging.error(
                f"Unsupported model type: {model_type}. Falling back to Random Forest."
            )
            return RandomForestRegressor(**kwargs)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Trains the model."""
        if self.model is None:
            raise ValueError("Model has not been initialized.")

        logging.info(
            f"Training {self.model_type.upper()} regressor on {X_train.shape[0]} samples."
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logging.info("Training complete.")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Predicts data."""
        if not self.is_fitted:
            logging.error("Model must be trained before calling predict.")
            raise ValueError("Model must be trained before calling predict.")

        logging.info(f"Predicting on {X_test.shape[0]} samples.")
        return self.model.predict(X_test)

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series):
        """Evaluates the predictions against true values."""
        logging.info("Evaluating regression predictions.")

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
