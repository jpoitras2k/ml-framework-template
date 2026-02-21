import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Classifier:
    """Class unifying common classification algorithms."""

    def __init__(self, model_type: str = "rf", **kwargs):
        """
        Initializes the classifier.

        Args:
           model_type (str): The type of model to initialize. Options are:
              'lr': Logistic Regression
              'rf': Random Forest
              'svc': Support Vector Classifier
              'knn': K-Nearest Neighbors
              'dt': Decision Tree
           **kwargs: Additional hyperparameters to pass into the model.
        """
        self.model_type = model_type.lower()
        self.model = self._initialize_model(self.model_type, **kwargs)
        self.is_fitted = False

    def _initialize_model(self, model_type: str, **kwargs):
        logging.info(f"Initializing {model_type.upper()} classifier.")
        if model_type == "lr":
            return LogisticRegression(**kwargs)
        elif model_type == "rf":
            return RandomForestClassifier(**kwargs)
        elif model_type == "svc":
            return SVC(**kwargs)
        elif model_type == "knn":
            return KNeighborsClassifier(**kwargs)
        elif model_type == "dt":
            return DecisionTreeClassifier(**kwargs)
        else:
            logging.error(
                f"Unsupported model type: {model_type}. Falling back to Random Forest."
            )
            return RandomForestClassifier(**kwargs)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Trains the model."""
        if self.model is None:
            raise ValueError("Model has not been initialized.")

        logging.info(
            f"Training {self.model_type.upper()} model on {X_train.shape[0]} samples."
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
        logging.info("Evaluating predictions.")

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print(f"Accuracy: {acc:.4f}\n")
        print("Classification Report:\n", report)
        return acc, report, cm

    def plot_confusion_matrix(
        self, y_true: pd.Series, y_pred: pd.Series, save_path: str = None
    ):
        """Plots the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix ({self.model_type.upper()})")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
