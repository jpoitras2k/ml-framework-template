import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Clustering:
    """Class unifying common clustering algorithms."""

    def __init__(self, model_type: str = "kmeans", **kwargs):
        """
        Initializes the clustering model.

        Args:
           model_type (str): The type of model to initialize. Options are:
              'kmeans': K-Means Clustering
              'dbscan': DBSCAN Clustering
           **kwargs: Additional hyperparameters.
        """
        self.model_type = model_type.lower()
        self.model = self._initialize_model(self.model_type, **kwargs)
        self.labels = None

    def _initialize_model(self, model_type: str, **kwargs):
        logging.info(f"Initializing {model_type.upper()} clustering model.")
        if model_type == "kmeans":
            return KMeans(**kwargs)
        elif model_type == "dbscan":
            return DBSCAN(**kwargs)
        else:
            logging.error(
                f"Unsupported model type: {model_type}. Falling back to K-Means."
            )
            return KMeans(**kwargs)

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        """Fits the model and returns cluster labels."""
        if self.model is None:
            raise ValueError("Model has not been initialized.")

        logging.info(
            f"Fitting {self.model_type.upper()} model on {X.shape[0]} samples."
        )
        self.labels = self.model.fit_predict(X)
        logging.info("Clustering complete.")
        return self.labels

    def evaluate(self, X: pd.DataFrame):
        """Evaluates the clustering using silhouette score (if possible)."""
        if self.labels is None:
            logging.error("Model must be fitted before evaluating.")
            raise ValueError("Model must be fitted before evaluating.")

        # Silhouette score requires more than 1 cluster and less clusters than number of samples
        unique_labels = set(self.labels)
        if len(unique_labels) > 1 and len(unique_labels) < X.shape[0]:
            score = silhouette_score(X, self.labels)
            print(f"Silhouette Score: {score:.4f}")
            return score
        else:
            logging.warning(
                "Cannot calculate Silhouette Score. Model found 1 or zero labels."
            )
            return None

    def plot_clusters(
        self, X: pd.DataFrame, feature_x: str, feature_y: str, save_path: str = None
    ):
        """Plots a 2D scatterplot of the clusters."""
        if self.labels is None:
            logging.error("Model must be fitted before plotting.")
            raise ValueError("Model must be fitted before plotting.")

        if feature_x not in X.columns or feature_y not in X.columns:
            logging.error("Specified features for plotting not found in DataFrame.")
            raise ValueError("Specified features for plotting not found.")

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=X[feature_x], y=X[feature_y], hue=self.labels, palette="viridis", data=X
        )
        plt.title(f"{self.model_type.upper()} Clustering result")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Cluster plot saved to {save_path}")
        else:
            plt.show()
