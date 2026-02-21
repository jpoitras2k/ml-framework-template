import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class Visualizer:
    """Class to handle data visualizations."""

    @staticmethod
    def plot_correlation_matrix(
        df_corr: pd.DataFrame, save_path: str = None
    ):
        """
        Plots a heatmap for a given correlation matrix.
        """
        logging.info("Plotting correlation matrix.")

        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix heat map")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Correlation matrix saved to {save_path}")
        else:
            plt.show()

    @staticmethod
    def plot_histogram(
        df: pd.DataFrame, column: str, bins: int = 30, save_path: str = None
    ):
        """
        Plots a histogram for a specified numerical column.
        """
        logging.info(f"Plotting histogram for {column}.")
        
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], bins=bins, kde=True, color="skyblue")
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Histogram saved to {save_path}")
        else:
            plt.show()
            
    @staticmethod
    def plot_scatter(
        df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None, save_path: str = None
    ):
        """
        Plots a scatter plot between two numerical columns.
        """
        logging.info(f"Plotting scatter plot: {x_col} vs {y_col}.")
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette="deep")
        plt.title(f"Scatter Plot: {x_col} vs {y_col}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Scatter plot saved to {save_path}")
        else:
            plt.show()
            
    @staticmethod
    def plot_boxplot(
        df: pd.DataFrame, column: str, by_col: str = None, save_path: str = None
    ):
        """
        Plots a boxplot to identify outliers in a numerical column.
        """
        logging.info(f"Plotting boxplot for {column}.")
        
        plt.figure(figsize=(8, 6))
        if by_col:
            sns.boxplot(data=df, x=by_col, y=column, palette="Set2")
            plt.title(f"Boxplot of {column} grouped by {by_col}")
        else:
            sns.boxplot(data=df, y=column, color="lightgreen")
            plt.title(f"Boxplot of {column}")
            
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Boxplot saved to {save_path}")
        else:
            plt.show()
