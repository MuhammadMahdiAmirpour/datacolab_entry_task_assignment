import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


class CovarianceMatrixCalculator:
    """
    A class to calculate the covariance matrix of a DataFrame with string columns using TF-IDF.

    Attributes:
        df (pandas.DataFrame): The input DataFrame with string columns.

    Methods:
        calculate_covariance_matrix(): Calculates the covariance matrix based on the TF-IDF features.
    """

    def __init__(self, df):
        """
        Initializes the CovarianceMatrixCalculator object.

        Args:
            df (pandas.DataFrame): The input DataFrame with string columns.
        """
        self.df = df

    def calculate_covariance_matrix(self):
        """
        Calculates the covariance matrix using TF-IDF for the input DataFrame.

        Returns:
            numpy.ndarray: The covariance matrix based on the TF-IDF features.
        """
        # Convert the string columns to a 1D array
        X = self.df.astype(str).values.flatten()

        # Create the TF-IDF matrix
        tfidf = TfidfVectorizer(max_features=self.df.shape[1])
        X_tfidf = tfidf.fit_transform(X)

        # Calculate the covariance matrix
        cov_matrix = np.cov(X_tfidf.T.toarray())

        return cov_matrix


class CovarianceMatrixPlotter:
    """
    A class to plot the covariance matrix of a DataFrame.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.
        covariance_matrix (numpy.ndarray): The covariance matrix to be plotted.

    Methods:
        plot_covariance_matrix(): Plots the covariance matrix of the DataFrame.
    """

    def __init__(self, df, cov_mat):
        """
        Initializes the CovarianceMatrixPlotter object.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            cov_mat (numpy.ndarray): The covariance matrix to be plotted.
        """
        self.df = df
        self.covariance_matrix = cov_mat

    def plot_covariance_matrix(self):
        """
        Plots the covariance matrix of the DataFrame.

        The function creates a new figure with a size of 12x10 inches and uses the seaborn.heatmap()
        function to plot the covariance matrix as a heatmap. The column names from the DataFrame are
        used as the x and y axis labels.

        The function sets the title, x-axis label, and y-axis label for the plot, and then displays
        the plot using plt.show().
        """
        plt.figure(figsize=(12, 10))

        # Get the column names from the DataFrame
        feature_names = self.df.columns

        # Create the heatmap
        sns.heatmap(self.covariance_matrix, annot=True, cmap='YlOrRd', vmin=-1, vmax=1, xticklabels=feature_names,
                    yticklabels=feature_names)

        plt.title('Covariance Matrix')
        plt.xlabel('Features')
        plt.ylabel('Features')
        plt.show()


class BiVariateAnalyzer:
    """
    A class to perform bivariate analysis on a DataFrame.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.

    Methods:
        analyze_bivariate_relationships(): Calculates and plots the covariance matrix of the DataFrame.
    """

    def __init__(self, df):
        """
        Initializes the BiVariateAnalyzer object.

        Args:
            df (pandas.DataFrame): The input DataFrame.
        """
        self.df = df

    def analyze_bivariate_relationships(self):
        """
        Calculates and plots the covariance matrix of the DataFrame.

        This method uses the CovarianceMatrixCalculator and CovarianceMatrixPlotter classes
        to calculate the covariance matrix and plot it as a heatmap.
        """
        # Calculate the covariance matrix
        calculator = CovarianceMatrixCalculator(self.df)
        covariance_matrix = calculator.calculate_covariance_matrix()
        print(f"Covariance matrix shape: {covariance_matrix.shape}")

        # Plot the covariance matrix
        plotter = CovarianceMatrixPlotter(self.df, covariance_matrix)
        plotter.plot_covariance_matrix()


if __name__ == '__main__':
    # Load the dataset
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)

    # Perform bivariate analysis
    analyzer = BiVariateAnalyzer(data)
    analyzer.analyze_bivariate_relationships()
