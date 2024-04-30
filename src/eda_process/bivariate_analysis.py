import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_cov_matrix(df):
    """
    Calculates the covariance matrix using TF-IDF for a DataFrame with string columns.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with string columns.

    Returns:
    numpy.ndarray: The covariance matrix based on the TF-IDF features.
    """
    # Convert the string columns to a 1D array
    X = df.astype(str).values.flatten()

    # Create the TF-IDF matrix
    tfidf = TfidfVectorizer(max_features=df.shape[1])
    X_tfidf = tfidf.fit_transform(X)

    # Calculate the covariance matrix
    cov_matrix = np.cov(X_tfidf.T.toarray())

    return cov_matrix


def plot_covariance_matrix(df, covariance_matrix):
    """
    Plots the covariance matrix of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    covariance_matrix (numpy.ndarray): The covariance matrix to be plotted.
    """
    plt.figure(figsize=(12, 10))

    # Get the column names from the DataFrame
    feature_names = df.columns

    # Create the heatmap
    sns.heatmap(covariance_matrix, annot=True, cmap='YlOrRd', vmin=-1, vmax=1, xticklabels=feature_names,
                yticklabels=feature_names)

    plt.title('Covariance Matrix')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.show()


if __name__ == '__main__':
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)
    covariance_matrix = get_tfidf_cov_matrix(data)
    print(covariance_matrix.shape)
    plot_covariance_matrix(data, covariance_matrix)
