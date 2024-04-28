import matplotlib.pyplot as plt
import pandas as pd


# Assuming you have the dataset stored in a pandas DataFrame called 'df'

def perform_univariate_analysis(dataframe):
    # Calculate summary statistics
    summary_stats = dataframe.describe()
    print(summary_stats)

    # Create a frequency table for each variable
    for column in dataframe.columns:
        freq_table = dataframe[column].value_counts()
        print(f"Frequency table for {column}:")
        print(freq_table.head(20))  # Print only the top 10 values
        print("...")  # Print ellipsis to indicate there are more values

        # Plot a histogram for numerical variables
        if dataframe[column].dtype in [int, float]:
            dataframe[column].plot.hist(bins=10)
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.title(f"Histogram of {column}")
            plt.show()

        # Plot a bar plot for categorical variables
        else:
            dataframe[column].value_counts().head(10).plot.bar()
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.title(f"Bar Plot of {column}")
            plt.xticks(rotation=90)
            plt.show()


if __name__ == "__main__":
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    df = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                     names=column_names)

    # Call the function to perform Univariate Analysis
    perform_univariate_analysis(df)
