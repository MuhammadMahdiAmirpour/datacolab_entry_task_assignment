import matplotlib.pyplot as plt
import pandas as pd


class UniVariateAnalyzer:
    """
    A class to perform uni-variate analysis on a DataFrame.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.

    Methods:
        generate_summary_statistics(): Generates summary statistics for the DataFrame.
        plot_histograms(): Plots histograms for numerical variables.
        plot_bar_plots(): Plots bar plots for categorical variables.
        perform_analysis(): Performs the complete uni-variate analysis.
    """

    def __init__(self, input_df):
        """
        Initializes the Uni variateSummary object.

        Args:
            input_df (pandas.DataFrame): The input DataFrame.
        """
        self.df = input_df

    def generate_summary_statistics(self):
        """
        Generates summary statistics for the DataFrame.

        Prints the summary statistics for the entire DataFrame.
        """
        summary_stats = self.df.describe()
        print("Summary Statistics:")
        print(summary_stats)

    def plot_histograms(self):
        """
        Plots histograms for numerical variables in the DataFrame.

        The function creates a histogram for each numerical variable and displays the plot.
        """
        for column in self.df.select_dtypes(include=[int, float]).columns:
            self.df[column].plot.hist(bins=10)
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.title(f"Histogram of {column}")
            plt.show()

    def plot_bar_plots(self):
        """
        Plots bar plots for categorical variables in the DataFrame.

        The function creates a bar plot for the top 10 values of each categorical variable and displays the plot.
        """
        for column in self.df.select_dtypes(exclude=[int, float]).columns:
            top_values = self.df[column].value_counts().head(10)
            top_values.plot.bar()
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.title(f"Bar Plot of {column}")
            plt.xticks(rotation=90)
            plt.show()

    def perform_analysis(self):
        """
        Performs the complete univariate analysis.

        The function calls the other methods to generate summary statistics and create visualizations.
        """
        self.generate_summary_statistics()
        self.plot_histograms()
        self.plot_bar_plots()


if __name__ == "__main__":
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    df = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                     names=column_names)

    # Create an instance of the Uni variateSummary class and perform the analysis
    uni_variate_analyzer = UniVariateAnalyzer(df)
    uni_variate_analyzer.perform_analysis()
