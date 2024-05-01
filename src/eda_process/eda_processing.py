import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter


class DataExplorer:
    """
    A class to perform comprehensive exploratory data analysis on a DataFrame.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.

    Methods:
        initial_exploration(): Performs initial exploration of the DataFrame.
        check_for_missing_values(): Checks for missing values in the DataFrame.
        get_sample_of_each_column(): Prints a sample value for each column in the DataFrame.
        explore_numerical_columns(): Analyzes the numerical columns in the DataFrame.
        explore_categorical_columns(): Analyzes the categorical columns in the DataFrame.
        explore_text_columns(): Analyzes the text columns in the DataFrame.
        explore_date_column(): Analyzes the date column in the DataFrame.
        explore_genre_column(): Analyzes the genre column in the DataFrame.
        check_loss_for_missing(column_name): Calculates the percentage of data loss if a column is dropped.
    """

    def __init__(self, input_df):
        """
        Initializes the DataExplorer object.

        Args:
            input_df (pandas.DataFrame): The input DataFrame.
        """
        self.df = input_df

    def initial_exploration(self):
        """
        Performs initial exploration of the DataFrame.

        Prints the head, info, and tail of the DataFrame.
        """
        print("Head:")
        print(self.df.head())
        print("\nInfo:")
        print(self.df.info())
        print("\nTail:")
        print(self.df.tail())

    def check_for_missing_values(self):
        """
        Checks for missing values in the DataFrame.

        Prints the sum of missing values for each column.
        """
        print("Missing values:")
        print(self.df.isnull().sum())

    def get_sample_of_each_column(self):
        """
        Prints a sample value for each column in the DataFrame.
        """
        for col in self.df.columns:
            print(f"Column: {col}")
            print(self.df[col][0])

    def explore_numerical_columns(self):
        """
        Analyzes the numerical columns in the DataFrame.

        Plots histograms and box plots for each numerical column.
        """
        num_cols = self.df.select_dtypes(include=[int, float]).columns
        for col in num_cols:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=self.df, x=col)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()

            plt.figure(figsize=(8, 6))
            sns.boxplot(data=self.df, x=col)
            plt.title(f'Box Plot of {col}')
            plt.xlabel(col)
            plt.ylabel('Value')
            plt.show()

    def explore_categorical_columns(self):
        """
        Analyzes the categorical columns in the DataFrame.

        Plots count plots for each categorical column.
        """
        cat_cols = self.df.select_dtypes(exclude=[int, float]).columns
        for col in cat_cols:
            plt.figure(figsize=(8, 6))
            sns.countplot(data=self.df, x=col)
            plt.title(f'Count Plot of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            plt.show()

    def explore_text_columns(self):
        """
        Analyzes the text columns in the DataFrame.

        Prints the most common words in each text column.
        """
        text_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in text_cols:
            # Handle NaN values by converting them to an empty string
            words = ' '.join(str(x) for x in self.df[col] if not pd.isnull(x)).split()
            word_counts = Counter(words)
            print(f"Most common words in {col}: ")
            print(word_counts.most_common(10))

    def explore_date_column(self):
        """
        Analyzes the date column in the DataFrame.

        Plots the number of books published per year.
        """
        self.df['date'] = pd.to_datetime(self.df['date'])
        plt.figure(figsize=(10, 6))
        self.df['date'].dt.year.value_counts().sort_index().plot(kind='bar')
        plt.title('Number of Books Published per Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.show()

    def explore_genre_column(self):
        """
        Analyzes the genre column in the DataFrame.

        Plots the top 10 most common genres.
        """
        genres = [genre for genres in self.df['freebase_id_json'].apply(self.extract_genre) for genre in genres]
        genre_counts = Counter(genres)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(genre_counts.most_common(10))[0], y=list(genre_counts.most_common(10))[1])
        plt.title('Top 10 Most Common Genres')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.show()

    @staticmethod
    def extract_genre(json_dict):
        """
        Extracts the genres from a JSON-like dictionary.

        Args:
            json_dict (dict): The input JSON-like dictionary.

        Returns:
            list: A list of genres extracted from the dictionary.
        """
        return list(json_dict.values())

    def check_loss_for_missing(self, column_name):
        """
        Calculates the percentage of data loss if a column is dropped.

        Args:
            column_name (str): The name of the column to check.

        Returns:
            float: The percentage of data loss if the column is dropped.
        """
        df1 = self.df.copy(deep=True)
        s1 = df1.size
        df1.drop([column_name], axis=1, inplace=True)
        s2 = df1.size
        return s2 / s1

    def explore_data(self):
        self.initial_exploration()
        self.check_for_missing_values()
        self.get_sample_of_each_column()
        self.explore_numerical_columns()
        # self.explore_categorical_columns()
        self.explore_text_columns()
        # self.explore_date_column()
        # self.explore_genre_column()


if __name__ == '__main__':
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    df = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                     names=column_names)

    # Create an instance of the DataExplorer class and perform the analysis
    data_explorer = DataExplorer(df)
    data_explorer.explore_data()

    # Check the data loss if a column is dropped
    print(
        f"Data loss if 'freebase_id_json' column is dropped: {data_explorer.check_loss_for_missing('freebase_id_json'):.2%}")
