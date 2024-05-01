import pandas as pd

from src.data_cleaning.data_utility_provider import DataUtilityProvider


class DataCleaner:
    """
    A class to clean and preprocess a DataFrame.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.
        data_utility_provider (DataUtilityProvider): The data utility provider object.

    Methods:
        clean_data(): Performs the complete data cleaning process.
    """

    def __init__(self, df):
        """
        Initializes the DataCleaner object.

        Args:
            df (pandas.DataFrame): The input DataFrame.
        """
        self.df = df
        self.data_utility_provider = DataUtilityProvider(df)

    def clean_data(self):
        """
        Cleans and preprocesses the input DataFrame.

        Returns:
            pandas.DataFrame: The cleaned and preprocessed DataFrame.
        """
        # Handle missing values (replace with empty string)
        self.df.fillna("", inplace=True)

        # Drop duplicate rows
        self.df.drop_duplicates(keep='first', inplace=True)

        # Transform date formats
        # self.data_utility_provider.transform_date_formats()

        # Parse JSON column
        self.data_utility_provider.parse_json_column(column_name="freebase_id_json")

        # Clean summary column
        self.data_utility_provider.clean_summary_df()

        # Handle missing values (drop rows with missing 'date' or 'freebase_id_json')
        self.df = self.df.dropna(subset=['date', 'freebase_id_json'])

        return self.df


def main():
    # Load data from text file into DataFrame
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)

    # Create an instance of the DataCleaner class and clean the data
    data_cleaner = DataCleaner(data)
    data_cleaned = data_cleaner.clean_data()

    # Print a sample of the cleaned 'date' column
    print(data_cleaned["date"][50:60])


if __name__ == '__main__':
    main()
