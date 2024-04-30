import json
import re
import string

import pandas as pd


class DataUtilityProvider:
    """
    A class to clean and preprocess a DataFrame.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.

    Methods:
        parse_json_column(column_name): Parses the JSON data in the specified column.
        transform_date_formats(): Transforms the date formats in the DataFrame.
        clean_summary(summary): Cleans the summary text.
        clean_summary_df(): Cleans the summary column in the DataFrame.
        unify_date_format_year(): Unifies the date format to year.
    """

    def __init__(self, df):
        """
        Initializes the DataCleaner object.

        Args:
            df (pandas.DataFrame): The input DataFrame.
        """
        self.df = df

    def parse_json_column(self, column_name):
        """
        Parses the JSON data in the specified column.

        Args:
            column_name (str): The name of the column to parse.

        Returns:
            pandas.DataFrame: The DataFrame with the parsed JSON data.
        """

        def parse_json(cell):
            try:
                return json.loads(cell) if pd.notnull(cell) and isinstance(cell, str) else cell
            except json.JSONDecodeError:
                return cell

        self.df[column_name] = self.df[column_name].apply(parse_json)
        return self.df

    def transform_date_formats(self):
        """
        Transforms the date formats in the DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with the transformed date formats.
        """
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df['date'] = self.df['date'].dt.strftime('%Y-%m-%d')
        self.df.loc[self.df['date'].str.len() == 7, 'date'] = pd.to_datetime(
            self.df.loc[self.df['date'].str.len() == 7, 'date'],
            format='%Y-%m').dt.strftime('%Y-%m')
        return self.df

    @staticmethod
    def clean_summary(summary):
        """
        Cleans the summary text.

        Args:
            summary (str): The input summary text.

        Returns:
            str: The cleaned summary text.
        """
        table = str.maketrans(dict.fromkeys(string.punctuation))
        text = summary.translate(table)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return text

    def clean_summary_df(self):
        """
        Cleans the summary column in the DataFrame.
        """
        # Calculate the frequency table
        freq_table = self.df["summary"].value_counts()
        # Get the top 10 most frequent summaries
        top_10_summaries = freq_table.head(10).index
        # Replace the top 10 most frequent summaries with an empty string
        self.df.loc[self.df["summary"].isin(top_10_summaries), "summary"] = ""
        self.df["summary"] = self.df["summary"].map(self.clean_summary)

    def unify_date_format_year(self):
        """
        Unifies the date format to year.

        Returns:
            pandas.DataFrame: The DataFrame with the unified date format.
        """
        years = [date[:4] if isinstance(date, str) and len(date) >= 4 else '' for date in self.df['date']]
        self.df['date'] = years
        self.df['date'] = pd.to_numeric(self.df['date'], errors='coerce')
        self.df['date'] = self.df['date'].astype(int, errors='ignore')
        return self.df


if __name__ == '__main__':
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)

    # Create an instance of the DataCleaner class and clean the data
    data_cleaner = DataUtilityProvider(data)
    data_cleaner.parse_json_column("freebase_id_json")
    data_cleaner.transform_date_formats()
    data_cleaner.clean_summary_df()
    data_cleaner.unify_date_format_year()

    # Access the cleaned DataFrame
    cleaned_data = data_cleaner.df
    print(cleaned_data.head())
