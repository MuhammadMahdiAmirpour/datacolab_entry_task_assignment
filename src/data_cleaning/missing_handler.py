import json
import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    A class to preprocess the input DataFrame.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.

    Methods:
        preprocess_data(): Preprocesses the input data.
        fill_date_column_with_default_values(): Fills the date column with default values.
        guess_author(): Guesses the author's name based on the genre and year of publishing.
        guess_publish_year(): Guesses the publish year based on the author's average publish year.
        guess_author_by_genre(): Guesses the author's name based on the genre.
    """

    def __init__(self, df):
        """
        Initializes the DataPreprocessor object.

        Args:
            df (pandas.DataFrame): The input DataFrame.
        """
        self.df = df

    def preprocess_data(self):
        """
        Preprocesses the input data.

        Returns:
            pandas.DataFrame: The cleaned and preprocessed DataFrame.
        """
        # Handle missing values (replace with empty string)
        self.df.fillna("", inplace=True)


        # Fill the date column with default values, this line is optional, uncomment it if you want to use it
        # self.fill_date_column_with_default_values()

        # Guess the publish year based on the author's average publish year
        self.guess_publish_year()

        # Guess the author's name based on the genre and year of publishing, this line is optional, uncomment it if
        # you want to use it
        # self.guess_author()

        # Guess the author's name based on the genre, this line is optional, uncomment it if you want to use it
        # self.guess_author_by_genre()

        # Handle missing values (drop rows with missing 'date' or 'freebase_id_json')
        self.df = self.df.dropna(subset=['date', 'freebase_id_json'])

        return self.df

    def fill_date_column_with_default_values(self):
        """
        Fills the date column with default values.
        """
        # Identify year-only, year-month, and mm/dd/YYYY records with non-missing values
        year_only_mask = self.df['date'].str.match(r'^\d{4}$') & self.df['date'].notna()
        year_month_mask = self.df['date'].str.match(r'^\d{4}-\d{2}$') & self.df['date'].notna()
        mm_dd_yyyy_mask = self.df['date'].str.match(r'^\d{2}/\d{2}/\d{4}$') & self.df['date'].notna()

        # Create new column for conversion, preserving original values
        self.df['converted_date'] = self.df['date']

        # Apply conversion to year-only, year-month, and mm/dd/YYYY records in the new column
        self.df.loc[year_only_mask, 'converted_date'] = pd.to_datetime(
            self.df.loc[year_only_mask, 'converted_date'] + '-01-01',
            format='%Y-%m-%d', errors='coerce'
        ).dt.strftime('%Y-%m-%d')  # Format as YYYY-mm-dd

        self.df.loc[year_month_mask, 'converted_date'] = pd.to_datetime(
            self.df.loc[year_month_mask, 'converted_date'] + '-01',
            format='%Y-%m-%d', errors='coerce'
        ).dt.strftime('%Y-%m-%d')  # Format as YYYY-mm-dd

        self.df.loc[mm_dd_yyyy_mask, 'converted_date'] = pd.to_datetime(
            self.df.loc[mm_dd_yyyy_mask, 'converted_date'],
            format='%m/%d/%Y', errors='coerce'
        ).dt.strftime('%Y-%m-%d')  # Format as YYYY-mm-dd

        # Replace the original column with the converted dates
        self.df['date'] = self.df['converted_date']

        # Drop the temporary column
        self.df.drop(columns=['converted_date'], inplace=True)

    def guess_author(self):
        """
        Guesses the author's name based on the genre and year of publishing.
        """
        # Iterate over each row in the dataframe
        for index, row in self.df.iterrows():
            # Get the freebase_id_json column value
            freebase_id_json = row['freebase_id_json']

            # If the freebase_id_json value is not missing and the author_name is missing
            if not pd.isna(freebase_id_json) and pd.isna(row['author_name']):
                # Check the data type of freebase_id_json
                if isinstance(freebase_id_json, str):
                    # Parse the JSON object
                    freebase_id_json = json.loads(freebase_id_json)

                    # Get the most frequent genre label from the JSON object
                    most_frequent_genre = max(set(freebase_id_json.values()), key=list(freebase_id_json.values()).count)

                    # Get the year of publishing
                    year = row['date']

                    # If the year is not missing
                    if not pd.isna(year):
                        # Filter the dataframe to get the authors who have written books in the same genre and year
                        filtered_df = self.df[
                            (self.df['freebase_id_json'].apply(
                                lambda x: isinstance(x, str) and most_frequent_genre in json.loads(x).values())) & (
                                    self.df['date'] == year)]

                        # If there are any matching authors
                        if not filtered_df.empty:
                            # Get the most frequent author name
                            if not filtered_df['author_name'].empty:
                                guessed_author = filtered_df['author_name'].mode()[0]
                            else:
                                guessed_author = "Unknown"

                            # Set the guessed author's name
                            self.df.at[index, 'author_name'] = guessed_author
                        else:
                            # Set the guessed author's name to "Unknown"
                            self.df.at[index, 'author_name'] = "Unknown"
                    else:
                        # Set the guessed author's name to "Unknown"
                        self.df.at[index, 'author_name'] = "Unknown"
                else:
                    # Set the guessed author's name to "Unknown"
                    self.df.at[index, 'author_name'] = "Unknown"
            else:
                # Do nothing if the freebase_id_json value is missing or the author_name is not missing
                pass

    def guess_publish_year(self):
        """
        Guesses the publish year based on the author's average publish year.
        """
        # Drop rows where both 'date' and 'author_name' are missing
        self.df.dropna(subset=['date', 'author_name'], how='all', inplace=True)

        # Group by author_name and calculate the mean publish year for each author
        author_stats = self.df.groupby('author_name')['date'].agg(
            lambda x: np.mean(pd.to_datetime(x, format='%Y-%m-%d', errors='coerce').dt.year)
        ).reset_index()

        # Function to sample from normal distribution for each author and return integer year
        def sample_publish_year(mean_year):
            if pd.isna(mean_year):  # Check if mean_year is NaN
                return np.nan  # Return NaN if NaN input
            else:
                return int(np.round(np.random.normal(mean_year, 5)))  # Assuming std deviation of 5 and rounding

        # Apply sampling to books with missing date values
        missing_date_mask = self.df['date'].isnull()
        self.df.loc[missing_date_mask, 'date'] = self.df.loc[missing_date_mask, 'author_name'].map(
            author_stats.set_index('author_name')['date']).apply(sample_publish_year)

    def guess_author_by_genre(self):
        """
        Guesses the author's name based on the genre.

        Returns:
            pandas.DataFrame: The DataFrame with the missing author names filled.
        """
        # Parse the JSON strings in 'freebase_id_json' to extract genre information
        self.df['genre'] = self.df['freebase_id_json'].apply(
            lambda x: list(json.loads(x).values()) if isinstance(x, str) and x.strip() else [])

        # Group by genre and calculate the mean author name for each genre
        genre_stats = self.df.explode('genre').groupby('genre')['author_name'].agg(
            lambda x: x.dropna().values.tolist() if not x.dropna().empty else []
        ).reset_index()

        # Define a custom sampling function that handles empty lists
        def sample_author(authors):
            if len(authors) > 0:  # Check if the list is not empty
                return np.random.choice(authors)
            else:
                return np.nan

        # Apply sampling to books with missing author names based on genre statistics
        missing_author_mask = self.df['author_name'].isnull()
        self.df.loc[missing_author_mask, 'author_name'] = self.df.loc[missing_author_mask, 'genre'].apply(
            lambda genres: sample_author(
                genre_stats[genre_stats['genre'].isin(genres)]['author_name'].explode().dropna())
        )

        # Drop the temporary 'genre' column
        self.df.drop(columns=['genre'], inplace=True)

        return self.df


class MissingHandler:
    """
    A class to handle missing values and preprocess a DataFrame.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.

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
        self.data_preprocessor = DataPreprocessor(df)

    def handle_missing(self):
        """
        Cleans and preprocesses the input DataFrame.

        Returns:
            pandas.DataFrame: The cleaned and preprocessed DataFrame.
        """
        return self.data_preprocessor.preprocess_data()


def main():
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)

    # Create an instance of the DataCleaner class and clean the data
    missing_handler = MissingHandler(data)
    cleaned_data = missing_handler.handle_missing()

    # Access the cleaned DataFrame
    print(cleaned_data.head())


if __name__ == '__main__':
    main()
