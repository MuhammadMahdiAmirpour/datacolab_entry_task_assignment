import json

import numpy as np
import pandas as pd

from src.data_cleaning.data_utils import unify_date_format_year


def fill_date_column_with_default_values(input_df, date_column="date"):
    # Identify year-only, year-month, and mm/dd/YYYY records with non-missing values
    year_only_mask = input_df[date_column].str.match(r'^\d{4}$') & input_df[date_column].notna()
    year_month_mask = input_df[date_column].str.match(r'^\d{4}-\d{2}$') & input_df[date_column].notna()
    mm_dd_yyyy_mask = input_df[date_column].str.match(r'^\d{2}/\d{2}/\d{4}$') & input_df[date_column].notna()

    # Create new column for conversion, preserving original values
    input_df['converted_date'] = input_df[date_column]

    # Apply conversion to year-only, year-month, and mm/dd/YYYY records in the new column
    input_df.loc[year_only_mask, 'converted_date'] = pd.to_datetime(
        input_df.loc[year_only_mask, 'converted_date'] + '-01-01',
        format='%Y-%m-%d', errors='coerce'
    ).dt.strftime('%Y-%m-%d')  # Format as YYYY-mm-dd

    input_df.loc[year_month_mask, 'converted_date'] = pd.to_datetime(
        input_df.loc[year_month_mask, 'converted_date'] + '-01',
        format='%Y-%m-%d', errors='coerce'
    ).dt.strftime('%Y-%m-%d')  # Format as YYYY-mm-dd

    input_df.loc[mm_dd_yyyy_mask, 'converted_date'] = pd.to_datetime(
        input_df.loc[mm_dd_yyyy_mask, 'converted_date'],
        format='%m/%d/%Y', errors='coerce'
    ).dt.strftime('%Y-%m-%d')  # Format as YYYY-mm-dd

    # Replace the original column with the converted dates
    input_df[date_column] = input_df['converted_date']

    # Drop the temporary column
    input_df.drop(columns=['converted_date'], inplace=True)

    return input_df


def guess_author(df):
    """
    Guess the author's name based on the genre and year of publishing.

    Parameters:
    df (pd.DataFrame): The dataframe containing the book data.

    Returns:
    None
    """
    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Get the freebase_id_json column value
        freebase_id_json = row['freebase_id_json']

        # If the freebase_id_json value is not missing and the author_name is missing
        if not pd.isna(freebase_id_json) and pd.isna(row['author_name']):
            # Parse the JSON object
            freebase_id_json = json.loads(freebase_id_json)

            # Get the most frequent genre label from the JSON object
            most_frequent_genre = max(set(freebase_id_json.values()), key=list(freebase_id_json.values()).count)

            # Get the year of publishing
            year = row['date']

            # If the year is not missing
            if not pd.isna(year):
                # Filter the dataframe to get the authors who have written books in the same genre and year
                filtered_df = df[
                    (df['freebase_id_json'].apply(lambda x: most_frequent_genre in json.loads(x).values())) & (
                            df['date'] == year)]

                # If there are any matching authors
                if not filtered_df.empty:
                    # Get the most frequent author name
                    guessed_author = filtered_df['author_name'].mode()[0]

                    # Set the guessed author's name
                    df.at[index, 'author_name'] = guessed_author
                else:
                    # Set the guessed author's name to "Unknown"
                    df.at[index, 'author_name'] = "Unknown"
            else:
                # Set the guessed author's name to "Unknown"
                df.at[index, 'author_name'] = "Unknown"
        else:
            # Do nothing if the freebase_id_json value is missing or the author_name is not missing
            pass


def guess_publish_year(input_df):
    # Drop rows where both 'date' and 'author_name' are missing
    input_df.dropna(subset=['date', 'author_name'], how='all', inplace=True)

    # Group by author_name and calculate the mean publish year for each author
    author_stats = input_df.groupby('author_name')['date'].agg(
        lambda x: np.mean(pd.to_datetime(x, format='%Y-%m-%d', errors='coerce').dt.year)
    ).reset_index()

    # Function to sample from normal distribution for each author and return integer year
    def sample_publish_year(mean_year):
        if pd.isna(mean_year):  # Check if mean_year is NaN
            return np.nan  # Return NaN if NaN input
        else:
            return int(np.round(np.random.normal(mean_year, 5)))  # Assuming std deviation of 5 and rounding

    # Apply sampling to books with missing date values
    missing_date_mask = input_df['date'].isnull()
    input_df.loc[missing_date_mask, 'date'] = input_df.loc[missing_date_mask, 'author_name'].map(
        author_stats.set_index('author_name')['date']).apply(sample_publish_year)

    return input_df


def guess_author_by_genre(input_df):
    # Parse the JSON strings in 'freebase_id_json' to extract genre information
    input_df['genre'] = input_df['freebase_id_json'].apply(
        lambda x: list(eval(x).values()) if isinstance(x, str) else [])

    # Group by genre and calculate the mean author name for each genre
    genre_stats = input_df.explode('genre').groupby('genre')['author_name'].agg(
        lambda x: x.dropna().values.tolist() if not x.dropna().empty else []
    ).reset_index()

    # Define a custom sampling function that handles empty lists
    def sample_author(authors):
        if len(authors) > 0:  # Check if the list is not empty
            return np.random.choice(authors)
        else:
            return np.nan

    # Apply sampling to books with missing author names based on genre statistics
    missing_author_mask = input_df['author_name'].isnull()
    input_df.loc[missing_author_mask, 'author_name'] = input_df.loc[missing_author_mask, 'genre'].apply(
        lambda genres: sample_author(genre_stats[genre_stats['genre'].isin(genres)]['author_name'].explode().dropna())
    )

    # Drop the temporary 'genre' column
    input_df.drop(columns=['genre'], inplace=True)

    return input_df


def handle_missing_values(input_df):
    unify_date_format_year(input_df)
    guess_publish_year(input_df)


if __name__ == '__main__':
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)
