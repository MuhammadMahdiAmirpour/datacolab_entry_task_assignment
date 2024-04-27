import numpy as np
import pandas as pd

from data_utils import (clean_summary_df)


def clean_data(input_df):
    # handling missing values (replace with empty string or custom value)
    input_df.fillna("")
    input_df.drop_duplicates(keep='first', inplace=True)
    # unify_date_format_year(input_df)
    clean_date_column(input_df, 'date')
    clean_summary_df(input_df)
    return input_df


def guess_publish_year(df):
    # Drop rows where both 'date' and 'author_name' are missing
    df.dropna(subset=['date', 'author_name'], how='all', inplace=True)

    # Group by author_name and calculate the mean publish year for each author
    author_stats = df.groupby('author_name')['date'].agg(
        lambda x: np.mean(pd.to_datetime(x, format='%Y-%m-%d', errors='coerce').dt.year)
    ).reset_index()

    # Function to sample from normal distribution for each author and return integer year
    def sample_publish_year(mean_year):
        if pd.isna(mean_year):  # Check if mean_year is NaN
            return np.nan  # Return NaN if NaN input
        else:
            return int(np.round(np.random.normal(mean_year, 5)))  # Assuming std deviation of 5 and rounding

    # Apply sampling to books with missing date values
    missing_date_mask = df['date'].isnull()
    df.loc[missing_date_mask, 'date'] = df.loc[missing_date_mask, 'author_name'].map(
        author_stats.set_index('author_name')['date']).apply(sample_publish_year)

    return df


def guess_author_by_genre(df):
    # Parse the JSON strings in 'freebase_id_json' to extract genre information
    df['genre'] = df['freebase_id_json'].apply(lambda x: list(eval(x).values()) if isinstance(x, str) else [])

    # Group by genre and calculate the mean author name for each genre
    genre_stats = df.explode('genre').groupby('genre')['author_name'].agg(
        lambda x: x.dropna().values.tolist() if not x.dropna().empty else []
    ).reset_index()

    # Define a custom sampling function that handles empty lists
    def sample_author(authors):
        if len(authors) > 0:  # Check if the list is not empty
            return np.random.choice(authors)
        else:
            return np.nan

    # Apply sampling to books with missing author names based on genre statistics
    missing_author_mask = df['author_name'].isnull()
    df.loc[missing_author_mask, 'author_name'] = df.loc[missing_author_mask, 'genre'].apply(
        lambda genres: sample_author(genre_stats[genre_stats['genre'].isin(genres)]['author_name'].explode().dropna())
    )

    # Drop the temporary 'genre' column
    df.drop(columns=['genre'], inplace=True)

    return df


def clean_date_column(df, date_column):
    # Identify year-only, year-month, and mm/dd/YYYY records with non-missing values
    year_only_mask = df[date_column].str.match(r'^\d{4}$') & df[date_column].notna()
    year_month_mask = df[date_column].str.match(r'^\d{4}-\d{2}$') & df[date_column].notna()
    mm_dd_yyyy_mask = df[date_column].str.match(r'^\d{2}/\d{2}/\d{4}$') & df[date_column].notna()

    # Create new column for conversion, preserving original values
    df['converted_date'] = df[date_column]

    # Apply conversion to year-only, year-month, and mm/dd/YYYY records in the new column
    df.loc[year_only_mask, 'converted_date'] = pd.to_datetime(
        df.loc[year_only_mask, 'converted_date'] + '-01-01',
        format='%Y-%m-%d', errors='coerce'
    ).dt.strftime('%Y-%m-%d')  # Format as YYYY-mm-dd

    df.loc[year_month_mask, 'converted_date'] = pd.to_datetime(
        df.loc[year_month_mask, 'converted_date'] + '-01',
        format='%Y-%m-%d', errors='coerce'
    ).dt.strftime('%Y-%m-%d')  # Format as YYYY-mm-dd

    df.loc[mm_dd_yyyy_mask, 'converted_date'] = pd.to_datetime(
        df.loc[mm_dd_yyyy_mask, 'converted_date'],
        format='%m/%d/%Y', errors='coerce'
    ).dt.strftime('%Y-%m-%d')  # Format as YYYY-mm-dd

    # Replace the original column with the converted dates
    df[date_column] = df['converted_date']

    # Drop the temporary column
    df.drop(columns=['converted_date'], inplace=True)

    return df


if __name__ == '__main__':
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)
    # print(data['author_name'].head(20))
    # print(data['book_name'].head(20))
    # print(data['author_name'].isna().sum())
    # print(data['author_name'].size)
    # guess_author_by_genre(data)
    # print(data['author_name'].size)
    # print(data['author_name'].isna().sum())
    # print(data['author_name'].head(20))
    # print(data['book_name'].head(20))
    # print(data['book_name'].tail())
    # print(data['date'].tail())
    # print(data['date'].isna().sum())
    # print(data['date'].head(50))
    print(data['author_name'].size)
    # print(data['author_name'].isna().sum())
    guess_author_by_genre(data)
    # data_with_author_guess = guess_publish_year(data.copy(deep=True))
    # Merge the output back to the initial dataset using a left join
    # merged_data = data.merge(data_with_author_guess[['author_name']], left_index=True, right_index=True, suffixes=('', '_guessed'), how='left')

    # Fill missing author names in the merged dataset with the original author names
    # merged_data['author_name'] = merged_data['author_name_guessed'].fillna(merged_data['author_name'])

    # Drop the guessed author name column after filling missing values
    # merged_data.drop(columns=['author_name_guessed'], inplace=True)
    # Merge the output back to the initial dataset based on the index
    print(data['author_name'].size)
    # print(data['author_name'].isna().sum())
    # print(data['date'].isna().sum())
    # print(data['date'].head(50))
    # print(data.info())
    # print(data['date'][0])
    # print(data["summary"][1])
