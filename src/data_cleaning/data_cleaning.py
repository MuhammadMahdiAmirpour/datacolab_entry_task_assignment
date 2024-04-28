import pandas as pd

from src.data_cleaning.data_utils import (transform_date_formats,
                                          parse_json_column,
                                          clean_summary_df)
from src.data_cleaning.missing_handler import handle_missing_values


def clean_data(input_df):
    # handling missing values (replace with empty string or custom value)
    input_df.fillna("")
    input_df.drop_duplicates(keep='first', inplace=True)
    # unify_date_format_year(input_df)
    transform_date_formats(input_df)
    parse_json_column(input_df, "freebase_id_json")
    clean_summary_df(input_df)
    handle_missing_values(input_df)
    input_df = input_df.dropna(subset=['date', 'freebase_id_json'])
    return input_df


if __name__ == '__main__':
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/datacolab_dataset/txt_format/booksummaries.txt', sep="\t", header=None,
                       names=column_names)
    data_cleaned = clean_data(data)
    print(data_cleaned["date"][50:60])
