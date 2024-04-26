import pandas as pd

from data_utils import (clean_summary_df,
                        rectify_data_type)


def clean_data(input_df):
    # handling missing values (replace with empty string or custom value)
    input_df.fillna("")
    input_df.drop_duplicates(keep='first', inplace=True)
    rectify_data_type(input_df)
    clean_summary_df(input_df)
    return input_df


if __name__ == '__main__':
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../data/booksummaries.txt', sep="\t", header=None, names=column_names)
    data = clean_data(data)
    print(data.info())
    # print(data["summary"][1])
