import json
import re
import string

import pandas as pd

from missing_handler import handle_missing_date


# Function to safely parse JSON-like text into dictionaries
def safe_json_parse(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


def clean_summary_df(input_data):
    # clean the summary in the dataframe
    input_data["summary"] = input_data["summary"].map(lambda summary: clean_summary(summary))


def parse_json_to_dict(input_data):
    # column number 5 of this data frame is in a json format, so we convert it to dict
    input_data["freebase_id_json"] = input_data["freebase_id_json"].apply(safe_json_parse)


def clean_summary(summary):
    # get rid of punctuations and periods
    table = str.maketrans(dict.fromkeys(string.punctuation))
    text = summary.translate(table)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return text


def convert_string_to_date(df, column_name="date"):
    df = handle_missing_date(df)
    # Separate dates into two groups based on format (year_month_day and year)
    year_month_day_dates = df[column_name][df[column_name].str.contains('-')]
    year_dates = df[column_name][~df[column_name].str.contains('-')]
    # Convert year_month_day dates to datetime format
    df[column_name][df[column_name].str.contains('-')] = pd.to_datetime(year_month_day_dates)
    # Convert year dates to datetime format with a specific day and month (e.g., January 1st)
    df[column_name][~df[column_name].str.contains('-')] = pd.to_datetime(year_dates + '-01-01')


def rectify_data_type(input_df):
    convert_string_to_date(input_df)
    parse_json_to_dict(input_df)
    return input_df
