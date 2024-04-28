import json
import re
import string

import pandas as pd


def parse_json_column(df, column_name):
    def parse_json(cell):
        try:
            return json.loads(cell) if pd.notnull(cell) and isinstance(cell, str) else cell
        except json.JSONDecodeError:
            return cell

    df[column_name] = df[column_name].apply(parse_json)
    return df


def transform_date_formats(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df.loc[df['date'].str.len() == 7, 'date'] = pd.to_datetime(df.loc[df['date'].str.len() == 7, 'date'],
                                                               format='%Y-%m').dt.strftime('%Y-%m')
    return df


# Clean the summary text
def clean_summary(summary):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    text = summary.translate(table)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return text


# Clean the summary column in the DataFrame
def clean_summary_df(input_data):
    # Calculate the frequency table
    freq_table = input_data["summary"].value_counts()
    # Get the top 10 most frequent summaries
    top_10_summaries = freq_table.head(10).index
    # Replace the top 10 most frequent summaries with an empty string
    input_data.loc[input_data["summary"].isin(top_10_summaries), "summary"] = ""
    input_data["summary"] = input_data["summary"].map(clean_summary)


# Function to unify date format to year
def unify_date_format_year(input_df):
    years = [date[:4] if isinstance(date, str) and len(date) >= 4 else '' for date in input_df['date']]
    input_df['date'] = years
    input_df['date'] = pd.to_numeric(input_df['date'], errors='coerce')
    input_df['date'] = input_df['date'].astype(int, errors='ignore')
