import json
import re
import string

import pandas as pd


# Function to safely parse JSON-like text into dictionaries
def safe_json_parse(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


# Clean the summary text
def clean_summary(summary):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    text = summary.translate(table)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return text


# Clean the summary column in the DataFrame
def clean_summary_df(input_data):
    input_data["summary"] = input_data["summary"].map(clean_summary)


# Function to extract genre IDs from JSON
def extract_genre_ids(json_str):
    genres = safe_json_parse(json_str)
    return list(genres.keys()) if genres else []


# Function to unify date format to year
def unify_date_format_year(input_df):
    years = [date[:4] if isinstance(date, str) and len(date) >= 4 else '' for date in input_df['date']]
    input_df['date'] = years
    input_df['date'] = pd.to_numeric(input_df['date'], errors='coerce')
    input_df['date'] = input_df['date'].astype(int, errors='ignore')
