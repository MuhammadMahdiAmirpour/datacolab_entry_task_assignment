import string
import json
import re

import pandas as pd


def clean_data(input_data):
    # handling missing values (replace with empty string or custom value)
    input_data.fillna("")
    # dropping duplicate rows
    input_data.drop_duplicates(keep='first', inplace=True)
    # column number 5 of this data frame is in a json format, so we convert it to dict
    input_data[5] = input_data[5].apply(safe_json_parse)
    # clean the summary in the dataframe
    input_data[6] = input_data[6].map(lambda summary: clean_summary(summary))
    return input_data


def clean_summary(summary):
    # get rid of punctuations and periods
    table = str.maketrans(dict.fromkeys(string.punctuation))
    text = summary.translate(table)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return text


# Function to safely parse JSON-like text into dictionaries
def safe_json_parse(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


if __name__ == '__main__':
    data = pd.read_csv('../data/booksummaries.txt', sep="\t", header=None)
    data = clean_data(data)
    print(data[6][1])
