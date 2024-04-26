import re

import pandas as pd


def is_valid_date(date):
    print(date)
    return re.match(r'^\d{4}-\d{2}-\d{2}$', date)


def handle_missing_date(df: pd.DataFrame, column_name="date"):
    # Generate date range covering entire period
    start_date = pd.to_datetime(df[df[column_name].apply(lambda date: is_valid_date(date))]).min()
    end_date = pd.to_datetime(df[df[column_name].apply(lambda date: is_valid_date(date))]).max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create DataFrame with complete date range
    df_dates = pd.DataFrame(date_range, columns=['date'])

    # Merge with original DataFrame to fill missing dates
    df_filled = df_dates.merge(df, on='date', how='left')
    return df_filled
