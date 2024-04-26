import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def initial_exploration(input_data):
    print("head")
    print(input_data.head())
    print("info")
    print(input_data.info())
    print("tail")
    print(input_data.tail())


def unify_date_formats_full(input_df):
    # Function to convert the date format
    # Function to convert the date format
    def convert_date_format(date_str):
        if isinstance(date_str, str) and len(date_str) == 4:
            return f"{date_str}-01-01"
        else:
            return date_str

    # Apply the conversion function to the 'date' column
    input_df['date'] = input_df['date'].apply(convert_date_format)

    # Convert the 'date' column to datetime format
    input_df['date'] = pd.to_datetime(input_df['date'], errors='coerce')


def unify_date_format_year(input_df):
    # Extract year from the date and fill missing values with ''
    years = [date[:4] if isinstance(date, str) and len(date) >= 4 else '' for date in input_df['date']]
    df['date'] = years


def check_for_missing_values(input_data):
    print(input_data.isnull().sum())


def get_sample_of_each_column(data):
    for col in data:
        print("column num: ", col)
        print(data[col][0])


def histogram_of_each_column(input_data):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=input_data, x="length")
    plt.title('Histogram of Numerical Column')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()


# def count_plot(input_data, hue, categorical_column):
#     plt.figure(figsize=(8, 6))
#     sns.countplot(data=input_data, hue=hue, x=categorical_column)
#     plt.title('Count Plot of Categorical Column')
#     plt.xlabel('Categories')
#     plt.ylabel('Count')
#     plt.show()

def count_plot(input_data, hue, categorical_column):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=input_data, hue=hue, x=categorical_column)
    plt.title('Count Plot of Categorical Column')
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.show()


def extract_genre(json_dict):
    return list(json_dict.values())


def check_loss_for_missing(input_df, column_name):
    df1 = input_df.copy(deep=True)
    s1 = df1.size
    df1.drop([column_name], axis=1, inplace=True)
    s2 = df1.size
    return s2 / s1


if __name__ == '__main__':
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    df = pd.read_csv('../../data/datacolab_dataset/booksummaries.txt', sep="\t", header=None, names=column_names)
    unify_date_format_year(df)
    print(df['date'])
    print(df['date'].value_counts())
    # data = clean_data(data)
    # Apply the extract_genre function to create a new 'genre' column
    # data['genre'] = data['freebase_id_json'].apply(extract_genre)
    # Flatten the 'genre' column (assuming each row has one or more genres)
    # data = data.explode('genre')
    # Reset the index to ensure unique labels
    # data.reset_index(drop=True, inplace=True)
    # print("initial exploration")
    # initial_exploration(df)
    # print("check for missing values")
    # check_for_missing_values(df)
    # print("get sample of each column")
    # get_sample_of_each_column(df)
    # histogram_of_each_column(data)
    # count_plot(data, hue="freebase_id", categorical_column="genre")
