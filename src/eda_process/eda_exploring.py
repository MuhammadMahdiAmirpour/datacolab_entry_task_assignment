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


if __name__ == '__main__':
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    df = pd.read_csv('../../data/booksummaries.txt', sep="\t", header=None, names=column_names)
    # data = clean_data(data)
    # Apply the extract_genre function to create a new 'genre' column
    # data['genre'] = data['freebase_id_json'].apply(extract_genre)
    # Flatten the 'genre' column (assuming each row has one or more genres)
    # data = data.explode('genre')
    # Reset the index to ensure unique labels
    # data.reset_index(drop=True, inplace=True)
    print("initial exploration")
    initial_exploration(df)
    print("check for missing values")
    check_for_missing_values(df)
    print("get sample of each column")
    get_sample_of_each_column(df)
    # histogram_of_each_column(data)
    # count_plot(data, hue="freebase_id", categorical_column="genre")
