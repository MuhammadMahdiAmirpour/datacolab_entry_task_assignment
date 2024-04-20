import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def initial_exploration(data):
    print(data.head())
    print(data.info())
    print(data.tail())


def check_for_missing_values(data):
    print(data.isnull().sum())


def get_sample_of_each_column(data):
    for col in data:
        print("column num: ", col)
        print(data[col][0])


def histogram_of_each_column(data):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x=0)
    plt.title('Histogram of Numerical Column')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()


def count_plot(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=6)
    plt.title('Count Plot of Categorical Column')
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.show()


if __name__ == '__main__':
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('../../data/booksummaries.txt', sep="\t", header=None, names=column_names)
    print("initial exploration")
    initial_exploration(data)
    print("check for missing values")
    check_for_missing_values(data)
    print("get sample of each column")
    get_sample_of_each_column(data)
    histogram_of_each_column(data)
    count_plot(data)
