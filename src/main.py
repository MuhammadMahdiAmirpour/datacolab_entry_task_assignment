import pandas as pd

from src.data_cleaning.data_cleaning import DataCleaner
from src.data_cleaning.missing_handler import MissingHandler
from src.eda_process.eda_processing import DataExplorer
from src.eda_process.univariate_analysis import UniVariateAnalyzer
from src.eda_process.bivariate_analysis import BiVariateAnalyzer
from src.text_summarization.text_summarizing import run_text_summarization
from src.image_generation.text_to_image import generate_image_from_text

INPUT_FILE_PATH = '../data/datacolab_dataset/txt_format/booksummaries.txt'
OUTPUT_FILE_PATH_FOR_SUMMARY = '../results/summary_outputs/output.csv'
OUTPUT_DIRECTORY_PATH_FOR_IMAGE_GENERATION = '../results/image_outputs/'
OUTPUT_DIRECTORY_PATH_FOR_VISUALIZATION = '../results/visualization/'
COLUMN_NAMES = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]


def load_data(input_file_path, column_names):
    data = pd.read_csv(input_file_path, sep="\t", header=None, names=column_names)
    return data


def clean_df(input_df):
    data_cleaner = DataCleaner(input_df)
    data_cleaner.clean_data()


def handle_missing_values(input_df):
    missing_handler = MissingHandler(input_df)
    missing_handler.handle_missing()


def explore_df(input_df):
    data_explorer = DataExplorer(input_df)
    data_explorer.explore_data()


def perform_uni_variate_analysis(input_df):
    uni_variate_analyzer = UniVariateAnalyzer(input_df)
    uni_variate_analyzer.perform_analysis()


def perform_bi_variate_analysis(input_df):
    bi_variate_analyzer = BiVariateAnalyzer(input_df)
    bi_variate_analyzer.analyze_bivariate_relationships()


def main():
    df = load_data(INPUT_FILE_PATH, COLUMN_NAMES)
    clean_df(df)
    handle_missing_values(df)
    explore_df(df)
    perform_uni_variate_analysis(df)
    perform_bi_variate_analysis(df)
    run_text_summarization(INPUT_FILE_PATH, OUTPUT_FILE_PATH_FOR_SUMMARY)
    generate_image_from_text(OUTPUT_FILE_PATH_FOR_SUMMARY, OUTPUT_DIRECTORY_PATH_FOR_IMAGE_GENERATION)


if __name__ == '__main__':
    main()
