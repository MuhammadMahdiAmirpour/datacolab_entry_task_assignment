import os
import sys
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# you need to uncomment this line when you want to run this code on Google colab and load you data from Google Drive
# from google.colab import drive
# Mount Google Drive as a local file system
# drive.mount('/content/drive')

# Set the CUDA allocation configuration to use expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define global variables for input and output file paths, you can change this if you have your custom dataset
# INPUT_FILE_PATH = '/content/drive/MyDrive/datacolab_dataset/booksummaries.txt'
# OUTPUT_FILE_PATH = '/content/drive/MyDrive/datacolab_dataset/summary_outputs/output.csv'
INPUT_FILE_PATH = '../../data/datacolab_dataset/txt_format/booksummaries.txt'
OUTPUT_FILE_PATH = '../../results/summary_outputs/output.csv'


class TextSummarizer:
    """
    A class responsible for generating text summaries using the T5 Transformer model.
    """

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False, use_fast=True)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.model = self.model.half()  # Enable mixed precision training

    def summarize_batch(self, input_texts: list[str]) -> list[str]:
        """
        Generate text summaries for a batch of input texts using the T5 Transformer model.

        Args:
            input_texts (list[str]): A list of input texts to be summarized.

        Returns:
            list[str]: A list of generated text summaries.
        """
        inputs = self.tokenizer.batch_encode_plus(["summarize: " + text for text in input_texts], return_tensors="pt",
                                                  max_length=1024, truncation=True, padding=True).to(self.model.device)
        with torch.cuda.stream(torch.cuda.Stream()):
            summary_ids = self.model.generate(**inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4,
                                              early_stopping=True)
        summarized_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
        return summarized_texts


class SummarizationManager:
    """
    A class responsible for managing the text summarization process, including loading input data,
    checking existing summaries, and writing the results to the output file.
    """

    def __init__(self, input_file_path: str, output_file_path: str):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path

    def load_input_data(self) -> pd.DataFrame:
        """
        Load the input data from the specified file path.

        Returns:
            pd.DataFrame: The input DataFrame containing the text summaries.
        """
        column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
        data = pd.read_csv(self.input_file_path, sep="\t", header=None, names=column_names)
        return pd.DataFrame(data)

    def get_existing_freebase_ids(self) -> list[str]:
        """
        Get a list of existing freebaseIDs from the output CSV file.

        Returns:
            list[str]: A list of existing freebaseIDs.
        """
        if os.path.exists(self.output_file_path):
            return pd.read_csv(self.output_file_path, header=None)[0].tolist()
        return []

    def summarize_and_save(self, pipeline: TextSummarizer, df: pd.DataFrame, batch_size: int) -> None:
        """
        Summarize the text summaries in the input DataFrame in parallel and save the results to the output file.

        Args:
            pipeline (TextSummarizer): An instance of the TextSummarizer class.
            df (pd.DataFrame): The input DataFrame containing the text summaries.
            batch_size (int): The batch size for parallel processing.
        """
        # Open the output file in append mode
        with open(self.output_file_path, 'a') as f:
            # Check if the output file is empty
            if os.stat(self.output_file_path).st_size == 0:
                # Write the header for the output CSV file
                f.write('freebaseID,CondensedSummary\n')
                # Start the summarization process from the beginning
                start_index = 0
            else:
                # Get the existing freebaseIDs from the output CSV file
                existing_freebase_ids = self.get_existing_freebase_ids()
                # Find the index of the first non-condensed summary
                start_index = df[~df['freebase_id'].isin(existing_freebase_ids)].index[0]

            # Summarize the remaining texts
            for i in range(start_index, len(df), batch_size):
                batch_texts = df['summary'].tolist()[i:i + batch_size]
                batch_summaries = pipeline.summarize_batch(batch_texts)
                batch_df = pd.DataFrame(
                    {'freebaseID': df['freebase_id'].tolist()[i:i + batch_size], 'CondensedSummary': batch_summaries})

                # Write the batch to the output CSV file
                batch_df.to_csv(f, index=False, header=False)


def run_text_summarization(input_file_path, output_file_path):
    """
    The main entry point of the application.

    1. Create a SummarizationManager instance to handle the input data and output file
    2. Load the input data
    3. Check if the output file exists and if all summaries have been generated
    4. Create a TextSummarizer instance and apply parallelized summarization to the input DataFrame
    """
    try:
        manager = SummarizationManager(input_file_path, output_file_path)
        df = manager.load_input_data()

        # Check if the output file exists and if all summaries have been generated
        existing_freebase_ids = manager.get_existing_freebase_ids()
        if set(df['freebase_id']).issubset(existing_freebase_ids):
            print("All summaries are condensed.")
            return

        # Apply parallelized summarization to the DataFrame
        pipeline = TextSummarizer(model_name="t5-small", device='cuda' if torch.cuda.is_available() else 'cpu')
        manager.summarize_and_save(pipeline, df, batch_size=32)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


def main():
    run_text_summarization(INPUT_FILE_PATH, OUTPUT_FILE_PATH)


if __name__ == '__main__':
    main()
