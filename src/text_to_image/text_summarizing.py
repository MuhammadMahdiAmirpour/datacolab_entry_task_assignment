import os
import sys
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# you need to uncomment this line when you want to run this code on google colab and load you data from google drive
# from google.colab import drive
# Mount Google Drive as a local file system
# drive.mount('/content/drive')

# Set the CUDA allocation configuration to use expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define global variables for input and output file paths, you can change this if you have your custom dataset
INPUT_FILE_PATH = '/content/drive/MyDrive/datacolab_dataset/booksummaries.txt'
OUTPUT_FILE_PATH = '/content/drive/MyDrive/datacolab_dataset/summary_outputs/output.csv'


class T5SummarizationPipeline:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False, use_fast=True)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.model = self.model.half()  # Enable mixed precision training

    def summarize_batch(self, input_texts: list[str]) -> list[str]:
        inputs = self.tokenizer.batch_encode_plus(["summarize: " + text for text in input_texts], return_tensors="pt",
                                                  max_length=1024, truncation=True, padding=True).to(self.model.device)
        with torch.cuda.stream(torch.cuda.Stream()):
            summary_ids = self.model.generate(**inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4,
                                              early_stopping=True)
        summarized_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
        return summarized_texts


def summarize_summaries_parallel(pipeline: T5SummarizationPipeline, df: pd.DataFrame, batch_size: int,
                                 output_file: str) -> None:
    # Open the output file in append mode
    with open(output_file, 'a') as f:
        # Check if the output file is empty
        if os.stat(output_file).st_size == 0:
            # Write the header for the output CSV file
            f.write('freebaseID,CondensedSummary\n')
            # Start the summarization process from the beginning
            start_index = 0
        else:
            # Read the existing freebaseIDs from the output CSV file
            existing_freebase_id = pd.read_csv(output_file, header=None)[0].tolist()
            # Find the index of the first non-condensed summary
            start_index = df[~df['freebase_id'].isin(existing_freebase_id)].index[0]

        # Summarize the remaining texts
        for i in range(start_index, len(df), batch_size):
            batch_texts = df['summary'].tolist()[i:i + batch_size]
            batch_summaries = pipeline.summarize_batch(batch_texts)
            batch_df = pd.DataFrame(
                {'freebaseID': df['freebase_id'].tolist()[i:i + batch_size], 'CondensedSummary': batch_summaries})

            # Write the batch to the output CSV file
            batch_df.to_csv(f, index=False, header=False)


def main():
    try:
        # Sample DataFrame with a 'summary' column
        column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
        data = pd.read_csv(INPUT_FILE_PATH, sep="\t", header=None, names=column_names)
        df = pd.DataFrame(data)

        # Set the output file path
        output_file = OUTPUT_FILE_PATH

        # Check if the output file exists
        if os.path.exists(output_file):
            # Read the existing freebaseIDs from the output CSV file
            existing_freebase_id = pd.read_csv(output_file, header=None)[0].tolist()
            # Check if all the freebaseIDs are already in the output CSV file
            if set(df['freebase_id']).issubset(existing_freebase_id):
                print("All summaries are condensed.")
                return

        # Apply parallelized summarization to the DataFrame
        pipeline = T5SummarizationPipeline(model_name="t5-small", device='cuda' if torch.cuda.is_available() else 'cpu')
        summarize_summaries_parallel(pipeline, df, batch_size=32, output_file=output_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
