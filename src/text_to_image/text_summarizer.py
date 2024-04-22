import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.cuda.amp import autocast, GradScaler
from google.colab import drive

# Mount Google Drive as a local file system
drive.mount('/content/drive')

# Set the CUDA allocation configuration to use expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load pre-trained T5 model and tokenizer outside the function for efficiency
model_name = "t5-small"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False, use_fast=True)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Enable mixed precision training
model = model.half()

class T5SummarizationModel(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False, use_fast=True)
        self.scaler = GradScaler()

    @autocast()
    def forward(self, input_texts):
        inputs = self.tokenizer.batch_encode_plus(["summarize: " + text for text in input_texts], return_tensors="pt", max_length=1024, truncation=True, padding=True).to(self.model.device)
        with torch.cuda.stream(torch.cuda.Stream()):
            summary_ids = self.model.generate(**inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summarized_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
        for summ_text in summarized_texts:
          print(summ_text)
        return summarized_texts

def summarize_summaries_parallel(model, df: pd.DataFrame, batch_size: int, output_file: str) -> None:
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
            existing_freebaseIDs = pd.read_csv(output_file, header=None)[0].tolist()
            # Find the index of the first non-condensed summary
            start_index = df[~df['freebase_id'].isin(existing_freebaseIDs)].index[0]

        # Summarize the remaining texts
        for i in range(start_index, len(df), batch_size):
            batch_texts = df['summary'].tolist()[i:i+batch_size]
            batch_summaries = model.forward(batch_texts)
            batch_df = pd.DataFrame({'freebaseID': df['freebase_id'].tolist()[i:i+batch_size], 'CondensedSummary': batch_summaries})

            # Write the batch to the output CSV file
            batch_df.to_csv(f, index=False, header=False)


if __name__ == '__main__':
    """
    You need to run this program in google colab using a GPU, as I did so.
    If you have strong hardware and want to run this file locally, 
    you need to change the path to dataset and output file.
    """
    # Sample DataFrame with a 'summary' column
    column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
    data = pd.read_csv('/content/drive/MyDrive/datacolab_dataset/booksummaries.txt', sep="\t", header=None, names=column_names)
    print(len(data))
    df = pd.DataFrame(data)
    print(df)

    # Set the output file path
    output_file = '/content/drive/MyDrive/datacolab_dataset/summary_outputs/output.csv'

    # Check if the output file exists
    if os.path.exists(output_file):
        # Read the existing freebaseIDs from the output CSV file
        existing_freebaseIDs = pd.read_csv(output_file, header=None)[0].tolist()
        # Check if all the freebaseIDs are already in the output CSV file
        if set(df['freebase_id']).issubset(existing_freebaseIDs):
            print("All summaries are condensed.")
            exit()

    # Apply parallelized summarization to the DataFrame
    model = T5SummarizationModel(model_name="t5-small", device=device)
    summarize_summaries_parallel(model, df, batch_size=32, output_file=output_file)