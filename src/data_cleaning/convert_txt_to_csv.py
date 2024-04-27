import pandas as pd

input_file = "../../data/datacolab_dataset/txt_format/booksummaries.txt"
output_file = "../../data/datacolab_dataset/csv_format/booksummaries.csv"

# Read the TXT file into a DataFrame
df = pd.read_csv(input_file, sep='\t')  # Assuming tab-separated values, adjust the separator as needed

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)  # Index=False to exclude row numbers in the CSV file
