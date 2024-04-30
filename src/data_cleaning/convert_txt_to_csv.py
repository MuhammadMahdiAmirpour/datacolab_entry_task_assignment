import argparse
import logging
import pandas as pd


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert a tab-separated text file to a CSV file.')
    parser.add_argument('input_file', help='Path to the input tab-separated text file')
    parser.add_argument('output_file', help='Path to the output CSV file')
    args = parser.parse_args()

    try:
        # Read the TXT file into a DataFrame
        logging.info(f'Reading data from {args.input_file}')
        df = pd.read_csv(args.input_file, sep='\t')

        # Save the DataFrame to a CSV file
        logging.info(f'Saving data to {args.output_file}')
        df.to_csv(args.output_file, index=False)
        logging.info('Data conversion completed successfully.')
    except FileNotFoundError:
        logging.error(f'Error: {args.input_file} not found.')
    except PermissionError:
        logging.error(f'Error: Unable to write to {args.output_file}. Check file permissions.')
    except Exception as e:
        logging.error(f'Error: {e}')


if __name__ == '__main__':
    main()
