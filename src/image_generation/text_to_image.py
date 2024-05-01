import concurrent.futures
import os
import re
import sys

import pandas as pd
import torch
from diffusers import AmusedPipeline
from tqdm import tqdm

# you need to uncomment this line when you want to run this code on google colab and load you data from google drive
# from google.colab import drive
# Mount Google Drive as a local file system
# drive.mount('/content/drive')

# Set the CUDA allocation configuration to use expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define global variables for input and output file paths, you can change this if you have your custom dataset
# INPUT_FILE_PATH = '/content/drive/MyDrive/datacolab_dataset/booksummaries.txt'
# OUTPUT_DIR = '/content/drive/MyDrive/datacolab_dataset/image_outputs'
INPUT_FILE_PATH = '../../results/summary_outputs/output.csv'
OUTPUT_DIRECTORY_PATH = '../../results/image_outputs/'


def get_existing_image_filenames(output_directory_path) -> set:
    """
    Get a set of existing image filenames in the output directory.
    """
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path, exist_ok=True)
    return set([f.split('.')[0] for f in os.listdir(output_directory_path) if f.endswith('.png')])


class ImageGenerator:
    """
    A class responsible for generating images from text prompts using the Amused Diffusion model.
    """

    def __init__(self, device: torch.device, output_directory_path):
        self.device = device
        self.pipe = self.load_amused_pipeline()
        self.out_dir_path = output_directory_path
        self.existing_images = get_existing_image_filenames(self.out_dir_path)

    def load_amused_pipeline(self) -> AmusedPipeline:
        """
        Load the pre-trained Amused Diffusion model and move it to the specified device.
        """
        pipe = AmusedPipeline.from_pretrained("amused/amused-256", device_map="auto", low_cpu_mem_usage=True)
        return pipe.to(self.device)

    def generate_image(self, freebase_id: str, prompt: str) -> None:
        """
        Generate an image from the given text prompt and save it with the freebaseID as the filename.

        If the image for the current freebaseID already exists, print a message and return.
        """
        # Replace forward slashes with underscores in the freebaseID
        cleaned_freebase_id = re.sub(r'[/]', '_', freebase_id)

        # Check if the image with the current freebaseID already exists
        if cleaned_freebase_id in self.existing_images:
            print(f"Image for {cleaned_freebase_id} already exists, skipping...")
            return

        try:
            # Generate the image using the Amused Diffusion model
            image = self.pipe(prompt, negative_prompt="low quality, ugly", generator=torch.manual_seed(0)).images[0]
            image_path = os.path.join(self.out_dir_path, f"{cleaned_freebase_id}.png")
            image.save(image_path)
            print(f"Image saved: {image_path}")
            sys.stdout.flush()
        except Exception as e:
            print(f"Error generating image for {cleaned_freebase_id}: {e}")


class DataManager:
    """
    A class responsible for managing the input data and the image generation process.
    """

    def __init__(self, input_file_path: str):
        self.input_file_path = input_file_path

    def load_input_data(self) -> pd.DataFrame:
        """
        Load the input data from the specified file path.

        Returns:
            pd.DataFrame: The input DataFrame containing the text prompts and freebaseIDs.
        """
        column_names = ["length", "freebase_id", "book_name", "author_name", "date", "freebase_id_json", "summary"]
        data = pd.read_csv(self.input_file_path, sep="\t", header=None, names=column_names)
        return pd.DataFrame(data)


def generate_image_from_text(input_file_path, output_directory_path):
    """
    The main entry point of the application.

    1. Create a DataManager instance to handle the input data
    2. Load the input data
    3. Create an ImageGenerator instance and start the parallel image generation process
    """
    try:
        manager = DataManager(input_file_path)
        df = manager.load_input_data()

        # Create an ImageGenerator instance and start the parallel image generation process
        generator = ImageGenerator(device='cuda' if torch.cuda.is_available() else 'cpu',
                                   output_directory_path=output_directory_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(generator.generate_image, row['freebase_id'], row['summary']) for _, row in
                       tqdm(df.iterrows(), total=len(df), desc="Generating images")]
            concurrent.futures.wait(futures)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


def main():
    generate_image_from_text(INPUT_FILE_PATH, OUTPUT_DIRECTORY_PATH)


if __name__ == '__main__':
    main()
