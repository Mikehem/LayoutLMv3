import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)

def main():
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # URL of the FUNSD dataset
    url = "https://guillaumejaume.github.io/FUNSD/dataset.zip"

    # Download the dataset
    zip_path = os.path.join(data_dir, 'funsd.zip')
    print("Downloading FUNSD dataset...")
    download_file(url, zip_path)

    # Extract the dataset
    print("Extracting FUNSD dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Remove the zip file
    os.remove(zip_path)

    print("FUNSD dataset downloaded and extracted successfully.")

if __name__ == "__main__":
    main()
