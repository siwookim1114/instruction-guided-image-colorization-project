import os
import requests
from zipfile import ZipFile
from tqdm import tqdm
from config import DATA_DIR, DATASETS


class Downloader:
    def __init__(self, url: str, dest_path: str, extract_dir: str):
        self.url = url
        self.dest_path = dest_path
        self.extract_dir = extract_dir

    def download(self):
        """Download the zip file"""
        if os.path.exists(self.dest_path):
            print(f"Already downloaded: {self.dest_path}")
            return

        os.makedirs(os.path.dirname(self.dest_path), exist_ok=True)
        response = requests.get(self.url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(self.dest_path, "wb") as file, tqdm(
            desc=f"Downloading {os.path.basename(self.dest_path)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

        print(f"Download complete: {self.dest_path}")

    def extract(self):
        """Extract the zip file"""
        # Skip extraction if already extracted
        if os.path.exists(self.extract_dir) and len(os.listdir(self.extract_dir)) > 0:
            print(f"Already extracted: {self.extract_dir}")
            return

        os.makedirs(self.extract_dir, exist_ok=True)
        with ZipFile(self.dest_path, "r") as zip_ref:
            for member in tqdm(
                zip_ref.infolist(),
                desc=f"Extracting {os.path.basename(self.dest_path)}",
            ):
                zip_ref.extract(member, self.extract_dir)
        print(f"Extraction complete: {self.extract_dir}")


if __name__ == "__main__":
    for ds in DATASETS:
        downloader = Downloader(ds["url"], ds["zip"], ds["dir"])
        downloader.download()
        downloader.extract()

    print("All done!")
