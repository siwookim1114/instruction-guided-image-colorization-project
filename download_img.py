import os
import requests
from zipfile import ZipFile
from tqdm import tqdm

# Config
DATA_DIR = "data/raw"

# COCO
TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"

# ZIP FILE
TRAIN_ZIP = os.path.join(DATA_DIR, "train2017.zip")
VAL_ZIP = os.path.join(DATA_DIR, "val2017.zip")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "valid")

# COCO Annotations
ANNOT_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
ANNOT_ZIP = os.path.join(DATA_DIR, "annotations_trainval2014.zip")
ANNOT_DIR = os.path.join(DATA_DIR, "annotations/annotations2014")

# More Extra datasets

EXTRA_DATASETS = [
    {
        "name": "COCO 2014 Train",
        "url": "http://images.cocodataset.org/zips/train2014.zip",
        "zip": os.path.join(DATA_DIR, "train2014.zip"),
        "dir": os.path.join(DATA_DIR, "train2014")
    },
    {
        "name": "COCO 2014 Val",
        "url": "http://images.cocodataset.org/zips/val2014.zip",
        "zip": os.path.join(DATA_DIR, "val2014.zip"),
        "dir": os.path.join(DATA_DIR, "val2014")
    },
    {
        "name" : "COCO Test 2017",
        "url": "http://images.cocodataset.org/zips/test2017.zip",
        "zip": os.path.join(DATA_DIR, "test2017.zip"),
        "dir": os.path.join(DATA_DIR, "test2017")
    },
    {
        "name": "Flickr30k Images",
        "url": "https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k-images.zip",
        "zip": os.path.join(DATA_DIR, "flickr30k.zip"),
        "dir": os.path.join(DATA_DIR, "flickr30k")
    }
]

class Downloader():
    def __init__(self, url : str, dest_path: str, extract_dir: str):
        self.url = url
        self.dest_path = dest_path
        self.extract_dir = extract_dir
    
    def download(self):
        """Download the zip file"""
        if os.path.exists(self.dest_path):
            print(f"Already downloaded: {self.dest_path}")
            return

        os.makedirs(os.path.dirname(self.dest_path), exist_ok = True)
        response = requests.get(self.url, stream = True)
        total_size = int(response.headers.get("content-length", 0))

        with open(self.dest_path, "wb") as file, tqdm(
            desc = f"Downloading {os.path.basename(self.dest_path)}",
            total = total_size,
            unit = "B",
            unit_scale = True,
            unit_divisor = 1024 
        ) as pbar:
            for data in response.iter_content(chunk_size = 1024):
                size = file.write(data)
                pbar.update(size)
        
        print(f"Download complete: {self.dest_path}")

    def extract(self):
        """Extract the zip file"""
        # Skip extraction if already extracted
        if os.path.exists(self.extract_dir) and len(os.listdir(self.extract_dir)) > 0:
            print(f"Already extracted: {self.extract_dir}")
            return
        
        os.makedirs(self.extract_dir, exist_ok = True)
        with ZipFile(self.dest_path, "r") as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc = f"Extracting {os.path.basename(self.dest_path)}"):
                zip_ref.extract(member, self.extract_dir)
        print(f"Extraction complete: {self.extract_dir}")
    
if __name__ == "__main__":
    # Coco
    # train_downloader = ImageDownloader(TRAIN_URL, TRAIN_ZIP, TRAIN_DIR)
    # val_downloader = ImageDownloader(VAL_URL, VAL_ZIP, VAL_DIR)

    # for ds in EXTRA_DATASETS:
    #     downloader = ImageDownloader(ds['url'], ds['zip'], ds['dir'])
    #     downloader.download()
    #     downloader.extract()

    # COCO 2017 Annotations
    annot_downloader = Downloader(ANNOT_URL, ANNOT_ZIP, ANNOT_DIR)
    annot_downloader.download()
    annot_downloader.extract()

    print(f"Annotations: {ANNOT_DIR}/annotations")
    print("All done!")
    


    

