import os

# Config
DATA_DIR = "data/raw"

DATASETS = [
    {
        "name": "COCO 2014 Train",
        "url": "http://images.cocodataset.org/zips/train2014.zip",
        "zip": os.path.join(DATA_DIR, "train2014.zip"),
        "dir": os.path.join(DATA_DIR, "train2014"),
    },
    {
        "name": "COCO 2014 Val",
        "url": "http://images.cocodataset.org/zips/val2014.zip",
        "zip": os.path.join(DATA_DIR, "val2014.zip"),
        "dir": os.path.join(DATA_DIR, "val2014"),
    },
    {
        "name": "COCO 2017 Train",
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "zip": os.path.join(DATA_DIR, "train2017.zip"),
        "dir": os.path.join(DATA_DIR, "train2017"),
    },
    {
        "name": "COCO 2017 Val",
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "zip": os.path.join(DATA_DIR, "val2017.zip"),
        "dir": os.path.join(DATA_DIR, "val2017"),
    },
    {
        "name": "COCO Test 2017",
        "url": "http://images.cocodataset.org/zips/test2017.zip",
        "zip": os.path.join(DATA_DIR, "test2017.zip"),
        "dir": os.path.join(DATA_DIR, "test2017"),
    },
    {
        "name": "COCO 2014 Annotations",
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        "zip": os.path.join(DATA_DIR, "annotations_trainval2014.zip"),
        "dir": os.path.join(DATA_DIR, "annotations/annotations2014"),
    },
    {
        "name": "COCO 2017 Annotations",
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "zip": os.path.join(DATA_DIR, "annotations_trainval2017.zip"),
        "dir": os.path.join(DATA_DIR, "annotations/annotations2017"),
    },
]

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

ANNOTATIONS = {
    "train2014": os.path.join(
        RAW_DIR, "annotations/annotations2014/captions_train2014.json"
    ),
    "val2014": os.path.join(
        RAW_DIR, "annotations/annotations2014/captions_val2014.json"
    ),
    "train2017": os.path.join(
        RAW_DIR, "annotations/annotations2017/captions_train2017.json"
    ),
    "val2017": os.path.join(
        RAW_DIR, "annotations/annotations2017/captions_val2017.json"
    ),
}
