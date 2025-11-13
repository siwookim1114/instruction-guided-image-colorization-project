import os
import json
import random
from tqdm import tqdm
from config import RAW_DIR, PROCESSED_DIR, ANNOTATIONS


def caption_to_instruction(caption: str) -> str:
    """Converts a caption into an instruction for colorization"""
    templates = [
        f"Colorize this image so that {caption}",
        f"Apply realistic colors according to: {caption}",
        f"Use appropriate colors to match: {caption}",
        f"Make the image colorful as described: {caption}",
        f"Color the objects to reflect this scene: {caption}",
    ]
    return random.choice(templates)


def build_instruction_json(annot_path: str, split_name: str):
    """Create an instruction mapping (image_id -> text)"""
    if not os.path.exists(annot_path):
        print(f"Missing file: {annot_path}")
        return

    with open(annot_path, "r") as f:
        data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    instructions = {}

    for ann in tqdm(data["annotations"], desc=f"Processing {split_name}"):
        img_id = ann["image_id"]
        caption = ann["caption"]
        instruction = caption_to_instruction(caption)
        file_name = id_to_filename.get(img_id)
        if not file_name:
            continue
        instructions.setdefault(file_name, []).append(instruction)

    single_instruction = {k: random.choice(v) for k, v in instructions.items()}

    out_path = os.path.join(PROCESSED_DIR, f"instructions_{split_name}.json")
    with open(out_path, "w") as f:
        json.dump(single_instruction, f, indent=2)

    print(f"Saved {len(single_instruction)} instructions → {out_path}")
    return out_path


if __name__ == "__main__":
    print("Generating instruction datasets (COCO 2014 + 2017) ...")
    for split_name, annot_path in ANNOTATIONS.items():
        build_instruction_json(annot_path, split_name)
