from collections import namedtuple
from parsers import load_karpathy_test, load_robo_dataset

Dataset = namedtuple("Dataset", [
    "images", "image_ids", "captions", "caption_ids", "caption_match_ids",
])

DATASET_CONFIGS = {
    "robo": {
        "dataset_path": "dataset/robotics_kitchen_dataset_v3",
        "ann_path": "dataset/robotics_kitchen_dataset_v3/annotations/ground_truth/robotics_kitchen.json",
    },
    "coco": {
        "dataset_path": "dataset/coco2014",
        "ann_path": "dataset/coco2014/annotations/karpathy_test.json",
    },
}


def load_dataset(name):
    cfg = DATASET_CONFIGS[name]
    if name == "robo":
        images, image_ids, captions, caption_ids = load_robo_dataset(
            cfg["ann_path"], cfg["dataset_path"])
        caption_match_ids = caption_ids
    elif name == "coco":
        images, image_ids, captions, caption_ids, caption_match_ids = load_karpathy_test(
            cfg["ann_path"], cfg["dataset_path"])
    print(f"Dataset '{name}': {len(images)} images, {len(captions)} captions")
    return Dataset(images, image_ids, captions, caption_ids, caption_match_ids)
