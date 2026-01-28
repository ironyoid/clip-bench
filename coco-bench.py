import glob
import io
import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PARQUET_DIR = "dataset/coco/data"
SEED = 0
MAX_IMAGES = 50
ALPHA = 0.35

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def iter_rows(rng):
    files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    rng.shuffle(files)
    for path in files:
        df = pd.read_parquet(path)
        idxs = list(df.index)
        rng.shuffle(idxs)
        for i in idxs:
            yield df.loc[i]


def main():
    rng = random.Random(SEED)
    shown = 0
    for row in iter_rows(rng):
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        objects = row["objects"]
        labels = [COCO_CLASSES[i] for i in objects["category"]]

        print("image_id:", row["image_id"])
        print("descriptions:", labels)

        img = np.array(image).astype(np.uint8)
        bboxes = np.stack(objects["bbox"])
        overlay = img.copy()
        for x, y, bw, bh in bboxes:
            color = (rng.randint(0, 255), rng.randint(
                0, 255), rng.randint(0, 255))
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        img = cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0)

        cv2.imshow("coco", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(0) == 27:
            break

        shown += 1
        if shown >= MAX_IMAGES:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
