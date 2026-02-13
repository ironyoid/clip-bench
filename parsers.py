import json
import os
import pickle

import cv2
import numpy as np
from PIL import Image


def decode_uncompressed_rle(rle):
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    value = 0
    for count in counts:
        if value == 1:
            flat[idx:idx + count] = 1
        idx += count
        value = 1 - value
    return flat.reshape((h, w), order="F")


def load_karpathy_test(ann_path, coco_root, single_caption=False):
    data = json.load(open(ann_path, "r", encoding="utf-8"))
    images = []
    image_ids = []
    captions = []
    caption_ids = []
    caption_image_ids = []

    for img in data.get("images", []):
        if img.get("split") != "test":
            continue
        sents = img.get("sentences", [])
        if not sents:
            continue
        img_path = f"{coco_root}/{img['filepath']}/{img['filename']}"
        images.append(img_path)
        image_ids.append(img["cocoid"])
        if single_caption:
            sent = sents[0]
            captions.append(sent["raw"].strip())
            caption_ids.append(sent["sentid"])
            caption_image_ids.append(img["cocoid"])
        else:
            for sent in sents:
                captions.append(sent["raw"].strip())
                caption_ids.append(sent["sentid"])
                caption_image_ids.append(img["cocoid"])

    return images, image_ids, captions, caption_ids, caption_image_ids


def load_robo_dataset(ann_path, coco_root, prephrase_path=None, padding=0):
    data = json.load(open(ann_path, "r", encoding="utf-8"))
    object_caption_path = os.path.join(coco_root, "objects_caption.json")
    if prephrase_path and os.path.exists(prephrase_path):
        object_captions = json.load(
            open(prephrase_path, "r", encoding="utf-8"))
    else:
        object_captions = json.load(
            open(object_caption_path, "r", encoding="utf-8"))
    masked_objects_dir = os.path.join(coco_root, "masked_objects")
    os.makedirs(masked_objects_dir, exist_ok=True)

    images = []
    image_ids = []
    captions = []
    caption_ids = []
    variants = []

    annotations = data["annotations"]
    for frame_id in sorted(annotations.keys(), key=lambda x: int(x)):
        for mask_info in annotations[frame_id]["masks"]:
            mask_rel_path = mask_info["mask_path"]
            mask_name = os.path.basename(mask_rel_path)
            frame_num = int(mask_name.replace(
                ".pkl", "").split("_")[0].replace("frame", ""))
            object_id = int(mask_info["object_id"])
            pad_suffix = f"_pad{padding}" if padding > 0 else ""
            out_path = os.path.join(
                masked_objects_dir, f"frame{frame_num}_obj{object_id}{pad_suffix}.jpg")
            frame_path = os.path.join(
                coco_root, "frames", "robotics_kitchen", f"{frame_num - 1}.jpg"
            )
            mask_path = os.path.join(
                coco_root, "mask_cache", mask_rel_path)

            image = np.array(Image.open(frame_path).convert("RGB"))
            rle = pickle.load(open(mask_path, "rb"))
            mask = decode_uncompressed_rle(rle).astype(bool)

            if padding > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * padding + 1, 2 * padding + 1))
                mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

            masked = np.zeros_like(image)
            masked[mask] = image[mask]
            ys, xs = np.where(mask)
            crop = masked[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            Image.fromarray(crop).save(out_path)

            images.append(out_path)
            image_ids.append(object_id)

    present_object_ids = sorted(set(image_ids))
    for object_id in present_object_ids:
        obj = object_captions[str(object_id)]
        caption = obj["object_caption"].strip()
        captions.append(caption)
        caption_ids.append(object_id)
        if prephrase_path and os.path.exists(prephrase_path):
            prephrases = obj.get("prephrases", [])
            seen = set()
            uniq = []

            def add_variant(text):
                text = text.strip()
                if text and text not in seen:
                    seen.add(text)
                    uniq.append(text)
            add_variant(caption)
            for p in prephrases:
                add_variant(f"{p} {caption}")
            variants.append(uniq)

    if prephrase_path and os.path.exists(prephrase_path):
        return images, image_ids, captions, caption_ids, variants
    return images, image_ids, captions, caption_ids
