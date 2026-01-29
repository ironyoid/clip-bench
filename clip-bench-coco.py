from PIL import Image
import json
import os
import time

import clip
import faiss
import numpy as np
import torch
from eccv_caption import metrics as eccv_metrics


DATASET_DIR = "dataset/coco2014"
ANNOTATION_FILES = [
    os.path.join(DATASET_DIR, "annotations", "captions_val2014.json"),
]
VAL_IMAGES_DIR = os.path.join(DATASET_DIR, "images", "val2014")

MODEL_NAME = "ViT-B/32"
IMAGE_BATCH = 64
TEXT_BATCH = 256
QUERY_BATCH = 512
K_VALUES = (1, 5, 10)


def load_coco_subset(annotation_files, image_ids, caption_ids):
    image_id_to_file = {}
    caption_id_to_text = {}

    for path in annotation_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for img in data["images"]:
            image_id = img["id"]
            if image_id in image_ids:
                image_id_to_file[image_id] = img["file_name"]
        for ann in data["annotations"]:
            caption_id = ann["id"]
            if caption_id in caption_ids and ann["image_id"] in image_id_to_file:
                caption_id_to_text[caption_id] = ann["caption"]

    return image_id_to_file, caption_id_to_text


def resolve_image_path(file_name):
    if file_name.startswith("COCO_val2014_"):
        return os.path.join(VAL_IMAGES_DIR, file_name)
    return os.path.join(VAL_IMAGES_DIR, file_name)


def encode_images(model, preprocess, image_paths, device, batch_size):
    features = []
    total = len(image_paths)
    for start in range(0, total, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            images.append(preprocess(img))
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(image_input)
        features.append(batch_features.cpu())
        if (start // batch_size) % 20 == 0:
            print(f"Encoded images: {min(start + batch_size, total)}/{total}")
    return torch.cat(features, dim=0)


def encode_texts(model, texts, device, batch_size):
    features = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch_texts = texts[start:start + batch_size]
        tokens = clip.tokenize(batch_texts).to(device)
        with torch.no_grad():
            batch_features = model.encode_text(tokens)
        features.append(batch_features.cpu())
        if (start // batch_size) % 20 == 0:
            print(
                f"Encoded captions: {min(start + batch_size, total)}/{total}")
    return torch.cat(features, dim=0)


def faiss_topk(query_features, query_ids, index_features, index_ids, top_k, batch_size):
    dim = index_features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(index_features)

    retrieved = {}
    total = len(query_ids)
    for start in range(0, total, batch_size):
        batch = query_features[start:start + batch_size]
        _, indices = index.search(batch, top_k)
        for row, row_indices in enumerate(indices):
            retrieved[query_ids[start + row]] = [index_ids[i]
                                                 for i in row_indices]
        if (start // batch_size) % 20 == 0:
            print(f"Retrieved: {min(start + batch_size, total)}/{total}")
    return retrieved


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    metrics = eccv_metrics.Metrics()
    coco_image_ids = set(metrics.coco_gts["i2t"].keys())
    coco_caption_ids = set(metrics.coco_gts["t2i"].keys())

    image_id_to_file, caption_id_to_text = load_coco_subset(
        ANNOTATION_FILES,
        coco_image_ids,
        coco_caption_ids,
    )

    image_ids = sorted(image_id_to_file.keys())
    caption_ids = sorted(caption_id_to_text.keys())

    image_paths = [resolve_image_path(image_id_to_file[iid])
                   for iid in image_ids]
    captions = [caption_id_to_text[cid] for cid in caption_ids]

    print(f"Images: {len(image_ids)}")
    print(f"Captions: {len(caption_ids)}")

    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    start = time.time()
    image_features = encode_images(
        model, preprocess, image_paths, device, IMAGE_BATCH)
    text_features = encode_texts(model, captions, device, TEXT_BATCH)
    print(f"Encoded in: {time.time() - start:.1f}s")

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    image_np = np.ascontiguousarray(
        image_features.cpu().numpy().astype("float32", copy=False))
    text_np = np.ascontiguousarray(
        text_features.cpu().numpy().astype("float32", copy=False))

    max_k = max(K_VALUES)
    max_i2t_gt = max(len(v) for v in metrics.eccv_gts["i2t"].values())
    max_t2i_gt = max(len(v) for v in metrics.eccv_gts["t2i"].values())
    top_k_i2t = max(max_k, max_i2t_gt)
    top_k_t2i = max(max_k, max_t2i_gt)

    print("Building image-to-text retrieval lists...")
    i2t = faiss_topk(
        image_np,
        image_ids,
        text_np,
        caption_ids,
        top_k_i2t,
        QUERY_BATCH,
    )

    print("Building text-to-image retrieval lists...")
    t2i = faiss_topk(
        text_np,
        caption_ids,
        image_np,
        image_ids,
        top_k_t2i,
        QUERY_BATCH,
    )

    target_metrics = [
        "coco_5k_recalls",
        "cxc_recalls",
        "eccv_map_at_r",
        "eccv_rprecision",
        "eccv_r1",
    ]
    scores = metrics.compute_all_metrics(
        i2t_retrieved_items=i2t,
        t2i_retrieved_items=t2i,
        target_metrics=target_metrics,
        Ks=list(K_VALUES),
    )

    print("Metrics (percent):")
    for name in sorted(scores.keys()):
        score = scores[name]
        print(
            f"{name} i2t: {score['i2t'] * 100:.2f} t2i: {score['t2i'] * 100:.2f}")


if __name__ == "__main__":
    main()
