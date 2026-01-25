from PIL import Image
import numpy as np
import clip
import torch
import csv
import os
import time
from collections import defaultdict
import faiss


def load_flickr8k_captions(captions_path, images_dir=None):
    captions = []
    caption_to_image = []
    image_index = {}
    image_files = []

    with open(captions_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image" not in reader.fieldnames or "caption" not in reader.fieldnames:
            raise ValueError("captions.txt must have columns: image, caption")
        for row in reader:
            image_name = row["image"].strip()
            caption = row["caption"].strip()
            if image_name not in image_index:
                image_index[image_name] = len(image_files)
                image_files.append(image_name)
            img_idx = image_index[image_name]
            captions.append(caption)
            caption_to_image.append(img_idx)

    return image_files, captions, caption_to_image


def encode_images(model, preprocess, image_root, image_files, device, batch_size):
    features = []
    total = len(image_files)
    for start in range(0, total, batch_size):
        batch_files = image_files[start:start + batch_size]
        images = []
        for name in batch_files:
            path = os.path.join(image_root, name)
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


def compute_metrics(text_features, image_features, caption_to_image, query_batch):
    K10 = 10
    K5 = 5
    text_np = text_features.cpu().numpy().astype("float32", copy=False)
    image_np = image_features.cpu().numpy().astype("float32", copy=False)
    text_np = np.ascontiguousarray(text_np)
    image_np = np.ascontiguousarray(image_np)

    dim = image_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(image_np)

    num_caps = text_np.shape[0]
    hits1 = 0
    hits5 = 0
    hits10 = 0

    for i0 in range(0, num_caps, query_batch):
        batch = text_np[i0:i0 + query_batch]
        _, indices = index.search(batch, K10)
        gt = np.array(caption_to_image[i0:i0 + len(batch)])
        hits1 += int(np.sum(indices[:, 0] == gt))
        hits5 += int(np.sum(np.any(indices[:, :K5] == gt[:, None], axis=1)))
        hits10 += int(np.sum(np.any(indices == gt[:, None], axis=1)))

    return {
        "R@1": hits1 / num_caps * 100.0,
        "R@5": hits5 / num_caps * 100.0,
        "R@10": hits10 / num_caps * 100.0,
    }


def main():
    dataset_dir = "dataset"
    captions_file = "captions.txt"
    images_folder = "Images"
    model_name = "ViT-B/32"
    image_batch = 64
    text_batch = 256
    query_batch = 64

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    captions_path = os.path.join(dataset_dir, captions_file)
    images_dir = os.path.join(dataset_dir, images_folder)
    image_files, captions, caption_to_image = load_flickr8k_captions(
        captions_path, images_dir)

    print(f"Images: {len(image_files)}")
    print(f"Captions: {len(captions)}")

    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    start = time.time()
    image_features = encode_images(
        model, preprocess, images_dir, image_files, device, image_batch)
    text_features = encode_texts(model, captions, device, text_batch)
    print(f"Encoded in: {time.time() - start:.1f}s")

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    print("Computing text-to-image retrieval!")
    t2i_metrics = compute_metrics(
        text_features,
        image_features,
        caption_to_image,
        query_batch=query_batch * 4,
    )
    print(t2i_metrics)


if __name__ == "__main__":
    main()
