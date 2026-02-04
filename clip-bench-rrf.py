import numpy as np
import faiss
import eccv_caption
import clip
import torch
from PIL import Image
import json
import os
import time

DATASET_PATH = "dataset/coco2014"
PARAPHRASE_PATH = f"{DATASET_PATH}/annotations/karpathy_paraphrases_hot.json"
MODEL_NAME = "ViT-B/32"
IMAGE_BATCH = 64
TEXT_BATCH = 256
RETRIEVE_K = 200
RRF_K = 20


def load_karpathy_paraphrases(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    images = []
    image_ids = []
    captions = []
    caption_ids = []
    variants = []

    for img in data:
        images.append(img["image"])
        image_ids.append(img["image_id"])
        for sent in img["captions"]:
            caption = sent["caption"].strip()
            captions.append(caption)
            caption_ids.append(sent["caption_id"])
            seen = set()
            uniq = []
            for t in [caption] + sent["paraphrases"]:
                t = t.strip()
                if t and t not in seen:
                    seen.add(t)
                    uniq.append(t)
            variants.append(uniq)

    return images, image_ids, captions, caption_ids, variants


def encode_images(model, preprocess, image_paths, batch_size, device):
    feats = []
    total = len(image_paths)
    for i in range(0, total, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [preprocess(Image.open(p).convert("RGB"))
                  for p in batch_paths]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            batch_feats = model.encode_image(image_input)
        feats.append(batch_feats)
        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, total)
            print(f"Encoded images: {done}/{total}")
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)


def encode_texts(model, texts, batch_size, device):
    feats = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        tokens = clip.tokenize(batch).to(device)
        with torch.no_grad():
            batch_feats = model.encode_text(tokens)
        feats.append(batch_feats)
        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, total)
            print(f"Encoded captions: {done}/{total}")
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)


def build_faiss_index(img_feats):
    imf = img_feats.detach().cpu().numpy().astype("float32", copy=False)
    imf = np.ascontiguousarray(imf)
    index = faiss.IndexFlatIP(imf.shape[1])
    index.add(imf)
    return index


def topk_faiss(index, text_feats, k):
    txf = text_feats.detach().cpu().numpy().astype("float32", copy=False)
    txf = np.ascontiguousarray(txf)
    k = min(k, index.ntotal)
    _, indices = index.search(txf, k)
    return indices


def rrf_fuse(rank_lists, k, topk):
    scores = {}
    for ranks in rank_lists:
        for r, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:topk]]


def main():
    device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    images, image_ids, captions, caption_ids, variants = load_karpathy_paraphrases(
        PARAPHRASE_PATH)
    print(f"Test images: {len(images)}")
    print(f"Test captions: {len(captions)}")

    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    t0 = time.time()
    image_feats = encode_images(
        model, preprocess, images, IMAGE_BATCH, device).cpu()

    all_texts = []
    offsets = []
    caption_indices = []
    for v in variants:
        offsets.append(len(all_texts))
        caption_indices.append(len(all_texts))
        all_texts.extend(v)
    variant_feats = encode_texts(model, all_texts, TEXT_BATCH, device).cpu()
    caption_feats = variant_feats[caption_indices]
    print(f"Encoded in {time.time() - t0:.1f}s")

    image_index = build_faiss_index(image_feats)
    text_index = build_faiss_index(caption_feats)

    i2t_rank = topk_faiss(text_index, image_feats, RETRIEVE_K).tolist()

    t2i_rank = []
    for i, v in enumerate(variants):
        start = offsets[i]
        end = start + len(v)
        ranks = topk_faiss(
            image_index, variant_feats[start:end], RETRIEVE_K).tolist()
        fused = rrf_fuse(ranks, RRF_K, RETRIEVE_K)
        t2i_rank.append(fused)
        if i % 1000 == 0:
            print(f"Fused {i}/{len(variants)} captions")

    i2t = {
        image_ids[i]: [caption_ids[j] for j in i2t_rank[i]]
        for i in range(len(image_ids))
    }
    t2i = {
        caption_ids[i]: [image_ids[j] for j in t2i_rank[i]]
        for i in range(len(caption_ids))
    }

    metric = eccv_caption.Metrics()
    scores = metric.compute_all_metrics(
        i2t_retrieved_items=i2t,
        t2i_retrieved_items=t2i,
        target_metrics=["coco_5k_recalls",
                        "eccv_map_at_r", "eccv_rprecision", "eccv_r1"],
        Ks=[1, 5, 10],
        verbose=True,
    )
    print("COCO 5K T2I recalls:")
    print(f"R@1: {scores['coco_5k_r1']['t2i']:.2f}")
    print(f"R@5: {scores['coco_5k_r5']['t2i']:.2f}")
    print(f"R@10: {scores['coco_5k_r10']['t2i']:.2f}")
    print("ECCV T2I metrics:")
    print(f"Map@R: {scores['eccv_map_at_r']['t2i']:.2f}")
    print(f"R-P: {scores['eccv_rprecision']['t2i']:.2f}")
    print(f"R@1: {scores['eccv_r1']['t2i']:.2f}")


if __name__ == "__main__":
    main()
