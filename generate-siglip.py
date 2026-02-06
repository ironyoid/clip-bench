import json
import os
import time

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import open_clip
import faiss

from ranx import Qrels, Run, evaluate
from transformers import AutoTokenizer
from parsers import load_robo_dataset


DATASET_PATH = "dataset/robotics_kitchen_dataset_v3"
ANN_PATH = f"{DATASET_PATH}/annotations/ground_truth/robotics_kitchen.json"

MODEL_NAME = "ViT-SO400M-14-SigLIP2-378"
PRETRAINED = "webli"

IMAGE_BATCH = 64
TEXT_BATCH = 256
RETRIEVE_K = 40
METRIC_KS = (1, 5, 40)

OUTPUT_PATH = "dataset/robotics_kitchen_dataset_v3/clip_output/siglip-output.json"


def encode_images(model, preprocess, image_paths, batch_size, device):
    feats = []
    total = len(image_paths)
    for i in tqdm(range(0, total, batch_size), desc="Encoded images"):
        batch_paths = image_paths[i:i + batch_size]
        images = [preprocess(Image.open(p).convert("RGB"))
                  for p in batch_paths]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            batch_feats = model.encode_image(image_input)
        feats.append(batch_feats)
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)


def get_hf_tokenizer(model_name):
    cfg = open_clip.get_model_config(model_name)
    text_cfg = cfg.get("text_cfg", {}) if cfg else {}
    tok_name = text_cfg.get(
        "hf_tokenizer_name") or text_cfg.get("hf_model_name")
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    return tokenizer, text_cfg.get("context_length")


def encode_texts_hf(model, tokenizer, texts, batch_size, device, context_length):
    feats = []
    total = len(texts)
    for i in tqdm(range(0, total, batch_size), desc="Encoded captions"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=context_length,
        )
        input_ids = tokens["input_ids"].to(device)
        with torch.no_grad():
            batch_feats = model.encode_text(input_ids)
        feats.append(batch_feats)
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)


def topk_faiss(query_feats, target_feats, k, return_scores=False):
    qf = query_feats.detach().cpu().numpy().astype("float32", copy=False)
    tf = target_feats.detach().cpu().numpy().astype("float32", copy=False)
    qf = np.ascontiguousarray(qf)
    tf = np.ascontiguousarray(tf)
    k = min(k, tf.shape[0])
    index = faiss.IndexFlatIP(tf.shape[1])
    index.add(tf)
    scores, indices = index.search(qf, k)
    if return_scores:
        return indices, scores
    return indices


def build_qrels(caption_ids, image_ids):
    obj_to_images = {}
    for idx, obj_id in enumerate(image_ids):
        obj_to_images.setdefault(obj_id, []).append(idx)

    qrels = {}
    for i, obj_id in enumerate(caption_ids):
        qid = f"q{i}"
        rels = {f"img{idx}": 1 for idx in obj_to_images.get(obj_id, [])}
        qrels[qid] = rels
    return qrels


def build_run(ranked_indices, ranked_scores=None):
    run = {}
    for i, indices in enumerate(ranked_indices):
        qid = f"q{i}"
        if ranked_scores is not None:
            scores = ranked_scores[i]
            run[qid] = {
                f"img{idx}": float(score)
                for idx, score in zip(indices, scores)
            }
        else:
            run[qid] = {
                f"img{idx}": float(len(indices) - rank)
                for rank, idx in enumerate(indices)
            }
    return run


def evaluate_metrics(caption_ids, image_ids, ranked_indices, ranked_scores, ks):
    qrels = Qrels(build_qrels(caption_ids, image_ids))
    run = Run(build_run(ranked_indices, ranked_scores))
    metrics = []
    for k in ks:
        metrics.extend([f"recall@{k}", f"precision@{k}", f"ndcg@{k}"])
    metrics.append(f"map@{max(ks)}")
    return evaluate(qrels, run, metrics)


def main():
    device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    images, image_ids, captions, caption_ids = load_robo_dataset(
        ANN_PATH, DATASET_PATH)
    print(f"Images: {len(images)}")
    print(f"Captions: {len(captions)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    tokenizer, ctx_len = get_hf_tokenizer(MODEL_NAME)

    t0 = time.time()
    image_feats = encode_images(
        model, preprocess, images, IMAGE_BATCH, device).cpu()
    text_feats = encode_texts_hf(
        model, tokenizer, captions, TEXT_BATCH, device, ctx_len).cpu()
    print(f"Encoded in {time.time() - t0:.1f}s")

    t_search = time.time()
    t2i_rank_np, t2i_scores_np = topk_faiss(
        text_feats, image_feats, RETRIEVE_K, return_scores=True
    )
    search_time = time.time() - t_search
    avg_query_time = search_time / max(1, len(captions))
    print(f"Avg query time: {avg_query_time:.4f}s")
    t2i_rank = t2i_rank_np.tolist()
    t2i_scores = t2i_scores_np.tolist()

    metrics = evaluate_metrics(
        caption_ids, image_ids, t2i_rank, t2i_scores, METRIC_KS
    )
    print("SigLIP2 Text-to-Image metrics (ranx):")
    for k in METRIC_KS:
        r = metrics[f"recall@{k}"]
        p = metrics[f"precision@{k}"]
        n = metrics[f"ndcg@{k}"]
        print(f"R@{k}: {r:.2f}  P@{k}: {p:.2f}  nDCG@{k}: {n:.2f}")
    print(f"MAP@{max(METRIC_KS)}: {metrics[f'map@{max(METRIC_KS)}']:.2f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out = {
        "meta": {
            "dataset_path": DATASET_PATH,
            "ann_path": ANN_PATH,
            "model": MODEL_NAME,
            "pretrained": PRETRAINED,
            "image_batch": IMAGE_BATCH,
            "text_batch": TEXT_BATCH,
            "retrieve_k": RETRIEVE_K,
        },
        "images": images,
        "image_ids": image_ids,
        "captions": captions,
        "caption_ids": caption_ids,
        "t2i_rank": t2i_rank,
        "t2i_scores": t2i_scores,
        "metrics": metrics,
    }
    json.dump(out, open(OUTPUT_PATH, "w", encoding="utf-8"), indent=2)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
