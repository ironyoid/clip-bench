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
from parsers import load_robo_dataset
from gen_params import IMAGE_BATCH, TEXT_BATCH, RETRIEVE_K, METRIC_KS, RRF_K


DATASET_PATH = "dataset/robotics_kitchen_dataset_v3"
ANN_PATH = f"{DATASET_PATH}/annotations/ground_truth/robotics_kitchen.json"
PREPHRASE_PATH = f"{DATASET_PATH}/objects_caption_prephrases.json"

MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"

OUTPUT_PATH = "dataset/robotics_kitchen_dataset_v3/clip_output/openclip-llm-output.json"


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


def encode_texts(model, tokenizer, texts, batch_size, device):
    feats = []
    total = len(texts)
    for i in tqdm(range(0, total, batch_size), desc="Encoded captions"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            batch_feats = model.encode_text(tokens)
        feats.append(batch_feats)
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)


def build_faiss_index(feats):
    tf = feats.detach().cpu().numpy().astype("float32", copy=False)
    tf = np.ascontiguousarray(tf)
    index = faiss.IndexFlatIP(tf.shape[1])
    index.add(tf)
    return index


def topk_faiss(index, query_feats, k, return_scores=False):
    qf = query_feats.detach().cpu().numpy().astype("float32", copy=False)
    qf = np.ascontiguousarray(qf)
    k = min(k, index.ntotal)
    scores, indices = index.search(qf, k)
    if return_scores:
        return indices, scores
    return indices


def rrf_fuse(rank_lists, k, topk):
    scores = {}
    for ranks in rank_lists:
        for r, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked = ranked[:topk]
    return [idx for idx, _ in ranked], [score for _, score in ranked]


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

    images, image_ids, captions, caption_ids, variants = load_robo_dataset(
        ANN_PATH, DATASET_PATH, PREPHRASE_PATH)
    print(f"Images: {len(images)}")
    print(f"Captions: {len(captions)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    t0 = time.time()
    image_feats = encode_images(
        model, preprocess, images, IMAGE_BATCH, device).cpu()

    all_texts = []
    offsets = []
    for v in variants:
        offsets.append(len(all_texts))
        all_texts.extend(v)
    variant_feats = encode_texts(
        model, tokenizer, all_texts, TEXT_BATCH, device).cpu()
    print(f"Encoded in {time.time() - t0:.1f}s")

    image_index = build_faiss_index(image_feats)

    t2i_rank = []
    t2i_scores = []
    clip_query_times = []
    for i, v in enumerate(variants):
        q_start = time.time()
        start = offsets[i]
        end = start + len(v)
        ranks = topk_faiss(
            image_index, variant_feats[start:end], RETRIEVE_K
        ).tolist()
        fused, fused_scores = rrf_fuse(ranks, RRF_K, RETRIEVE_K)
        t2i_rank.append(fused)
        t2i_scores.append(fused_scores)
        clip_query_times.append(time.time() - q_start)
        if i % 1000 == 0:
            print(f"Fused {i}/{len(variants)} captions")
    avg_query_time = sum(clip_query_times) / max(1, len(clip_query_times))
    print(f"Avg CLIP query time: {avg_query_time:.4f}s")

    metrics = evaluate_metrics(
        caption_ids, image_ids, t2i_rank, t2i_scores, METRIC_KS
    )
    print("CLIP Text-to-Image metrics (ranx):")
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
            "prephrase_path": PREPHRASE_PATH,
            "model": MODEL_NAME,
            "pretrained": PRETRAINED,
            "image_batch": IMAGE_BATCH,
            "text_batch": TEXT_BATCH,
            "retrieve_k": RETRIEVE_K,
            "rrf_k": RRF_K,
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
