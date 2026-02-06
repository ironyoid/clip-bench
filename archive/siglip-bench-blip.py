import time

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import open_clip
import faiss

from ranx import Qrels, Run, evaluate
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
from parsers import load_robo_dataset


DATASET_PATH = "dataset/robotics_kitchen_dataset_v3"
ANN_PATH = f"{DATASET_PATH}/annotations/ground_truth/robotics_kitchen.json"

MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"

IMAGE_BATCH = 64
TEXT_BATCH = 256
RETRIEVE_K = 10
METRIC_KS = (1, 5, 10)
BLIP2_MODEL_ID = "Salesforce/blip2-itm-vit-g"
BLIP2_IMAGE_BATCH = 2


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


def blip2_rerank(blip2_model, blip2_processor, captions, image_paths, t2i_rank, batch_size, device, dtype):
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    reranked = []
    reranked_scores = []
    total_caps = len(captions)
    for i in tqdm(range(total_caps), desc="BLIP2 rerank"):
        cand = t2i_rank[i]
        scores = []
        for j in range(0, len(cand), batch_size):
            chunk = cand[j:j + batch_size]
            imgs = [pil_images[k] for k in chunk]
            texts = [captions[i]] * len(chunk)
            inputs = blip2_processor(
                images=imgs,
                text=texts,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    device, dtype=dtype)
            with torch.no_grad():
                outputs = blip2_model(
                    **inputs, use_image_text_matching_head=True
                )
                logits = outputs.logits_per_image
                scores.append(torch.softmax(logits, dim=1)[:, 1].float().cpu())
        scores = torch.cat(scores, dim=0)
        order = torch.argsort(scores, descending=True)
        order_list = order.tolist()
        reranked.append([cand[idx] for idx in order_list])
        reranked_scores.append(scores[order].tolist())
    return reranked, reranked_scores


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
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    t0 = time.time()
    image_feats = encode_images(
        model, preprocess, images, IMAGE_BATCH, device).cpu()
    text_feats = encode_texts(
        model, tokenizer, captions, TEXT_BATCH, device).cpu()
    print(f"CLIP encoded in {time.time() - t0:.1f}s")

    t2i_rank_np, t2i_scores_np = topk_faiss(
        text_feats, image_feats, RETRIEVE_K, return_scores=True
    )
    t2i_rank = t2i_rank_np.tolist()
    t2i_scores = t2i_scores_np.tolist()

    clip_metrics = evaluate_metrics(
        caption_ids, image_ids, t2i_rank, t2i_scores, METRIC_KS
    )
    print("CLIP Text-to-Image metrics (ranx):")
    for k in METRIC_KS:
        r = clip_metrics[f"recall@{k}"]
        p = clip_metrics[f"precision@{k}"]
        n = clip_metrics[f"ndcg@{k}"]
        print(f"R@{k}: {r:.2f}  P@{k}: {p:.2f}  nDCG@{k}: {n:.2f}")
    print(
        f"MAP@{max(METRIC_KS)}: {clip_metrics[f'map@{max(METRIC_KS)}']:.2f}")

    blip2_dtype = torch.float16 if device == "cuda" else torch.float32
    blip2_model = Blip2ForImageTextRetrieval.from_pretrained(
        BLIP2_MODEL_ID, torch_dtype=blip2_dtype
    )
    blip2_processor = AutoProcessor.from_pretrained(BLIP2_MODEL_ID)
    blip2_model.to(device)
    blip2_model.eval()

    t2i_rank_blip2, t2i_scores_blip2 = blip2_rerank(
        blip2_model,
        blip2_processor,
        captions,
        images,
        t2i_rank,
        BLIP2_IMAGE_BATCH,
        device,
        blip2_dtype,
    )
    blip2_metrics = evaluate_metrics(
        caption_ids, image_ids, t2i_rank_blip2, t2i_scores_blip2, METRIC_KS
    )
    print("BLIP2-reranked Text-to-Image metrics (ranx):")
    for k in METRIC_KS:
        r = blip2_metrics[f"recall@{k}"]
        p = blip2_metrics[f"precision@{k}"]
        n = blip2_metrics[f"ndcg@{k}"]
        print(f"R@{k}: {r:.2f}  P@{k}: {p:.2f}  nDCG@{k}: {n:.2f}")
    print(
        f"MAP@{max(METRIC_KS)}: {blip2_metrics[f'map@{max(METRIC_KS)}']:.2f}")


if __name__ == "__main__":
    main()
