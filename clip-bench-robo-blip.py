import json
import os
import pickle
import time
import gc

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import open_clip
import faiss

from ranx import Qrels, Run, evaluate
from lavis.models import load_model_and_preprocess


DATASET_PATH = "dataset/robotics_kitchen_dataset_v3"
ANN_PATH = f"{DATASET_PATH}/annotations/ground_truth/robotics_kitchen.json"
PREPHRASE_PATH = f"{DATASET_PATH}/objects_caption_prephrases.json"
MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"
IMAGE_BATCH = 64
TEXT_BATCH = 256
RETRIEVE_K = 10
METRIC_KS = (1, 5, 10)
RRF_K = 60
BLIP2_IMAGE_BATCH = 2


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


def load_robo_dataset(ann_path, coco_root, prephrase_path=None):
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
            out_path = os.path.join(
                masked_objects_dir, f"frame{frame_num}_obj{object_id}.jpg")
            if not os.path.exists(out_path):
                frame_path = os.path.join(
                    coco_root, "frames", "robotics_kitchen", f"{frame_num - 1}.jpg"
                )
                mask_path = os.path.join(
                    coco_root, "mask_cache", mask_rel_path)

                image = np.array(Image.open(frame_path).convert("RGB"))
                rle = pickle.load(open(mask_path, "rb"))
                mask = decode_uncompressed_rle(rle).astype(bool)

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
    return images, image_ids, captions, caption_ids, variants


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


def blip2_rerank(blip2_model, blip2_vis, blip2_txt, captions, image_paths, t2i_rank, batch_size, device):
    processed_images = []
    total = len(image_paths)
    batch_start = time.time()
    for i in tqdm(range(total), desc="BLIP2 preprocess"):
        image = blip2_vis(Image.open(image_paths[i]).convert("RGB"))
        processed_images.append(image)
    batch_time = time.time() - batch_start
    print(f"BLIP2 preprocess: {total} in {batch_time:.3f}s")

    reranked = []
    reranked_scores = []
    query_times = []
    total_caps = len(captions)
    batch_start = time.time()
    for i in tqdm(range(total_caps), desc="BLIP2 rerank"):
        q_start = time.time()
        cand = t2i_rank[i]
        text_input = blip2_txt(captions[i])
        scores = []
        for j in range(0, len(cand), batch_size):
            chunk = cand[j:j + batch_size]
            images = torch.stack([processed_images[k]
                                 for k in chunk]).to(device)
            text_batch = [text_input] * len(chunk)
            with torch.no_grad():
                logits = blip2_model(
                    {"image": images, "text_input": text_batch}, match_head="itm"
                )
            scores.append(logits[:, 1].float().cpu())
        scores = torch.cat(scores, dim=0)
        order = torch.argsort(scores, descending=True)
        order_list = order.tolist()
        reranked.append([cand[idx] for idx in order_list])
        reranked_scores.append(scores[order].tolist())
        query_times.append(time.time() - q_start)

    batch_time = time.time() - batch_start
    print(f"BLIP2 rerank: {total_caps} in {batch_time:.3f}s")
    avg_query_time = sum(query_times) / max(1, len(query_times))
    return reranked, reranked_scores, avg_query_time


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
    metrics.append("map@10")
    return evaluate(qrels, run, metrics)


def make_grid(image_paths, highlight_flags, tile_size=160, cols=5, rows=2):
    grid = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
    for i, path in enumerate(image_paths[:rows * cols]):
        img = Image.open(path).convert("RGB")
        img = img.resize((tile_size, tile_size), Image.BICUBIC)
        img = np.asarray(img)[:, :, ::-1].copy()
        y = (i // cols) * tile_size
        x = (i % cols) * tile_size
        grid[y:y + tile_size, x:x + tile_size] = img
        if highlight_flags and highlight_flags[i]:
            cv2.rectangle(
                grid,
                (x, y),
                (x + tile_size - 1, y + tile_size - 1),
                (0, 255, 0),
                3,
            )
    return grid


def wrap_text(text, max_width, font, font_scale, thickness):
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        test = f"{current} {word}"
        (w, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if w <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def visualize_results(captions, images, caption_ids, image_ids, t2i_rank, t2i_rank_blip2, tile_size=160):
    window_name = "retrieval"
    total = len(captions)
    cols = 5
    rows = 2
    gap = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_padding = 12
    header_bg = (20, 20, 20)
    text_color = (240, 240, 240)
    for i, caption in enumerate(captions):
        clip_top = t2i_rank[i][:10]
        blip2_top = t2i_rank_blip2[i][:10]

        clip_paths = [images[j] for j in clip_top]
        blip2_paths = [images[j] for j in blip2_top]
        caption_obj_id = caption_ids[i]
        clip_highlight = [image_ids[j] == caption_obj_id for j in clip_top]
        blip2_highlight = [image_ids[j] == caption_obj_id for j in blip2_top]

        left = make_grid(
            clip_paths, clip_highlight, tile_size=tile_size, cols=cols, rows=rows
        )
        right = make_grid(
            blip2_paths, blip2_highlight, tile_size=tile_size, cols=cols, rows=rows
        )
        grid_h = left.shape[0]
        grid_w = left.shape[1] + right.shape[1] + gap
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        canvas[:, :left.shape[1]] = left
        canvas[:, left.shape[1] + gap:] = right

        lines = wrap_text(
            f"Query: {caption}",
            grid_w - 2 * text_padding,
            font,
            font_scale,
            thickness,
        )
        (_, text_h), _ = cv2.getTextSize("Ag", font, font_scale, thickness)
        line_h = text_h + 8
        header_h = min(grid_h, text_padding * 2 + line_h * len(lines))
        cv2.rectangle(canvas, (0, 0), (grid_w - 1, header_h), header_bg, -1)

        y = text_padding + line_h - 6
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (text_padding, y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
            y += line_h
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    avg_clip_query_time = sum(clip_query_times) / max(1, len(clip_query_times))
    print(f"Avg CLIP query time: {avg_clip_query_time:.4f}s")

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

    blip2_model, blip2_vis, blip2_txt = load_model_and_preprocess(
        name="blip2_image_text_matching",
        model_type="pretrain",
        is_eval=True,
        device=device,
    )
    blip2_model.eval()

    t2i_rank_blip2, t2i_scores_blip2, avg_blip_query_time = blip2_rerank(
        blip2_model,
        blip2_vis["eval"],
        blip2_txt["eval"],
        captions,
        images,
        t2i_rank,
        BLIP2_IMAGE_BATCH,
        device,
    )
    print(f"Avg BLIP2 query time: {avg_blip_query_time:.4f}s")
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

    visualize_results(captions, images, caption_ids,
                      image_ids, t2i_rank, t2i_rank_blip2)


if __name__ == "__main__":
    main()
