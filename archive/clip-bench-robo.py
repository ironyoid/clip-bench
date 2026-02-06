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
from omegaconf import OmegaConf
from lavis.models import load_preprocess
from lavis.models.albef_models.albef_retrieval import AlbefRetrieval
from parsers import load_robo_dataset


DATASET_PATH = "dataset/robotics_kitchen_dataset_v3"
ANN_PATH = f"{DATASET_PATH}/annotations/ground_truth/robotics_kitchen.json"
MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"
IMAGE_BATCH = 64
TEXT_BATCH = 256
RETRIEVE_K = 10
METRIC_KS = (1, 5, 10)
ALBEF_IMAGE_BATCH = 64
ALBEF_CFG_PATH = "configs/albef_retrieval_base.yaml"


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


def albef_rerank(albef_model, albef_preprocess, captions, image_paths, t2i_rank, batch_size, device):
    image_feats = []
    total = len(image_paths)
    batch_start = time.time()
    for i in tqdm(range(0, total, batch_size), desc="ALBEF image feats"):
        batch_paths = image_paths[i:i + batch_size]
        images = [albef_preprocess(Image.open(p).convert("RGB"))
                  for p in batch_paths]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            batch_feats = albef_model.visual_encoder.forward_features(
                image_input)
        image_feats.append(batch_feats.cpu())
    image_feats = torch.cat(image_feats, dim=0)
    batch_time = time.time() - batch_start

    print(f"ALBEF image feats: {total} in {batch_time:.3f}s")

    reranked = []
    reranked_scores = []
    total_caps = len(captions)
    batch_start = time.time()
    for i in tqdm(range(total_caps), desc="ALBEF rerank"):
        cand = t2i_rank[i]
        img_feat = image_feats[cand].to(device)
        encoder_att = torch.ones(
            img_feat.size()[:-1], dtype=torch.long).to(device)
        text_input = albef_model.tokenizer(
            [captions[i]] * len(cand),
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = albef_model.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=img_feat,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            scores = albef_model.itm_head(
                output.last_hidden_state[:, 0, :])[:, 1]
        order = torch.argsort(scores, descending=True)
        order_list = order.tolist()
        reranked.append([cand[j] for j in order_list])
        reranked_scores.append(scores[order].tolist())

    batch_time = time.time() - batch_start
    print(f"ALBEF rerank: {total_caps} in {batch_time:.3f}s")
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


def visualize_results(captions, images, caption_ids, image_ids, t2i_rank, t2i_rank_albef, tile_size=160):
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
        albef_top = t2i_rank_albef[i][:10]

        clip_paths = [images[j] for j in clip_top]
        albef_paths = [images[j] for j in albef_top]
        caption_obj_id = caption_ids[i]
        clip_highlight = [image_ids[j] == caption_obj_id for j in clip_top]
        albef_highlight = [image_ids[j] == caption_obj_id for j in albef_top]

        left = make_grid(
            clip_paths, clip_highlight, tile_size=tile_size, cols=cols, rows=rows
        )
        right = make_grid(
            albef_paths, albef_highlight, tile_size=tile_size, cols=cols, rows=rows
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
    print(f"Encoded in {time.time() - t0:.1f}s")

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
    print(f"MAP@{max(METRIC_KS)}: {clip_metrics[f'map@{max(METRIC_KS)}']:.2f}")

    albef_cfg = OmegaConf.load(ALBEF_CFG_PATH)
    albef_model = AlbefRetrieval.from_config(albef_cfg.model)
    albef_model.eval()
    albef_model = albef_model.to(device)
    albef_vis, _ = load_preprocess(albef_cfg.preprocess)

    t2i_rank_albef, t2i_scores_albef = albef_rerank(
        albef_model, albef_vis["eval"], captions, images, t2i_rank, ALBEF_IMAGE_BATCH, device
    )
    albef_metrics = evaluate_metrics(
        caption_ids, image_ids, t2i_rank_albef, t2i_scores_albef, METRIC_KS
    )
    print("ALBEF-reranked Text-to-Image metrics (ranx):")
    for k in METRIC_KS:
        r = albef_metrics[f"recall@{k}"]
        p = albef_metrics[f"precision@{k}"]
        n = albef_metrics[f"ndcg@{k}"]
        print(f"R@{k}: {r:.2f}  P@{k}: {p:.2f}  nDCG@{k}: {n:.2f}")
    print(
        f"MAP@{max(METRIC_KS)}: {albef_metrics[f'map@{max(METRIC_KS)}']:.2f}")

    visualize_results(captions, images, caption_ids,
                      image_ids, t2i_rank, t2i_rank_albef)


if __name__ == "__main__":
    main()
