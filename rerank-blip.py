import json
import sys
import time

from PIL import Image
from tqdm import tqdm
import torch

from ranx import Qrels, Run, evaluate
from lavis.models import load_model_and_preprocess


BLIP2_IMAGE_BATCH = 2
METRIC_KS = (1, 5, 40)


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
    total_caps = len(captions)
    batch_start = time.time()
    for i in tqdm(range(total_caps), desc="BLIP2 rerank"):
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

    batch_time = time.time() - batch_start
    print(f"BLIP2 rerank: {total_caps} in {batch_time:.3f}s")
    avg_query_time = batch_time / max(1, total_caps)
    print(f"Avg BLIP2 query time: {avg_query_time:.4f}s")
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
    path = sys.argv[1]
    data = json.load(open(path, "r", encoding="utf-8"))
    images = data["images"]
    image_ids = data["image_ids"]
    captions = data["captions"]
    caption_ids = data["caption_ids"]
    t2i_rank = data["t2i_rank"]

    device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    blip2_model, blip2_vis, blip2_txt = load_model_and_preprocess(
        name="blip2_image_text_matching",
        model_type="pretrain",
        is_eval=True,
        device=device,
    )
    blip2_model.eval()

    t2i_rank_blip2, t2i_scores_blip2 = blip2_rerank(
        blip2_model,
        blip2_vis["eval"],
        blip2_txt["eval"],
        captions,
        images,
        t2i_rank,
        BLIP2_IMAGE_BATCH,
        device,
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
