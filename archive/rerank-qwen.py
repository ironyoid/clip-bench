import json
import sys
import time

from tqdm import tqdm

from ranx import Qrels, Run, evaluate
from gen_params import METRIC_KS
from visualize import visualize_topk

sys.path.insert(0, "reqs/Qwen3-VL-Embedding")
from src.models.qwen3_vl_reranker import Qwen3VLReranker


QWEN_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
QWEN_BATCH = 4


def qwen_rerank(qwen_model, captions, image_paths, t2i_rank, batch_size):
    reranked = []
    reranked_scores = []
    t0 = time.time()
    for i in tqdm(range(len(captions)), desc="Qwen rerank"):
        caption = captions[i]
        cand_ids = t2i_rank[i]
        scores = []
        for s in range(0, len(cand_ids), batch_size):
            batch_ids = cand_ids[s:s + batch_size]
            docs = [{"image": image_paths[j]} for j in batch_ids]
            inputs = {"query": {"text": caption}, "documents": docs}
            scores.extend(qwen_model.process(inputs))
        order = sorted(range(len(cand_ids)), key=lambda k: scores[k], reverse=True)
        reranked.append([cand_ids[k] for k in order])
        reranked_scores.append([scores[k] for k in order])
    total_time = time.time() - t0
    avg_query_time = total_time / max(1, len(captions))
    print(f"Avg Qwen query time: {avg_query_time:.4f}s")
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

    qwen_model = Qwen3VLReranker(model_name_or_path=QWEN_MODEL)

    t2i_rank_qwen, t2i_scores_qwen = qwen_rerank(
        qwen_model, captions, images, t2i_rank, QWEN_BATCH
    )
    qwen_metrics = evaluate_metrics(
        caption_ids, image_ids, t2i_rank_qwen, t2i_scores_qwen, METRIC_KS
    )
    print("Qwen-reranked Text-to-Image metrics (ranx):")
    for k in METRIC_KS:
        r = qwen_metrics[f"recall@{k}"]
        p = qwen_metrics[f"precision@{k}"]
        n = qwen_metrics[f"ndcg@{k}"]
        print(f"R@{k}: {r:.2f}  P@{k}: {p:.2f}  nDCG@{k}: {n:.2f}")
    print(
        f"MAP@{max(METRIC_KS)}: {qwen_metrics[f'map@{max(METRIC_KS)}']:.2f}")

    visualize_topk(captions, images, t2i_rank_qwen, caption_ids, image_ids, model_name="qwen")


if __name__ == "__main__":
    main()
