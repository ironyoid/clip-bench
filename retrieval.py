import numpy as np
import faiss


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


def build_faiss_index(feats):
    tf = feats.detach().cpu().numpy().astype("float32", copy=False)
    tf = np.ascontiguousarray(tf)
    index = faiss.IndexFlatIP(tf.shape[1])
    index.add(tf)
    return index


def query_faiss_index(index, query_feats, k, return_scores=False):
    qf = query_feats.detach().cpu().numpy().astype("float32", copy=False)
    qf = np.ascontiguousarray(qf)
    k = min(k, index.ntotal)
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


def rrf_fuse(rank_lists, k, topk):
    scores = {}
    for ranks in rank_lists:
        for r, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked = ranked[:topk]
    return [idx for idx, _ in ranked], [score for _, score in ranked]
