import sys
sys.path.insert(0, "reqs/Qwen3-VL-Embedding")

from tqdm import tqdm
from src.models.qwen3_vl_reranker import Qwen3VLReranker

QWEN_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
BATCH_SIZE = 4


def load(device):
    model = Qwen3VLReranker(model_name_or_path=QWEN_MODEL)
    return model,


def rerank(bundle, image_paths, captions, t2i_rank, batch_size, device):
    model = bundle[0]
    batch_size = batch_size or BATCH_SIZE

    reranked = []
    reranked_scores = []
    for i in tqdm(range(len(captions)), desc="Qwen rerank"):
        cand = t2i_rank[i]
        scores = []
        for s in range(0, len(cand), batch_size):
            batch_ids = cand[s:s + batch_size]
            docs = [{"image": image_paths[j]} for j in batch_ids]
            inputs = {"query": {"text": captions[i]}, "documents": docs}
            scores.extend(model.process(inputs))
        order = sorted(range(len(cand)), key=lambda k: scores[k], reverse=True)
        reranked.append([cand[k] for k in order])
        reranked_scores.append([scores[k] for k in order])

    return reranked, reranked_scores
