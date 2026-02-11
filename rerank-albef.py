import json
import sys
import time

from PIL import Image
from tqdm import tqdm
import torch

from ranx import Qrels, Run, evaluate
from omegaconf import OmegaConf
from lavis.models import load_preprocess
from lavis.models.albef_models.albef_retrieval import AlbefRetrieval
from gen_params import METRIC_KS
from visualize import visualize_topk


ALBEF_IMAGE_BATCH = 64
ALBEF_CFG_PATH = "configs/albef_retrieval_base.yaml"


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
    avg_query_time = batch_time / max(1, total_caps)
    print(f"Avg ALBEF query time: {avg_query_time:.4f}s")
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

    visualize_topk(captions, images, t2i_rank_albef, caption_ids, image_ids, model_name="albef")


if __name__ == "__main__":
    main()
