from open_clip.factory import load_checkpoint as oc_load_checkpoint
from open_clip import create_model_and_transforms, get_tokenizer
import open_clip.factory as _oc_factory
import json
import os
import sys
import time

from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

from ranx import Qrels, Run, evaluate
from gen_params import METRIC_KS, TEXT_BATCH
from visualize import visualize_topk

sys.path.insert(0, 'reqs/ELIP/ELIP-C/src')


_oc_factory._MODEL_CONFIGS['ViT-B-16'] = {
    "embed_dim": 512,
    "init_logit_scale": 4.6052,
    "vision_cfg": {
        "layers": 12,
        "width": 768,
        "patch_size": 16,
        "image_size": 224,
        "vpt_cfg": {
            "type": "vpt",
            "vpt_type": "text_guided",
            "project": -1,
            "num_tokens": 2,
            "dropout": 0.0,
            "num_mapped_vpt": 10,
            "deep": True,
        },
    },
    "text_cfg": {
        "context_length": 77,
        "vocab_size": 49408,
        "width": 512,
        "heads": 8,
        "layers": 12,
        "text_tune_type": 0,
    },
}


ELIPC_IMAGE_BATCH = 64
CHECKPOINT_PATH = "reqs/12.15_v2_2024_12_15-07_14_55-model_ViT-B-16-lr_0.001-b_20-j_8-p_amp-epoch_1.pt"


def elipc_rerank(model, preprocess, tokenizer, captions, image_paths, t2i_rank, batch_size, device):
    processed_images = []
    total = len(image_paths)
    batch_start = time.time()
    for i in tqdm(range(total), desc="ELIP-C preprocess"):
        image = preprocess(Image.open(image_paths[i]).convert("RGB"))
        processed_images.append(image)
    batch_time = time.time() - batch_start
    print(f"ELIP-C preprocess: {total} in {batch_time:.3f}s")

    all_text_embeds = []
    batch_start = time.time()
    for i in tqdm(range(0, len(captions), TEXT_BATCH), desc="ELIP-C text encode"):
        batch_captions = captions[i:i + TEXT_BATCH]
        text_tokens = tokenizer(batch_captions).to(device)
        with torch.no_grad():
            text_embed = model.encode_text(text_tokens, normalize=True)
        all_text_embeds.append(text_embed.cpu())
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    batch_time = time.time() - batch_start
    print(f"ELIP-C text encode: {len(captions)} in {batch_time:.3f}s")

    reranked = []
    reranked_scores = []
    total_caps = len(captions)
    batch_start = time.time()
    for i in tqdm(range(total_caps), desc="ELIP-C rerank"):
        cand = t2i_rank[i]
        text_embed_i = all_text_embeds[i].to(device)
        scores = []
        for j in range(0, len(cand), batch_size):
            chunk = cand[j:j + batch_size]
            chunk_size = len(chunk)
            images = torch.stack([processed_images[k]
                                 for k in chunk]).to(device)
            text_embed_batch = text_embed_i.unsqueeze(0).expand(chunk_size, -1)
            with torch.no_grad():
                image_features = model.encode_image(
                    images,
                    text_embed=text_embed_batch,
                    normalize=True,
                )
            chunk_scores = (text_embed_batch * image_features).sum(dim=-1)
            scores.append(chunk_scores.float().cpu())
        scores = torch.cat(scores, dim=0)
        order = torch.argsort(scores, descending=True)
        order_list = order.tolist()
        reranked.append([cand[idx] for idx in order_list])
        reranked_scores.append(scores[order].tolist())

    batch_time = time.time() - batch_start
    print(f"ELIP-C rerank: {total_caps} in {batch_time:.3f}s")
    avg_query_time = batch_time / max(1, total_caps)
    print(f"Avg ELIP-C query time: {avg_query_time:.4f}s")
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

    model, _, preprocess_val = create_model_and_transforms(
        'ViT-B-16',
        pretrained='datacomp_xl_s13b_b90k',
        precision='fp32',
        device=device,
        output_dict=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
    elif 'model' in checkpoint:
        sd = checkpoint['model']
    else:
        sd = checkpoint
    if next(iter(sd.keys())).startswith('module.'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("ELIP-C model loaded.")

    tokenizer = get_tokenizer('ViT-B-16')

    t2i_rank_elipc, t2i_scores_elipc = elipc_rerank(
        model,
        preprocess_val,
        tokenizer,
        captions,
        images,
        t2i_rank,
        ELIPC_IMAGE_BATCH,
        device,
    )
    elipc_metrics = evaluate_metrics(
        caption_ids, image_ids, t2i_rank_elipc, t2i_scores_elipc, METRIC_KS
    )
    print("ELIP-C reranked Text-to-Image metrics (ranx):")
    for k in METRIC_KS:
        r = elipc_metrics[f"recall@{k}"]
        p = elipc_metrics[f"precision@{k}"]
        n = elipc_metrics[f"ndcg@{k}"]
        print(f"R@{k}: {r:.2f}  P@{k}: {p:.2f}  nDCG@{k}: {n:.2f}")
    print(
        f"MAP@{max(METRIC_KS)}: {elipc_metrics[f'map@{max(METRIC_KS)}']:.2f}")

    visualize_topk(captions, images, t2i_rank_elipc,
                   caption_ids, image_ids, model_name="elip-c")


if __name__ == "__main__":
    main()
