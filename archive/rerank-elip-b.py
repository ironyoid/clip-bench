from visualize import visualize_topk
from gen_params import METRIC_KS, TEXT_BATCH
from ranx import Qrels, Run, evaluate
import torch.nn.functional as F
import torch
from tqdm import tqdm
from PIL import Image
import time
import json
from lavis.models import load_model_and_preprocess
import sys
sys.path.insert(0, 'reqs/ELIP/ELIP-B')


ELIP_IMAGE_BATCH = 8
CHECKPOINT_PATH = "reqs/full_model_iccv_v27-20241229044-checkpoint_0.pth"


def elip_rerank(elip_model, elip_vis, captions, image_paths, t2i_rank, batch_size, device):
    processed_images = []
    total = len(image_paths)
    batch_start = time.time()
    for i in tqdm(range(total), desc="ELIP preprocess"):
        image = elip_vis(Image.open(image_paths[i]).convert("RGB"))
        processed_images.append(image)
    batch_time = time.time() - batch_start
    print(f"ELIP preprocess: {total} in {batch_time:.3f}s")

    tokenizer = elip_model.tokenizer
    all_text_ids = []
    all_text_atts = []
    all_text_embeds = []
    batch_start = time.time()
    for i in tqdm(range(0, len(captions), TEXT_BATCH), desc="ELIP text encode"):
        batch_captions = captions[i:i + TEXT_BATCH]
        text_tokens = tokenizer(
            batch_captions,
            padding="max_length",
            truncation=True,
            max_length=elip_model.max_txt_len,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_feat = elip_model.forward_text(text_tokens)
            text_embed = F.normalize(elip_model.text_proj(text_feat))
        all_text_ids.append(text_tokens.input_ids.cpu())
        all_text_atts.append(text_tokens.attention_mask.cpu())
        all_text_embeds.append(text_embed.cpu())
    all_text_ids = torch.cat(all_text_ids, dim=0)
    all_text_atts = torch.cat(all_text_atts, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    batch_time = time.time() - batch_start
    print(f"ELIP text encode: {len(captions)} in {batch_time:.3f}s")

    reranked = []
    reranked_scores = []
    total_caps = len(captions)
    batch_start = time.time()
    for i in tqdm(range(total_caps), desc="ELIP rerank"):
        cand = t2i_rank[i]
        text_ids_i = all_text_ids[i]
        text_atts_i = all_text_atts[i]
        text_embeds_i = all_text_embeds[i]
        scores = []
        for j in range(0, len(cand), batch_size):
            chunk = cand[j:j + batch_size]
            chunk_size = len(chunk)
            images = torch.stack([processed_images[k]
                                 for k in chunk]).to(device)
            text_ids_batch = text_ids_i.unsqueeze(
                0).expand(chunk_size, -1).to(device)
            text_atts_batch = text_atts_i.unsqueeze(
                0).expand(chunk_size, -1).to(device)
            text_embeds_batch = text_embeds_i.unsqueeze(
                0).expand(chunk_size, -1).to(device)
            with torch.no_grad():
                itm_scores = elip_model.compute_itm_tgvpt(
                    image_inputs=images,
                    text_ids=text_ids_batch,
                    text_atts=text_atts_batch,
                    text_embeds=text_embeds_batch,
                )
            scores.append(itm_scores.float().cpu())
        scores = torch.cat(scores, dim=0)
        order = torch.argsort(scores, descending=True)
        order_list = order.tolist()
        reranked.append([cand[idx] for idx in order_list])
        reranked_scores.append(scores[order].tolist())

    batch_time = time.time() - batch_start
    print(f"ELIP rerank: {total_caps} in {batch_time:.3f}s")
    avg_query_time = batch_time / max(1, total_caps)
    print(f"Avg ELIP query time: {avg_query_time:.4f}s")
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

    elip_model, elip_vis, _ = load_model_and_preprocess(
        name="blip2_less_noisy_also_itm_head_multi_prompt3",
        model_type="coco",
        is_eval=True,
        device=device,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    elip_model.load_state_dict(checkpoint["model"], strict=False)

    elip_model.eval()

    t2i_rank_elip, t2i_scores_elip = elip_rerank(
        elip_model,
        elip_vis["eval"],
        captions,
        images,
        t2i_rank,
        ELIP_IMAGE_BATCH,
        device,
    )
    elip_metrics = evaluate_metrics(
        caption_ids, image_ids, t2i_rank_elip, t2i_scores_elip, METRIC_KS
    )
    print("ELIP-reranked Text-to-Image metrics (ranx):")
    for k in METRIC_KS:
        r = elip_metrics[f"recall@{k}"]
        p = elip_metrics[f"precision@{k}"]
        n = elip_metrics[f"ndcg@{k}"]
        print(f"R@{k}: {r:.2f}  P@{k}: {p:.2f}  nDCG@{k}: {n:.2f}")
    print(
        f"MAP@{max(METRIC_KS)}: {elip_metrics[f'map@{max(METRIC_KS)}']:.2f}")

    visualize_topk(captions, images, t2i_rank_elip,
                   caption_ids, image_ids, model_name="elip-b")


if __name__ == "__main__":
    main()
