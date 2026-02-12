import sys
sys.path.insert(0, 'reqs/ELIP/ELIP-B')

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from gen_params import TEXT_BATCH

BATCH_SIZE = 8
CHECKPOINT_PATH = "reqs/full_model_iccv_v27-20241229044-checkpoint_0.pth"


def load(device):
    model, vis, _ = load_model_and_preprocess(
        name="blip2_less_noisy_also_itm_head_multi_prompt3",
        model_type="coco",
        is_eval=True,
        device=device,
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, vis["eval"], device


def rerank(bundle, image_paths, captions, t2i_rank, batch_size, device):
    model, vis_proc, _ = bundle
    batch_size = batch_size or BATCH_SIZE

    processed = []
    for i in tqdm(range(len(image_paths)), desc="ELIP-B preprocess"):
        processed.append(vis_proc(Image.open(image_paths[i]).convert("RGB")))

    tokenizer = model.tokenizer
    all_text_ids = []
    all_text_atts = []
    all_text_embeds = []
    for i in tqdm(range(0, len(captions), TEXT_BATCH), desc="ELIP-B text encode"):
        batch_captions = captions[i:i + TEXT_BATCH]
        text_tokens = tokenizer(
            batch_captions, padding="max_length", truncation=True,
            max_length=model.max_txt_len, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_feat = model.forward_text(text_tokens)
            text_embed = F.normalize(model.text_proj(text_feat))
        all_text_ids.append(text_tokens.input_ids.cpu())
        all_text_atts.append(text_tokens.attention_mask.cpu())
        all_text_embeds.append(text_embed.cpu())
    all_text_ids = torch.cat(all_text_ids, dim=0)
    all_text_atts = torch.cat(all_text_atts, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    reranked = []
    reranked_scores = []
    for i in tqdm(range(len(captions)), desc="ELIP-B rerank"):
        cand = t2i_rank[i]
        text_ids_i = all_text_ids[i]
        text_atts_i = all_text_atts[i]
        text_embeds_i = all_text_embeds[i]
        scores = []
        for j in range(0, len(cand), batch_size):
            chunk = cand[j:j + batch_size]
            chunk_size = len(chunk)
            images = torch.stack([processed[k] for k in chunk]).to(device)
            text_ids_batch = text_ids_i.unsqueeze(0).expand(chunk_size, -1).to(device)
            text_atts_batch = text_atts_i.unsqueeze(0).expand(chunk_size, -1).to(device)
            text_embeds_batch = text_embeds_i.unsqueeze(0).expand(chunk_size, -1).to(device)
            with torch.no_grad():
                itm_scores = model.compute_itm_tgvpt(
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

    return reranked, reranked_scores
