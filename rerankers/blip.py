import torch
from PIL import Image
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

BATCH_SIZE = 2


def load(device):
    model, vis, txt = load_model_and_preprocess(
        name="blip2_image_text_matching",
        model_type="pretrain",
        is_eval=True,
        device=device,
    )
    model.eval()
    return model, vis["eval"], txt["eval"], device


def rerank(bundle, image_paths, captions, t2i_rank, batch_size, device):
    model, vis_proc, txt_proc, _ = bundle
    batch_size = batch_size or BATCH_SIZE

    processed = []
    for i in tqdm(range(len(image_paths)), desc="BLIP preprocess"):
        processed.append(vis_proc(Image.open(image_paths[i]).convert("RGB")))

    reranked = []
    reranked_scores = []
    for i in tqdm(range(len(captions)), desc="BLIP rerank"):
        cand = t2i_rank[i]
        text_input = txt_proc(captions[i])
        scores = []
        for j in range(0, len(cand), batch_size):
            chunk = cand[j:j + batch_size]
            images = torch.stack([processed[k] for k in chunk]).to(device)
            text_batch = [text_input] * len(chunk)
            with torch.no_grad():
                logits = model(
                    {"image": images, "text_input": text_batch}, match_head="itm")
            scores.append(logits[:, 1].float().cpu())
        scores = torch.cat(scores, dim=0)
        order = torch.argsort(scores, descending=True)
        order_list = order.tolist()
        reranked.append([cand[idx] for idx in order_list])
        reranked_scores.append(scores[order].tolist())

    return reranked, reranked_scores
