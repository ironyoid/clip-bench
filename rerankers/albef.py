import torch
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from lavis.models import load_preprocess
from lavis.models.albef_models.albef_retrieval import AlbefRetrieval

BATCH_SIZE = 64
CFG_PATH = "configs/albef_retrieval_base.yaml"


def load(device):
    cfg = OmegaConf.load(CFG_PATH)
    model = AlbefRetrieval.from_config(cfg.model)
    model.eval()
    model = model.to(device)
    vis, _ = load_preprocess(cfg.preprocess)
    return model, vis["eval"], device


def rerank(bundle, image_paths, captions, t2i_rank, batch_size, device):
    model, vis_proc, _ = bundle
    batch_size = batch_size or BATCH_SIZE

    image_feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="ALBEF image feats"):
        batch_paths = image_paths[i:i + batch_size]
        images = [vis_proc(Image.open(p).convert("RGB")) for p in batch_paths]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            batch_feats = model.visual_encoder.forward_features(image_input)
        image_feats.append(batch_feats.cpu())
    image_feats = torch.cat(image_feats, dim=0)

    reranked = []
    reranked_scores = []
    for i in tqdm(range(len(captions)), desc="ALBEF rerank"):
        cand = t2i_rank[i]
        img_feat = image_feats[cand].to(device)
        encoder_att = torch.ones(img_feat.size()[:-1], dtype=torch.long).to(device)
        text_input = model.tokenizer(
            [captions[i]] * len(cand),
            padding="max_length", truncation=True,
            max_length=35, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=img_feat,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            scores = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        order = torch.argsort(scores, descending=True)
        order_list = order.tolist()
        reranked.append([cand[j] for j in order_list])
        reranked_scores.append(scores[order].tolist())

    return reranked, reranked_scores
