import sys
sys.path.insert(0, 'reqs/ELIP/ELIP-C/src')

import torch
from PIL import Image
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
import open_clip.factory as _oc_factory
from gen_params import TEXT_BATCH

_oc_factory._MODEL_CONFIGS['ViT-B-16'] = {
    "embed_dim": 512,
    "init_logit_scale": 4.6052,
    "vision_cfg": {
        "layers": 12, "width": 768, "patch_size": 16, "image_size": 224,
        "vpt_cfg": {
            "type": "vpt", "vpt_type": "text_guided", "project": -1,
            "num_tokens": 2, "dropout": 0.0, "num_mapped_vpt": 10, "deep": True,
        },
    },
    "text_cfg": {
        "context_length": 77, "vocab_size": 49408, "width": 512,
        "heads": 8, "layers": 12, "text_tune_type": 0,
    },
}

BATCH_SIZE = 64
CHECKPOINT_PATH = "reqs/12.15_v2_2024_12_15-07_14_55-model_ViT-B-16-lr_0.001-b_20-j_8-p_amp-epoch_1.pt"


def load(device):
    model, _, preprocess = create_model_and_transforms(
        'ViT-B-16', pretrained='datacomp_xl_s13b_b90k',
        precision='fp32', device=device, output_dict=True,
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    sd = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    if next(iter(sd.keys())).startswith('module.'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    tokenizer = get_tokenizer('ViT-B-16')
    return model, preprocess, tokenizer, device


def rerank(bundle, image_paths, captions, t2i_rank, batch_size, device):
    model, preprocess, tokenizer, _ = bundle
    batch_size = batch_size or BATCH_SIZE

    processed = []
    for i in tqdm(range(len(image_paths)), desc="ELIP-C preprocess"):
        processed.append(preprocess(Image.open(image_paths[i]).convert("RGB")))

    all_text_embeds = []
    for i in tqdm(range(0, len(captions), TEXT_BATCH), desc="ELIP-C text encode"):
        batch_captions = captions[i:i + TEXT_BATCH]
        text_tokens = tokenizer(batch_captions).to(device)
        with torch.no_grad():
            text_embed = model.encode_text(text_tokens, normalize=True)
        all_text_embeds.append(text_embed.cpu())
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    reranked = []
    reranked_scores = []
    for i in tqdm(range(len(captions)), desc="ELIP-C rerank"):
        cand = t2i_rank[i]
        text_embed_i = all_text_embeds[i].to(device)
        scores = []
        for j in range(0, len(cand), batch_size):
            chunk = cand[j:j + batch_size]
            chunk_size = len(chunk)
            images = torch.stack([processed[k] for k in chunk]).to(device)
            text_embed_batch = text_embed_i.unsqueeze(0).expand(chunk_size, -1)
            with torch.no_grad():
                image_features = model.encode_image(
                    images, text_embed=text_embed_batch, normalize=True)
            chunk_scores = (text_embed_batch * image_features).sum(dim=-1)
            scores.append(chunk_scores.float().cpu())
        scores = torch.cat(scores, dim=0)
        order = torch.argsort(scores, descending=True)
        order_list = order.tolist()
        reranked.append([cand[idx] for idx in order_list])
        reranked_scores.append(scores[order].tolist())

    return reranked, reranked_scores
