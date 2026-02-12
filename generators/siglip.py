import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL_NAME = "ViT-SO400M-14-SigLIP2-378"
PRETRAINED = "webli"


def _get_hf_tokenizer(model_name):
    cfg = open_clip.get_model_config(model_name)
    text_cfg = cfg.get("text_cfg", {}) if cfg else {}
    tok_name = text_cfg.get("hf_tokenizer_name") or text_cfg.get("hf_model_name")
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    return tokenizer, text_cfg.get("context_length")


def load(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device)
    model.eval()
    tokenizer, ctx_len = _get_hf_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer, ctx_len


def encode_images(bundle, image_paths, batch_size, device):
    model, preprocess = bundle[0], bundle[1]
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="SigLIP images"):
        batch_paths = image_paths[i:i + batch_size]
        images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            batch_feats = model.encode_image(image_input)
        feats.append(batch_feats)
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)


def encode_texts(bundle, texts, batch_size, device):
    model, _, tokenizer, ctx_len = bundle
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="SigLIP texts"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch, return_tensors="pt", padding="max_length",
            truncation=True, max_length=ctx_len,
        )
        input_ids = tokens["input_ids"].to(device)
        with torch.no_grad():
            batch_feats = model.encode_text(input_ids)
        feats.append(batch_feats)
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)
