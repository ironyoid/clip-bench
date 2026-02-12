import torch
import open_clip
from PIL import Image
from tqdm import tqdm

MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"


def load(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer


def encode_images(bundle, image_paths, batch_size, device):
    model, preprocess, _ = bundle
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP images"):
        batch_paths = image_paths[i:i + batch_size]
        images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            batch_feats = model.encode_image(image_input)
        feats.append(batch_feats)
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)


def encode_texts(bundle, texts, batch_size, device):
    model, _, tokenizer = bundle
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="CLIP texts"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            batch_feats = model.encode_text(tokens)
        feats.append(batch_feats)
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)
