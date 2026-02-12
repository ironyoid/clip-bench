import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from dinov2.hub.dinotxt import (
    dinov2_vitl14_reg4_dinotxt_tet1280d20h24l,
    get_tokenizer,
)


def _build_preprocess():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load(device):
    model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(device)
    model.eval()
    tokenizer = get_tokenizer()
    preprocess = _build_preprocess()
    return model, preprocess, tokenizer


def encode_images(bundle, image_paths, batch_size, device):
    model, preprocess, _ = bundle
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="DINO images"):
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
    for i in tqdm(range(0, len(texts), batch_size), desc="DINO texts"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer.tokenize(batch).to(device)
        with torch.no_grad():
            batch_feats = model.encode_text(tokens)
        feats.append(batch_feats)
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)
