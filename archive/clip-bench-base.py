import numpy as np
import faiss
import eccv_caption
import open_clip
import torch
from PIL import Image
import time
from tqdm import tqdm
from parsers import load_karpathy_test


DATASET_PATH = "../dataset/coco2014"
ANN_PATH = f"{DATASET_PATH}/annotations/karpathy_test.json"
MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"
IMAGE_BATCH = 64
TEXT_BATCH = 256
RETRIEVE_K = 100


def encode_images(model, preprocess, image_paths, batch_size, device):
    feats = []
    total = len(image_paths)
    for i in tqdm(range(0, total, batch_size), desc="Encoded images"):
        batch_paths = image_paths[i:i + batch_size]
        images = [preprocess(Image.open(p).convert("RGB"))
                  for p in batch_paths]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            batch_feats = model.encode_image(image_input)
        feats.append(batch_feats)
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)


def encode_texts(model, tokenizer, texts, batch_size, device):
    feats = []
    total = len(texts)
    for i in tqdm(range(0, total, batch_size), desc="Encoded captions"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            batch_feats = model.encode_text(tokens)
        feats.append(batch_feats)
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)


def topk_faiss(text_feats, img_feats, k):
    txf = text_feats.detach().cpu().numpy().astype("float32", copy=False)
    imf = img_feats.detach().cpu().numpy().astype("float32", copy=False)
    txf = np.ascontiguousarray(txf)
    imf = np.ascontiguousarray(imf)
    k = min(k, imf.shape[0])
    index = faiss.IndexFlatIP(imf.shape[1])
    index.add(imf)
    _, indices = index.search(txf, k)
    return indices


def main():
    device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    images, image_ids, captions, caption_ids = load_karpathy_test(
        ANN_PATH, DATASET_PATH)
    print(f"Test images: {len(images)}")
    print(f"Test captions: {len(captions)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    t0 = time.time()
    image_feats = encode_images(
        model, preprocess, images, IMAGE_BATCH, device).cpu()

    text_feats = encode_texts(
        model, tokenizer, captions, TEXT_BATCH, device).cpu()
    print(f"Encoded in {time.time() - t0:.1f}s")

    i2t_rank = topk_faiss(image_feats, text_feats, RETRIEVE_K).tolist()
    t2i_rank = topk_faiss(text_feats, image_feats, RETRIEVE_K).tolist()

    i2t = {
        image_ids[i]: [caption_ids[j] for j in i2t_rank[i]]
        for i in range(len(image_ids))
    }
    t2i = {
        caption_ids[i]: [image_ids[j] for j in t2i_rank[i]]
        for i in range(len(caption_ids))
    }

    metric = eccv_caption.Metrics()
    scores = metric.compute_all_metrics(
        i2t_retrieved_items=i2t,
        t2i_retrieved_items=t2i,
        target_metrics=["coco_5k_recalls",
                        "eccv_map_at_r", "eccv_rprecision", "eccv_r1"],
        Ks=[1, 5, 10],
        verbose=True,
    )
    print("COCO 5K T2I recalls:")
    print(f"R@1: {scores['coco_5k_r1']['t2i']:.2f}")
    print(f"R@5: {scores['coco_5k_r5']['t2i']:.2f}")
    print(f"R@10: {scores['coco_5k_r10']['t2i']:.2f}")
    print("ECCV T2I metrics:")
    print(f"Map@R: {scores['eccv_map_at_r']['t2i']:.2f}")
    print(f"R-P: {scores['eccv_rprecision']['t2i']:.2f}")
    print(f"R@1: {scores['eccv_r1']['t2i']:.2f}")


if __name__ == "__main__":
    main()
