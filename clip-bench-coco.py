import json
import os
import time
from PIL import Image
import torch
import clip
import eccv_caption
import cv2
import faiss
import numpy as np

DATASET_PATH = "dataset/coco2014"
ANN_PATH = f"{DATASET_PATH}/annotations/karpathy_test.json"
DATA_DIR = f"{DATASET_PATH}/eccv/data"
MODEL_NAME = "ViT-B/32"
IMAGE_BATCH = 64
TEXT_BATCH = 256
RETRIEVE_K = 50


def load_karpathy_test(ann_path, coco_root):
    data = json.load(open(ann_path, "r", encoding="utf-8"))
    images = []
    image_ids = []
    captions = []
    caption_ids = []

    for img in data["images"]:
        if img["split"] != "test":
            continue
        img_path = os.path.join(coco_root, img["filepath"], img["filename"])
        images.append(img_path)
        image_ids.append(img["cocoid"])
        for sent in img["sentences"]:
            captions.append(sent["raw"].strip())
            caption_ids.append(sent["sentid"])

    return images, image_ids, captions, caption_ids


def encode_images(model, preprocess, image_paths, batch_size, device):
    feats = []
    total = len(image_paths)
    for i in range(0, total, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [preprocess(Image.open(p).convert("RGB"))
                  for p in batch_paths]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            batch_feats = model.encode_image(image_input)
        feats.append(batch_feats)
        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, total)
            print(f"Encoded images: {done}/{total}")
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)


def encode_texts(model, texts, batch_size, device):
    feats = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        tokens = clip.tokenize(batch).to(device)
        with torch.no_grad():
            batch_feats = model.encode_text(tokens)
        feats.append(batch_feats)
        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, total)
            print(f"Encoded captions: {done}/{total}")
    feats = torch.cat(feats, dim=0)
    return feats / feats.norm(dim=1, keepdim=True)


def show_top10(captions, caption_ids, images, image_ids, t2i_rank, gt_t2i, max_queries=20, size=224):
    valid_idx = [i for i, cid in enumerate(caption_ids) if cid in gt_t2i]
    print(f"ECCV-labeled queries: {len(valid_idx)} / {len(captions)}")
    for rank, i in enumerate(valid_idx[:max_queries]):
        print(f"\nQuery {rank}: {captions[i]}")
        gt = set(gt_t2i[caption_ids[i]])
        tiles = []
        for idx in t2i_rank[i][:10]:
            img = cv2.imread(images[idx])
            img = cv2.resize(img, (size, size))
            img_id = image_ids[idx]
            color = (0, 255, 0) if img_id in gt else (0, 0, 255)
            img = cv2.copyMakeBorder(
                img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
            tiles.append(img)
        grid = cv2.vconcat([cv2.hconcat(tiles[:5]), cv2.hconcat(tiles[5:])])
        cv2.imshow("Top10", grid)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()


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

    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    t0 = time.time()
    image_feats = encode_images(model, preprocess, images, IMAGE_BATCH, device)
    text_feats = encode_texts(model, captions, TEXT_BATCH, device)
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
        target_metrics=["eccv_map_at_r", "eccv_rprecision", "eccv_r1"],
        Ks=[1],
        verbose=True,
    )
    print("ECCV T2I metrics:")
    print(f"Map@R: {scores['eccv_map_at_r']['t2i']:.2f}")
    print(f"R-P: {scores['eccv_rprecision']['t2i']:.2f}")
    print(f"R@1: {scores['eccv_r1']['t2i']:.2f}")
    # show_top10(captions, caption_ids, images,
    #            image_ids, t2i_rank, metric.eccv_gts["t2i"])


if __name__ == "__main__":
    main()
