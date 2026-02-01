import numpy as np
import faiss
import eccv_caption
from omegaconf import OmegaConf
from lavis.models import load_preprocess
from lavis.models.blip_models.blip_image_text_matching import BlipITM
import clip
import torch
from PIL import Image
import json
import os
import time
import contextlib


DATASET_PATH = "dataset/coco2014"
ANN_PATH = f"{DATASET_PATH}/annotations/karpathy_test.json"
DATA_DIR = f"{DATASET_PATH}/eccv/data"
MODEL_NAME = "ViT-B/32"
IMAGE_BATCH = 64
TEXT_BATCH = 256
RETRIEVE_K = 50
BLIP_ITM_IMAGE_BATCH = 2
BLIP_ITM_CFG_PATH = "blip_itm_base_pretrain.yaml"


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
    import cv2
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


def rerank_t2i_with_albef(albef_model, albef_preprocess, captions, image_paths, t2i_rank, batch_size, device):
    image_feats = []
    total = len(image_paths)
    for i in range(0, total, batch_size):
        batch_start = time.time()
        batch_paths = image_paths[i:i + batch_size]
        images = [albef_preprocess(Image.open(p).convert("RGB"))
                  for p in batch_paths]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            batch_feats = albef_model.visual_encoder.forward_features(
                image_input)
        image_feats.append(batch_feats.cpu())
        done = min(i + batch_size, total)
        batch_time = time.time() - batch_start
        print(f"ALBEF image feats batch {done}/{total} in {batch_time:.3f}s")
    image_feats = torch.cat(image_feats, dim=0)

    reranked = []
    total_caps = len(captions)
    for i in range(total_caps):
        batch_start = time.time()
        cand = t2i_rank[i]
        img_feat = image_feats[cand].to(device)
        encoder_att = torch.ones(
            img_feat.size()[:-1], dtype=torch.long).to(device)
        text_input = albef_model.tokenizer(
            [captions[i]] * len(cand),
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = albef_model.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=img_feat,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            scores = albef_model.itm_head(
                output.last_hidden_state[:, 0, :])[:, 1]
        order = torch.argsort(scores, descending=True)
        reranked.append([cand[j] for j in order.tolist()])
        batch_time = time.time() - batch_start
        print(f"ALBEF rerank batch {i + 1}/{total_caps} in {batch_time:.3f}s")
    return reranked


def blip_rerank(blip_model, blip_preprocess, captions, image_paths, t2i_rank, batch_size, device):
    amp_ctx = (
        torch.autocast(device_type=device, dtype=torch.float16)
        if device in ("cuda", "mps")
        else contextlib.nullcontext()
    )
    image_feats = []
    total = len(image_paths)
    for i in range(0, total, batch_size):
        batch_start = time.time()
        batch_paths = image_paths[i:i + batch_size]
        images = [blip_preprocess(Image.open(p).convert("RGB"))
                  for p in batch_paths]
        image_input = torch.stack(images).to(device)
        with torch.no_grad(), amp_ctx:
            batch_feats = blip_model.visual_encoder.forward_features(
                image_input)
        image_feats.append(batch_feats.cpu())
        done = min(i + batch_size, total)
        batch_time = time.time() - batch_start
        print(f"BLIP image feats batch {done}/{total} in {batch_time:.3f}s")
    image_feats = torch.cat(image_feats, dim=0)

    reranked = []
    total_caps = len(captions)
    for i in range(total_caps):
        batch_start = time.time()
        cand = t2i_rank[i]
        img_feat = image_feats[cand].to(device)
        encoder_att = torch.ones(
            img_feat.size()[:-1], dtype=torch.long).to(device)
        text_input = blip_model.tokenizer(
            [captions[i]] * len(cand),
            padding="longest",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)
        encoder_input_ids = text_input.input_ids.clone()
        encoder_input_ids[:, 0] = blip_model.tokenizer.enc_token_id
        with torch.no_grad(), amp_ctx:
            output = blip_model.text_encoder(
                encoder_input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=img_feat,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            scores = blip_model.itm_head(
                output.last_hidden_state[:, 0, :])[:, 1]
        order = torch.argsort(scores, descending=True)
        reranked.append([cand[j] for j in order.tolist()])
        batch_time = time.time() - batch_start
        print(f"BLIP rerank batch {i + 1}/{total_caps} in {batch_time:.3f}s")
    return reranked


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

    blip_cfg = OmegaConf.load(BLIP_ITM_CFG_PATH)
    blip_model = BlipITM.from_config(blip_cfg.model)
    blip_model.eval()
    blip_model = blip_model.to(device)
    blip_vis, _ = load_preprocess(blip_cfg.preprocess)

    t2i_rank = blip_rerank(
        blip_model, blip_vis["eval"], captions, images, t2i_rank, BLIP_ITM_IMAGE_BATCH, device
    )

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


if __name__ == "__main__":
    main()
