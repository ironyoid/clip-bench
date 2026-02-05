import json
import os
import pickle
import time
import gc

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


DATASET_PATH = "dataset/robotics_kitchen_dataset_v3"
ANN_PATH = f"{DATASET_PATH}/annotations/ground_truth/robotics_kitchen.json"
MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"
IMAGE_BATCH = 64
TEXT_BATCH = 256
RETRIEVE_K = 100
METRIC_KS = (1, 5, 10, 50, 100)
ALBEF_IMAGE_BATCH = 32
ALBEF_CFG_PATH = "configs/albef_retrieval_base.yaml"


def decode_uncompressed_rle(rle):
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    value = 0
    for count in counts:
        if value == 1:
            flat[idx:idx + count] = 1
        idx += count
        value = 1 - value
    return flat.reshape((h, w), order="F")


def load_karpathy_test(ann_path, coco_root):
    data = json.load(open(ann_path, "r", encoding="utf-8"))
    object_caption_path = os.path.join(coco_root, "objects_caption.json")
    masked_objects_dir = os.path.join(coco_root, "masked_objects")
    object_captions = json.load(
        open(object_caption_path, "r", encoding="utf-8"))
    os.makedirs(masked_objects_dir, exist_ok=True)

    images = []
    image_ids = []
    captions = []
    caption_ids = []

    annotations = data["annotations"]
    for frame_id in sorted(annotations.keys(), key=lambda x: int(x)):
        for mask_info in annotations[frame_id]["masks"]:
            mask_rel_path = mask_info["mask_path"]
            mask_name = os.path.basename(mask_rel_path)
            frame_num = int(mask_name.replace(
                ".pkl", "").split("_")[0].replace("frame", ""))
            object_id = int(mask_info["object_id"])
            out_path = os.path.join(
                masked_objects_dir, f"frame{frame_num}_obj{object_id}.jpg")
            if not os.path.exists(out_path):
                frame_path = os.path.join(
                    coco_root, "frames", "robotics_kitchen", f"{frame_num - 1}.jpg"
                )
                mask_path = os.path.join(
                    coco_root, "mask_cache", mask_rel_path)

                image = np.array(Image.open(frame_path).convert("RGB"))
                rle = pickle.load(open(mask_path, "rb"))
                mask = decode_uncompressed_rle(rle).astype(bool)

                masked = np.zeros_like(image)
                masked[mask] = image[mask]
                ys, xs = np.where(mask)
                crop = masked[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                Image.fromarray(crop).save(out_path)

            images.append(out_path)
            image_ids.append(object_id)

    present_object_ids = sorted(set(image_ids))
    for object_id in present_object_ids:
        captions.append(
            object_captions[str(object_id)]["object_caption"].strip())
        caption_ids.append(object_id)

    return images, image_ids, captions, caption_ids


def encode_images(model, preprocess, image_paths, batch_size, device):
    import torch

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
    import torch

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


def topk_faiss(query_feats, target_feats, k):
    import faiss

    qf = query_feats.detach().cpu().numpy().astype("float32", copy=False)
    tf = target_feats.detach().cpu().numpy().astype("float32", copy=False)
    qf = np.ascontiguousarray(qf)
    tf = np.ascontiguousarray(tf)
    k = min(k, tf.shape[0])
    index = faiss.IndexFlatIP(tf.shape[1])
    index.add(tf)
    _, indices = index.search(qf, k)
    return indices


def wrap_text(text, max_chars=80):
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if len(trial) <= max_chars:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def visualize_clip_top10(captions, caption_ids, image_paths, image_ids, t2i_rank_clip, t2i_rank_albef, topk=10):
    rows = 2
    cols = 5
    tile_h = 220
    tile_w = 220
    pad = 10
    header_h = 190
    panel_w = cols * tile_w + (cols + 1) * pad
    panel_h = header_h + rows * tile_h + (rows + 1) * pad
    win_clip = "CLIP Top-10"
    win_albef = "ALBEF Top-10"

    def build_panel(title, caption, query_obj_id, ranked_indices):
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        caption_lines = wrap_text(caption, max_chars=85)[:2]
        header_lines = [
            title,
            f"query object_id: {query_obj_id}",
            f"caption: {caption_lines[0] if caption_lines else ''}",
            f"{caption_lines[1] if len(caption_lines) > 1 else ''}",
            "key: any next | q/esc quit",
        ]

        y = 28
        for line in header_lines:
            if line:
                cv2.putText(
                    panel,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            y += 28

        cv2.putText(
            panel,
            "2x5 top-10",
            (10, header_h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        top_indices = ranked_indices[:topk]
        for rank, img_idx in enumerate(top_indices):
            r = rank // cols
            c = rank % cols
            x = pad + c * (tile_w + pad)
            y = header_h + pad + r * (tile_h + pad)

            img = cv2.imread(image_paths[img_idx])
            if img is None:
                tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            else:
                tile = cv2.resize(img, (tile_w, tile_h),
                                  interpolation=cv2.INTER_AREA)
            panel[y:y + tile_h, x:x + tile_w] = tile

            pred_obj_id = image_ids[img_idx]
            is_hit = pred_obj_id == query_obj_id
            border_color = (0, 255, 0) if is_hit else (200, 200, 200)
            cv2.rectangle(panel, (x, y), (x + tile_w - 1,
                                          y + tile_h - 1), border_color, 2)
            cv2.putText(
                panel,
                f"#{rank + 1} obj:{pred_obj_id}",
                (x + 6, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                border_color,
                2,
                cv2.LINE_AA,
            )
        return panel

    for i, caption in enumerate(captions):
        clip_panel = build_panel(
            f"CLIP Top-10 | query {i + 1}/{len(captions)}",
            caption,
            caption_ids[i],
            t2i_rank_clip[i],
        )
        albef_panel = build_panel(
            f"ALBEF Top-10 | query {i + 1}/{len(captions)}",
            caption,
            caption_ids[i],
            t2i_rank_albef[i],
        )

        cv2.imshow(win_clip, clip_panel)
        cv2.imshow(win_albef, albef_panel)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


def albef_rerank(albef_model, albef_preprocess, captions, image_paths, t2i_rank, batch_size, device):
    import torch

    image_feats = []
    image_embeds = []
    total = len(image_paths)
    batch_start = time.time()
    for i in tqdm(range(0, total, batch_size), desc="ALBEF image feats"):
        batch_paths = image_paths[i:i + batch_size]
        images = [albef_preprocess(Image.open(p).convert("RGB"))
                  for p in batch_paths]
        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            batch_feats = albef_model.visual_encoder.forward_features(
                image_input)
            batch_embeds = torch.nn.functional.normalize(
                albef_model.vision_proj(batch_feats[:, 0, :]), dim=-1
            )
        image_feats.append(batch_feats.cpu())
        image_embeds.append(batch_embeds.cpu())
    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    batch_time = time.time() - batch_start
    print(f"ALBEF image feats: {total} in {batch_time:.3f}s")

    text_input = albef_model.tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="pt",
    )
    if hasattr(albef_model.tokenizer, "enc_token_id"):
        text_input.input_ids[:, 0] = albef_model.tokenizer.enc_token_id
    with torch.no_grad():
        text_output = albef_model.text_encoder.forward_text(
            text_input.to(device))
        text_embeds = torch.nn.functional.normalize(
            albef_model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        ).cpu()
    text_ids = text_input.input_ids
    text_atts = text_input.attention_mask

    reranked = []
    total_caps = len(captions)
    batch_start = time.time()
    for i in tqdm(range(total_caps), desc="ALBEF rerank"):
        cand = t2i_rank[i]
        cand_tensor = torch.tensor(cand, dtype=torch.long)
        img_feat = image_feats[cand_tensor].to(device)
        img_embed = image_embeds[cand_tensor].to(device)
        encoder_att = torch.ones(
            img_feat.size()[:-1], dtype=torch.long).to(device)
        text_ids_i = text_ids[i].unsqueeze(0).repeat(len(cand), 1).to(device)
        text_atts_i = text_atts[i].unsqueeze(0).repeat(len(cand), 1).to(device)
        with torch.no_grad():
            output = albef_model.text_encoder(
                text_ids_i,
                attention_mask=text_atts_i,
                encoder_hidden_states=img_feat,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            itm_scores = albef_model.itm_head(
                output.last_hidden_state[:, 0, :])[:, 1]
            sim_scores = torch.matmul(img_embed, text_embeds[i].to(device))
            scores = itm_scores + sim_scores
        order = torch.argsort(scores, descending=True)
        reranked.append([cand[j] for j in order.tolist()])

    batch_time = time.time() - batch_start
    print(f"ALBEF rerank: {total_caps} in {batch_time:.3f}s")
    return reranked


def compute_recalls(ranked_indices, query_object_ids, target_object_ids, ks=(1, 5, 10)):
    hits = {k: 0 for k in ks}

    for i, query_obj_id in enumerate(query_object_ids):
        pred_obj_ids = [target_object_ids[j] for j in ranked_indices[i]]
        for k in ks:
            kk = min(k, len(pred_obj_ids))
            if query_obj_id in pred_obj_ids[:kk]:
                hits[k] += 1

    total = len(query_object_ids)
    return {k: hits[k] / total for k in ks}


def main():
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    images, image_ids, captions, caption_ids = load_karpathy_test(
        ANN_PATH, DATASET_PATH)
    print(f"Images: {len(images)}")
    print(f"Captions: {len(captions)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    t0 = time.time()
    image_feats = encode_images(
        model, preprocess, images, IMAGE_BATCH, device).cpu()
    text_feats = encode_texts(
        model, tokenizer, captions, TEXT_BATCH, device).cpu()
    print(f"Encoded in {time.time() - t0:.1f}s")

    t2i_rank = topk_faiss(text_feats, image_feats, RETRIEVE_K).tolist()

    t2i_clip = compute_recalls(t2i_rank, caption_ids, image_ids, ks=METRIC_KS)
    print("CLIP Text-to-Image recalls:")
    for k in METRIC_KS:
        print(f"R@{k}: {t2i_clip[k]:.2f}")

    from omegaconf import OmegaConf
    from lavis.models import load_preprocess
    from lavis.models.albef_models.albef_retrieval import AlbefRetrieval

    albef_cfg = OmegaConf.load(ALBEF_CFG_PATH)
    albef_model = AlbefRetrieval.from_config(albef_cfg.model)
    albef_model.eval()
    albef_model = albef_model.to(device)
    albef_vis, _ = load_preprocess(albef_cfg.preprocess)

    t2i_rank_albef = albef_rerank(
        albef_model, albef_vis["eval"], captions, images, t2i_rank, ALBEF_IMAGE_BATCH, device
    )
    t2i_albef = compute_recalls(
        t2i_rank_albef, caption_ids, image_ids, ks=METRIC_KS)

    print("ALBEF-reranked Text-to-Image recalls:")
    for k in METRIC_KS:
        print(f"R@{k}: {t2i_albef[k]:.2f}")

    del image_feats, text_feats, albef_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    visualize_clip_top10(
        captions, caption_ids, images, image_ids, t2i_rank, t2i_rank_albef, topk=10
    )


if __name__ == "__main__":
    main()
