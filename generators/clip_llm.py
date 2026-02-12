import time

from tqdm import tqdm

from generators.clip import load, encode_images, encode_texts  # noqa: F401
from retrieval import build_faiss_index, query_faiss_index, rrf_fuse
from gen_params import RETRIEVE_K, TEXT_BATCH, RRF_K
from parsers import load_robo_dataset


PREPHRASE_PATH = "dataset/robotics_kitchen_dataset_v3/objects_caption_prephrases.json"


def load_variants(dataset_name):
    if dataset_name != "robo":
        raise ValueError("clip_llm generator only supports robo dataset (needs prephrases)")
    from datasets import DATASET_CONFIGS
    cfg = DATASET_CONFIGS["robo"]
    _, _, _, _, variants = load_robo_dataset(
        cfg["ann_path"], cfg["dataset_path"], PREPHRASE_PATH)
    return variants


def retrieve(bundle, dataset, image_feats, text_feats, device):
    variants = load_variants("robo")

    all_texts = []
    offsets = []
    for v in variants:
        offsets.append(len(all_texts))
        all_texts.extend(v)
    variant_feats = encode_texts(bundle, all_texts, TEXT_BATCH, device).cpu()

    image_index = build_faiss_index(image_feats)

    t2i_rank = []
    t2i_scores = []
    for i, v in enumerate(tqdm(variants, desc="RRF fusion")):
        start = offsets[i]
        end = start + len(v)
        ranks = query_faiss_index(
            image_index, variant_feats[start:end], RETRIEVE_K
        ).tolist()
        fused, fused_scores = rrf_fuse(ranks, RRF_K, RETRIEVE_K)
        t2i_rank.append(fused)
        t2i_scores.append(fused_scores)

    return t2i_rank, t2i_scores
