import argparse
import importlib
import json
import os
import time
from collections import namedtuple

import torch

from retrieval import topk_faiss
from metrics import compute_ranx, compute_eccv
from gen_params import IMAGE_BATCH, TEXT_BATCH, RETRIEVE_K, METRIC_KS
from generators import GENERATORS
from rerankers import RERANKERS

Dataset = namedtuple("Dataset", ["images", "image_ids", "captions", "caption_ids", "caption_match_ids"])


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["robo", "coco"])
    parser.add_argument("--generator", choices=list(GENERATORS.keys()))
    parser.add_argument("--input", help="Path to JSON from a previous run (skips generator)")
    parser.add_argument("--reranker", choices=list(RERANKERS.keys()))
    parser.add_argument("--metrics", default="ranx", choices=["ranx", "eccv"])
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save", help="Path to save JSON output")
    parser.add_argument("--padding", type=int, default=0,
                        help="Mask padding in pixels (robo dataset only)")
    parser.add_argument("-n", type=int, help="Limit to first N queries")
    args = parser.parse_args()

    if not args.input and not (args.dataset and args.generator):
        parser.error("either --input or both --dataset and --generator are required")

    device = get_device()
    print(f"Device: {device}")

    img_feats = None
    txt_feats = None
    n = args.n

    if args.input:
        data = json.load(open(args.input, "r", encoding="utf-8"))
        caption_match_ids = data.get("caption_match_ids", data["caption_ids"])
        captions = data["captions"][:n]
        caption_ids = data["caption_ids"][:n]
        caption_match_ids = caption_match_ids[:n]
        dataset = Dataset(
            data["images"], data["image_ids"],
            captions, caption_ids, caption_match_ids,
        )
        t2i_rank = data["t2i_rank"][:n]
        t2i_scores = data["t2i_scores"][:n]
        feats_path = args.input.rsplit(".", 1)[0] + ".pt"
        if os.path.exists(feats_path):
            feats = torch.load(feats_path, weights_only=True)
            img_feats = feats["img_feats"]
            txt_feats = feats["txt_feats"][:n] if n else feats["txt_feats"]
            print(f"Loaded features from {feats_path}")
        print(f"Loaded from {args.input}: {len(dataset.images)} images, {len(dataset.captions)} captions")
    else:
        from datasets import load_dataset
        dataset = load_dataset(args.dataset, padding=args.padding)
        if n:
            dataset = Dataset(
                dataset.images, dataset.image_ids,
                dataset.captions[:n], dataset.caption_ids[:n],
                dataset.caption_match_ids[:n],
            )
            print(f"Limited to {n} queries")

        gen = importlib.import_module(GENERATORS[args.generator])
        print(f"Loading generator: {args.generator}")
        bundle = gen.load(device)

        t0 = time.time()
        img_feats = gen.encode_images(bundle, dataset.images, IMAGE_BATCH, device).cpu()
        txt_feats = gen.encode_texts(bundle, dataset.captions, TEXT_BATCH, device).cpu()
        print(f"Encoded in {time.time() - t0:.1f}s")

        t_search = time.time()
        if hasattr(gen, "retrieve"):
            t2i_rank, t2i_scores = gen.retrieve(
                bundle, dataset, img_feats, txt_feats, device)
        else:
            t2i_rank_np, t2i_scores_np = topk_faiss(
                txt_feats, img_feats, RETRIEVE_K, return_scores=True)
            t2i_rank = t2i_rank_np.tolist()
            t2i_scores = t2i_scores_np.tolist()
        search_time = time.time() - t_search
        print(f"Retrieval: {search_time:.3f}s, avg query: {search_time / max(1, len(dataset.captions)):.4f}s")

    if args.reranker:
        rr = importlib.import_module(RERANKERS[args.reranker])
        print(f"Loading reranker: {args.reranker}")
        rr_bundle = rr.load(device)
        t_rerank = time.time()
        t2i_rank, t2i_scores = rr.rerank(
            rr_bundle, dataset.images, dataset.captions,
            t2i_rank, None, device)
        rerank_time = time.time() - t_rerank
        print(f"Rerank: {rerank_time:.3f}s, avg query: {rerank_time / max(1, len(dataset.captions)):.4f}s")

    if args.metrics == "ranx":
        compute_ranx(dataset.caption_match_ids, dataset.image_ids,
                      t2i_rank, t2i_scores, METRIC_KS)
    elif args.metrics == "eccv":
        if args.n:
            print("eccv metrics are incompatible with -n (requires full dataset).")
        elif img_feats is None or txt_feats is None:
            print("eccv metrics require features (not available with --input).")
        else:
            i2t_rank_np, _ = topk_faiss(
                img_feats, txt_feats, RETRIEVE_K, return_scores=True)
            compute_eccv(dataset.image_ids, dataset.caption_ids,
                          i2t_rank_np.tolist(), t2i_rank)

    if args.visualize:
        from visualize import visualize_topk
        parts = [args.dataset or "input", args.generator or "input"]
        parts.append(args.reranker or "norerank")
        if args.padding:
            parts.append(f"pad{args.padding}")
        model_name = "_".join(parts)
        visualize_topk(dataset.captions, dataset.images, t2i_rank,
                       dataset.caption_match_ids, dataset.image_ids,
                       model_name=model_name)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        out = {
            "meta": {
                "dataset": args.dataset,
                "generator": args.generator,
                "reranker": args.reranker,
                "metrics_type": args.metrics,
                "padding": args.padding,
            },
            "images": dataset.images,
            "image_ids": dataset.image_ids,
            "captions": dataset.captions,
            "caption_ids": dataset.caption_ids,
            "caption_match_ids": dataset.caption_match_ids,
            "t2i_rank": t2i_rank,
            "t2i_scores": t2i_scores,
        }
        json.dump(out, open(args.save, "w", encoding="utf-8"), indent=2)
        print(f"Saved: {args.save}")
        if img_feats is not None and txt_feats is not None:
            feats_path = args.save.rsplit(".", 1)[0] + ".pt"
            torch.save({"img_feats": img_feats, "txt_feats": txt_feats}, feats_path)
            print(f"Saved features: {feats_path}")


if __name__ == "__main__":
    main()
