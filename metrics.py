from ranx import Qrels, Run, evaluate
from retrieval import build_qrels, build_run


def compute_ranx(caption_match_ids, image_ids, t2i_rank, t2i_scores, ks):
    qrels = Qrels(build_qrels(caption_match_ids, image_ids))
    run = Run(build_run(t2i_rank, t2i_scores))
    metric_names = []
    for k in ks:
        metric_names.extend([f"recall@{k}", f"precision@{k}", f"ndcg@{k}"])
    metric_names.append(f"map@{max(ks)}")
    results = evaluate(qrels, run, metric_names)

    print("Ranx metrics:")
    for k in ks:
        r = results[f"recall@{k}"]
        p = results[f"precision@{k}"]
        n = results[f"ndcg@{k}"]
        print(f"  R@{k}: {r:.2f}  P@{k}: {p:.2f}  nDCG@{k}: {n:.2f}")
    print(f"  MAP@{max(ks)}: {results[f'map@{max(ks)}']:.2f}")
    return results


def compute_eccv(image_ids, caption_ids, i2t_rank, t2i_rank):
    import eccv_caption

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

    r1 = scores['coco_5k_r1']['t2i']
    r5 = scores['coco_5k_r5']['t2i']
    r10 = scores['coco_5k_r10']['t2i']
    mapr = scores['eccv_map_at_r']['t2i']
    rp = scores['eccv_rprecision']['t2i']
    er1 = scores['eccv_r1']['t2i']
    print("ECCV T2I metrics:")
    print(f"  R@1:  {r1:.2f}    Map@R: {mapr:.2f}")
    print(f"  R@5:  {r5:.2f}    R-P:   {rp:.2f}")
    print(f"  R@10: {r10:.2f}   R@1 (eccv): {er1:.2f}")
    return scores
