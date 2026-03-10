#set document(
  title: "Text to Image Retrieval for Humanoid Robot Knowledge Base: CLIP Tuning",
  author: "Nikita Pichugin",
)

#set page(
  paper: "a4",
  margin: (top: 2cm, bottom: 2cm, left: 2.5cm, right: 1.5cm),
  numbering: "1",
  number-align: center,
)

#set text(font: "New Computer Modern", size: 12pt, lang: "en")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  v(1em)
  text(size: 16pt, weight: "bold", it)
  v(0.5em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 14pt, weight: "bold", it)
  v(0.4em)
}

#show heading.where(level: 3): it => {
  v(0.6em)
  text(size: 12pt, weight: "bold", it)
  v(0.3em)
}

// ── Title page ──────────────────────────────────────────────
#page(numbering: none)[
  #set align(center)
  #v(1cm)

  #text(size: 12pt)[
    Federal State Autonomous Educational Institution \
    of Higher Education \
    *"National Research University 'Higher School of Economics'"*
  ]

  #v(0.8cm)
  #text(size: 12pt)[Faculty of Informatics, Mathematics, and Computer Science ]
  #v(0.3cm)
  #text(size: 11pt)[Master's Programme in Artificial Intelligence and Computer Vision]
  #v(0.3cm)
  #text(size: 11pt)[Master of Science]

  #v(2.5cm)
  #text(size: 18pt, weight: "bold")[REPORT]
  #v(0.3cm)
  #text(size: 14pt)[on Project Internship]

  #v(1cm)
  #text(size: 13pt, weight: "bold")[
    Text to Image Retrieval for Humanoid Robot Knowledge Base: \
    CLIP Tuning
  ]


  #v(1.5cm)
  #set align(left)
  #pad(left: 5cm)[
    #grid(
      columns: (6cm, 5cm),
      row-gutter: 0.8em,
      [Completed by:], [Nikita Pichugin, group 25 ИИКЗ-1],
      [], [#line(length: 5cm) #v(-0.4em) #text(size: 9pt)[(signature)]],
      [Reviewed by:], [Gleb Neshchetkin],
      [], [#line(length: 5cm) #v(-0.4em) #text(size: 9pt)[(signature and date)]],
      [Project grade:], [],
      [], [#line(length: 5cm) #v(-0.4em)],
    )

  ]

  #v(1fr)
  #set align(center)
  #text(size: 12pt)[Moscow, 2026]
]

// ── Table of Contents ───────────────────────────────────────
#outline(title: "Contents", indent: 1.5em, depth: 3)

// ════════════════════════════════════════════════════════════
= Project Overview
// ════════════════════════════════════════════════════════════

*Type of project:* Research

*Project workplace:* Sber Robotics Laboratory

This project investigates two-stage text-to-image retrieval pipelines, where a fast first-stage dense retriever (e.g., CLIP, SigLIP, or DINOv2) produces an initial set of candidate images, and an optional second-stage reranker (e.g., BLIP2 ITM, ALBEF, Qwen3-VL, or ELIP) refines the ranking using more expensive cross-modal scoring.

The goal is to find the optimal trade-off between retrieval quality and inference latency across different model combinations, evaluated on two datasets:
- The standard *COCO 5K Karpathy split* benchmark (image captioning / retrieval community standard).
- A custom *Robotics Kitchen dataset* with object-level segmentation masks and textual descriptions, targeting the practical use case of grounding natural language queries in a humanoid robot's visual knowledge base.

The project also explores LLM-based caption enrichment using Gemma 3 4B to generate multiple textual variants per query, fused via Reciprocal Rank Fusion.

// ════════════════════════════════════════════════════════════
= Methodology and Results
// ════════════════════════════════════════════════════════════

== Description of the Project Task Execution Process

The project was organized into six phases:

+ *Literature review and problem formulation.* Studied state-of-the-art text-to-image retrieval methods including CLIP @clip, SigLIP @siglip, DINOv2 @dinov2 (with text extensions @dinotxt), BLIP2 @blip2, ALBEF @albef, Qwen3-VL @qwen, and ELIP @elip. Identified the two-stage _retrieve-then-rerank_ paradigm as a promising approach for balancing quality and latency.

+ *Dataset preparation.* Prepared two evaluation datasets:
  - *COCO 5K Karpathy split* — a widely used benchmark containing ~5,000 test images with ~5 human-authored captions per image. Evaluations follow the standard text-to-image and image-to-text retrieval protocol.
  - *Robotics Kitchen dataset v3* — a custom dataset of 53 video frames from a humanoid robot operating in a kitchen environment, containing 1155 segmented object instances across 34 unique object categories. Objects are segmented using RLE-encoded masks, cropped with configurable dilation padding, and paired with textual descriptions.

+ *Pipeline implementation.* Developed a modular Python pipeline (`pipeline.py`) supporting pluggable first-stage generators and second-stage rerankers. The architecture uses FAISS `IndexFlatIP` for efficient $k$-nearest-neighbor search ($k = 50$) on $ell_2$-normalized dense embeddings, followed by optional cross-modal reranking.

+ *Model integration and evaluation.* Integrated four first-stage retrievers (CLIP ViT-L-14, SigLIP ViT-SO400M, DINOv2 with text, CLIP+LLM with RRF fusion) and five rerankers (BLIP2 ITM, ALBEF, Qwen3-VL Reranker-2B, ELIP-B, ELIP-C). Evaluated all viable combinations using two metric suites (Ranx and ECCV).

+ *LLM-enriched caption experiments.* Explored enhancing retrieval by generating $N = 4$ text caption variants per query using Gemma 3 4B (temperature $T = 2$) and fusing the per-variant ranked lists via Reciprocal Rank Fusion (RRF, $k = 60$).

+ *Analysis and reporting.* Analyzed quality–latency trade-offs across all model combinations, produced qualitative visualizations of top-$k$ retrieval results, and compiled comprehensive benchmark tables.


== Description of the Project Results

The main deliverable is a comprehensive benchmarking study comparing two-stage text-to-image retrieval pipelines. The key quantitative results are summarized in @coco-table through @alt-table below.

=== COCO 5K Karpathy Split — Text-to-Image Retrieval

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center, center),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Model*], [*R\@1*], [*R\@5*], [*R\@10*], [*MAP\@R*], [*R-P*], [*ECCV R\@1*], [*Avg Query\ Time*]),
    [CLIP _(baseline)_], [0.36], [0.61], [0.71], [0.32], [0.42], [0.73], [0.0002 s],
    [DinoTxt], [0.47], [0.72], [0.80], [0.40], [0.49], [0.83], [0.0007 s],
    [CLIP + ALBEF], [0.52], [0.76], [0.84], [0.41], [0.50], [0.87], [0.9762 s],
    [CLIP + Qwen3-VL], [0.60], [0.81], [0.86], [0.41], [0.49], [0.89], [2.9850 s],
    [*CLIP + BLIP2 ITM*], [*0.63*], [*0.83*], [*0.87*], [*0.42*], [*0.49*], [*0.92*], [*1.3981 s*],
  ),
  caption: [COCO 5K Karpathy Split — text-to-image retrieval results. Best results in bold.],
) <coco-table>

=== Robotics Kitchen — Baseline Captions

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Model*], [*P\@1*], [*P\@5*], [*nDCG\@5*], [*nDCG\@50*], [*MAP\@50*], [*Avg Query\ Time*]),
    [CLIP _(baseline)_], [0.59], [0.51], [0.53], [0.50], [0.38], [0.0002 s],
    [CLIP + ALBEF], [0.53], [0.54], [0.54], [0.50], [0.39], [0.0901 s],
    [CLIP + BLIP2 ITM], [0.71], [0.60], [0.62], [0.53], [0.42], [0.9285 s],
    [CLIP + Qwen3-VL], [0.74], [0.63], [0.65], [0.54], [0.42], [2.0804 s],
    [CLIP + ELIP-B], [0.71], [0.66], [0.67], [0.55], [0.44], [6.0723 s],
    [*DinoTxt*], [*0.74*], [*0.72*], [*0.74*], [*0.67*], [*0.55*], [*0.0001 s*],
  ),
  caption: [Robotics Kitchen — baseline captions. DinoTxt achieves top quality and lowest latency simultaneously.],
) <robo-table>

=== Robotics Kitchen — LLM-Enriched Captions

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Model*], [*P\@1*], [*P\@5*], [*nDCG\@5*], [*nDCG\@50*], [*MAP\@50*], [*Avg Query\ Time*]),
    [CLIP + BLIP2 ITM], [0.65], [0.62], [0.63], [0.54], [0.42], [0.9454 s],
    [*CLIP + Qwen3-VL*], [*0.74*], [*0.66*], [*0.68*], [*0.55*], [*0.42*], [*2.1155 s*],
  ),
  caption: [Robotics Kitchen — LLM-enriched captions (Gemma 3 4B, $N=4$, $T=2$). Marginal gains over baseline captions.],
) <llm-table>

=== Alternative First-Stage Retrievers with ELIP-B Reranking

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*First-Stage Retriever*], [*P\@1*], [*P\@5*], [*nDCG\@5*], [*nDCG\@50*], [*MAP\@50*]),
    [SigLIP + ELIP-B], [0.68], [0.68], [0.68], [0.54], [0.43],
    [*CLIP + ELIP-B*], [*0.71*], [*0.66*], [*0.67*], [*0.55*], [*0.44*],
  ),
  caption: [Effect of swapping the first-stage retriever while keeping ELIP-B as reranker.],
) <alt-table>

=== Key Findings

#enum(
  [On *COCO 5K*, CLIP + BLIP2 ITM achieves the highest quality across all metrics (R\@1: 0.63, ECCV R\@1: 0.92) with moderate latency (1.40 s/query). This represents a *+75% relative improvement* in R\@1 over the CLIP baseline.],
  [On *Robotics Kitchen*, DinoTxt achieves the best overall quality (MAP\@50: 0.55, P\@5: 0.72) while being the _fastest_ model tested (0.0001 s/query) — *4–5 orders of magnitude faster* than two-stage reranking pipelines.],
  [*LLM-enriched captions* (Gemma 3 4B) provide only marginal improvements: Qwen3-VL with LLM captions gains +0.03 on P\@5 and nDCG\@5 over baseline captions, while BLIP2 ITM actually _degrades_ in P\@1 (0.71 #sym.arrow 0.65).],
  [*ALBEF is the weakest reranker* on the Robotics Kitchen dataset, with P\@1 (0.53) actually lower than the CLIP baseline (0.59), suggesting its cross-attention mechanism does not generalize well to object-level crops.],
  [*Replacing CLIP with SigLIP* as first-stage retriever (with ELIP-B reranking) slightly degrades all metrics, indicating CLIP remains the stronger first-stage model for this task.],
)


== Description of the Methods and Technologies Used

=== Architecture Overview

The system follows a two-stage _retrieve-then-rerank_ architecture (@arch-fig):

#figure(
  block(
    width: 100%,
    inset: 12pt,
    stroke: 0.5pt + luma(180),
    radius: 4pt,
    fill: luma(248),
  )[
    #set text(size: 10pt)
    #align(center)[
      #grid(
        columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
        align: center + horizon,
        gutter: 4pt,
        rect(inset: 8pt, stroke: 1pt, radius: 3pt)[
          *Text Query* \
          _"red mug"_
        ],
        sym.arrow.r,
        rect(inset: 8pt, stroke: 1pt, radius: 3pt, fill: rgb("#e8f4e8"))[
          *Dense Encoder* \
          CLIP / SigLIP / DinoTxt
        ],
        sym.arrow.r,
        rect(inset: 8pt, stroke: 1pt, radius: 3pt, fill: rgb("#e8e8f4"))[
          *FAISS Top-$k$* \
          $k = 50$
        ],
        sym.arrow.r,
        rect(inset: 8pt, stroke: 1pt, radius: 3pt, fill: rgb("#f4e8e8"))[
          *Reranker* \
          BLIP2 / ALBEF / Qwen / ELIP
        ],
      )
      #v(6pt)
      #text(size: 9pt, fill: luma(100))[
        Stage 1: $ell_2$-normalized embeddings + inner-product search #h(2em)
        Stage 2: Cross-modal ITM scoring (optional)
      ]
    ]
  ],
  caption: [Two-stage retrieve-then-rerank pipeline architecture.],
) <arch-fig>

+ *First-stage retrieval.* A dense encoder produces $ell_2$-normalized embeddings for both images and text queries. FAISS `IndexFlatIP` performs inner-product-based $k$-nearest-neighbor search to retrieve the top-50 candidate images per query.

+ *Second-stage reranking* _(optional)._ A cross-modal model scores each (query, candidate) pair with finer-grained image-text matching (ITM), re-sorting the candidates by their cross-modal scores.

=== First-Stage Dense Retrievers

*CLIP* (ViT-L-14, OpenAI pretrain) @clip — Contrastive Language-Image Pretraining model with separate vision and text transformer encoders producing aligned 768-dimensional embeddings. Trained on 400M image-text pairs with a contrastive objective that maximizes cosine similarity between matching pairs while minimizing it for non-matching ones.

*SigLIP* (ViT-SO400M-14-SigLIP2-378, WebLI pretrain) @siglip — A variant of CLIP that replaces the softmax-based contrastive loss with a sigmoid loss, enabling more scalable training. Features a larger vision backbone (SO400M) and higher input resolution (378 px).

*DINOv2 with text* (dinov2\_vitl14\_reg4\_dinotxt) @dinov2 @dinotxt — Self-supervised vision transformer extended with native text encoding capability, producing 1280-dimensional embeddings. Unlike CLIP, DINOv2 is pretrained purely on images using self-distillation, with text alignment added as a lightweight extension. Achieves remarkably strong retrieval without any reranking stage.

*CLIP + LLM variants* — CLIP encoder combined with multiple text rephrases generated by Gemma 3 4B, fused via Reciprocal Rank Fusion (RRF, $k = 60$). For each original query caption, $N = 4$ paraphrases are generated at temperature $T = 2$, each producing an independent ranked list that is then fused.

=== Second-Stage Rerankers

*BLIP2 ITM* (pretrain) @blip2 — Vision-language model from the LAVIS library using a Q-Former architecture that bridges a frozen image encoder and a frozen LLM. Uses image-text matching scoring for pairwise cross-modal relevance assessment. Strongest reranker on COCO 5K.

*ALBEF* (base) @albef — Vision-language model with cross-attention fusion between visual and textual representations, from the LAVIS library. Employs momentum distillation for more robust training. Weakest reranker in this evaluation, particularly on domain-specific object crops.

*Qwen3-VL-Reranker-2B* @qwen — Multimodal large language model with 2 billion parameters, adapted for image-text reranking. Leverages the strong visual understanding of VLMs for fine-grained relevance scoring. Best P\@1 on Robotics Kitchen alongside DinoTxt.

*ELIP-B* (iccv\_v27 checkpoint) @elip — Enhanced Language-Image Pretraining model with a custom ITM head and text projection layer. Uses a 7.7 GB checkpoint. Competitive on Robotics Kitchen but with the highest latency (6.07 s/query).

*ELIP-C* — Vision Prompt Tuning (VPT) variant of ELIP, using a ViT-B-16 backbone with text-guided visual prompt tokens. A 2.9 GB checkpoint with learnable prompt parameters injected into the vision transformer.

=== Evaluation Metrics

Two complementary metric suites are employed:

*Ranx metrics* (primary for Robotics Kitchen):
- *Recall\@$K$* — fraction of relevant items found within the top $K$ results.
- *Precision\@$K$* — fraction of top $K$ results that are relevant.
- *nDCG\@$K$* — normalized Discounted Cumulative Gain, rewarding relevant items appearing earlier in the ranked list.
- *MAP\@$K$* — Mean Average Precision, the mean of per-query average precision at rank $K$.

*ECCV COCO-specific metrics* (primary for COCO 5K):
- *MAP\@R* — Mean Average Precision at $R$ relevant items, accounting for the variable number of matching captions per image.
- *R-Precision* — precision at the rank equal to the number of relevant items.
- *ECCV R\@1* — a stricter recall metric designed for the many-to-many matching structure of COCO.

=== Technology Stack

#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: 0.5pt,
  inset: 6pt,
  [*Component*], [*Technology*],
  [Language], [Python 3.10+],
  [Deep Learning], [PyTorch (CUDA / MPS / CPU)],
  [Similarity Search], [FAISS — Facebook AI Similarity Search],
  [CLIP / SigLIP], [OpenCLIP library],
  [BLIP2 / ALBEF], [LAVIS (Salesforce)],
  [Qwen3-VL / Tokenizers], [HuggingFace Transformers],
  [Retrieval Metrics], [ranx library],
  [COCO Metrics], [eccv\_caption, pycocotools],
)


== Description of the Role in the Project

This is an individual project. The Robotics Kitchen dataset (video frames, segmentation masks, and object annotations) was provided by the company's ML team. All other stages were completed independently:
- Literature review and selection of models to benchmark.
- Dataset preprocessing pipeline, including RLE mask decoding, configurable object cropping with dilation padding, and caption mapping.
- Pipeline architecture design and implementation with modular generator/reranker interfaces.
- Integration of 4 first-stage retrievers and 5 rerankers.
- Systematic experimental evaluation across 10+ model combinations on 2 datasets.
- Analysis of results and preparation of the final report.


== Description of Deviations and Difficulties

*Domain gap between datasets.* The most significant difficulty was the large performance gap between the two evaluation datasets. Pipelines that perform well on COCO 5K — where images are full natural scenes with rich, descriptive captions — give substantially worse results on the Robotics Kitchen dataset, which consists of tightly cropped masked objects with short domain-specific descriptions. For example, CLIP + ALBEF achieves R\@1 of 0.52 on COCO but drops to P\@1 of just 0.53 on Robotics Kitchen (below the CLIP-only baseline of 0.59). This domain gap meant that model selection and hyperparameter choices could not simply transfer from one dataset to the other, requiring independent tuning and evaluation for each setting.

*Latency–quality trade-off.* The most accurate rerankers (BLIP2, Qwen3-VL, ELIP-B) are 3–5 orders of magnitude slower than first-stage retrievers. For example, ELIP-B requires ~6 s per query versus ~0.0001 s for DinoTxt. This makes two-stage pipelines impractical for real-time or large-scale deployment without further optimization such as model distillation, quantization, or batched inference.

*LLM caption enrichment limitations.* LLM-generated caption variants via Gemma 3 4B provided only marginal gains and in some cases _degraded_ performance (BLIP2 P\@1 dropped from 0.71 to 0.65). This indicates that current text augmentation strategies may introduce semantic noise rather than useful additional signal, and that retrieval performance is currently bottlenecked by _visual representation quality_ rather than text query coverage.

*GPU memory constraints.* Running large rerankers (especially Qwen3-VL at 2B parameters and ELIP models with 7.7 GB checkpoints) required careful batch size tuning (`IMAGE_BATCH=64`, `TEXT_BATCH=256`) and sequential per-query processing to fit within available GPU memory. This also contributed to the high per-query latency of reranking models.

*Robotics Kitchen dataset specifics.* The custom dataset required handling RLE-encoded segmentation masks from pickle files, implementing configurable padding/dilation for small objects that might otherwise lose context when tightly cropped, and maintaining consistent mappings between object-level captions and multi-frame image sources — adding significant complexity to the data loading pipeline.


// ════════════════════════════════════════════════════════════
= Conclusion
// ════════════════════════════════════════════════════════════

This project provided a systematic comparison of two-stage text-to-image retrieval pipelines, evaluating multiple combinations of first-stage dense retrievers and second-stage rerankers across two distinct datasets. The main conclusions are:

+ *The two-stage paradigm consistently improves retrieval quality over single-stage baselines.* On COCO 5K, CLIP + BLIP2 ITM improves R\@1 from 0.36 to 0.63 (+75%) and ECCV R\@1 from 0.73 to 0.92 (+26%). The improvement is systematic across all rerankers except ALBEF on domain-specific data.

+ *DINOv2 with text (DinoTxt) emerges as a Pareto-optimal single-stage alternative.* It achieves the best quality on the Robotics Kitchen dataset (MAP\@50: 0.55, P\@1: 0.74) while being the fastest model tested (0.0001 s/query). This suggests that advances in self-supervised vision-language models can _eliminate the need for expensive reranking_ in certain application domains.

+ *LLM-based text augmentation does not provide consistent improvements.* Caption enrichment via Gemma 3 4B yields at most +0.03 gain on individual metrics and sometimes degrades performance. This indicates that retrieval quality is currently bottlenecked by visual representation fidelity rather than textual query expressiveness.

+ *Model choice is dataset-dependent.* BLIP2 ITM excels on COCO 5K (general-domain natural images with rich captions), while DinoTxt dominates on Robotics Kitchen (domain-specific object crops with short descriptions). No single model wins across all settings.

+ *Practical deployment requires latency-aware model selection.* For real-time applications (e.g., humanoid robot object grounding), DinoTxt provides the best quality at sub-millisecond latency. For offline or batch-processing scenarios where quality is paramount, CLIP + BLIP2 ITM offers the highest accuracy at ~1.4 s per query.

*Competencies developed* during this project include: deep understanding of modern vision-language models (CLIP, SigLIP, DINOv2, BLIP2, ALBEF, Qwen3-VL), practical experience with information retrieval evaluation methodology (Ranx, nDCG, MAP, ECCV metrics), proficiency in building modular ML pipelines with pluggable components, and experience with efficient similarity search using FAISS.


// ════════════════════════════════════════════════════════════
= Project Results
// ════════════════════════════════════════════════════════════

The project produced the following artifacts:

+ *A modular two-stage retrieval pipeline* (`pipeline.py`) supporting 4 first-stage generators and 5 rerankers with configurable evaluation metrics, visualization, and result serialization.

+ *Comprehensive benchmark tables* comparing 10+ model combinations on 2 datasets across 6+ retrieval metrics, with per-query latency measurements.

+ *Qualitative visualizations* of top-$k$ retrieval results for every evaluated model configuration (42 visualization images across both datasets).


=== Summary of Best Results

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, center, center),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Dataset*], [*Best Model*], [*Key Metric*], [*Value*], [*Query Time*]),
    [COCO 5K], [CLIP + BLIP2 ITM], [R\@1], [0.63], [1.40 s],
    [COCO 5K], [CLIP + BLIP2 ITM], [ECCV R\@1], [0.92], [1.40 s],
    [Robotics Kitchen], [DinoTxt], [MAP\@50], [0.55], [0.0001 s],
    [Robotics Kitchen], [DinoTxt], [P\@1], [0.74], [0.0001 s],
  ),
  caption: [Best-performing models per dataset and metric.],
)

Source code repository: #link("https://github.com/ironyoid/clip-bench")


// ════════════════════════════════════════════════════════════
= Appendices
// ════════════════════════════════════════════════════════════

== Pipeline Architecture <appendix-arch>

The pipeline follows a modular design pattern with clearly defined interfaces:

- *Generators* (`generators/` directory) implement three functions: `load()`, `encode_images()`, and `encode_texts()`. Each returns $ell_2$-normalized embeddings suitable for inner-product similarity search.
- *Rerankers* (`rerankers/` directory) implement `load()` and `rerank()`. The reranker receives the top-$k$ candidates from the first stage and returns a re-scored ranking.
- *The main pipeline* (`pipeline.py`) orchestrates the full flow: dataset loading, encoding, FAISS retrieval ($k = 50$), optional reranking, metric computation, interactive visualization, and JSON result serialization.

Results can be saved to JSON (with associated `.pt` feature tensors) for efficient reranking experiments without re-encoding.


== Pipeline Usage <appendix-usage>

```bash
# Generate embeddings + retrieve + evaluate
python pipeline.py --dataset robo --generator clip --metrics ranx

# Two-stage: generate + rerank
python pipeline.py --dataset coco --generator clip --reranker blip --metrics eccv

# Save first-stage output, then experiment with rerankers
python pipeline.py --dataset robo --generator dino --save dino_robo.json
python pipeline.py --input dino_robo.json --reranker qwen --metrics ranx

# Visualize top-k results interactively
python pipeline.py --input dino_robo.json --visualize

# Object crop padding (robotics dataset only)
python pipeline.py --dataset robo --generator clip --padding 10

# Limit to N queries for fast debugging
python pipeline.py --dataset robo --generator clip -n 5
```

== Hyperparameters <appendix-hyper>

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, center, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Parameter*], [*Value*], [*Description*]),
    [`IMAGE_BATCH`], [64], [Batch size for image encoding],
    [`TEXT_BATCH`], [256], [Batch size for text encoding],
    [`RETRIEVE_K`], [50], [Number of candidates retrieved from first stage],
    [`METRIC_KS`], [(1, 5, 50)], [Cutoff values for evaluation metrics],
    [`RRF_K`], [60], [Reciprocal Rank Fusion smoothing parameter],
    [LLM $N$], [4], [Number of text variants per query (LLM enrichment)],
    [LLM $T$], [2], [Temperature for LLM caption generation (Gemma 3 4B)],
  ),
  caption: [Hyperparameters used throughout all experiments.],
)


== Qualitative Retrieval Examples <appendix-viz>

Selected top-$k$ retrieval visualizations from the Robotics Kitchen dataset are shown below. Green-bordered images indicate correct matches.

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  figure(
    image("images/query_0004-clip.png", width: 100%),
    caption: [CLIP baseline — query "cabbage"],
  ),
  figure(
    image("images/query_0004-dinotxt.png", width: 100%),
    caption: [DinoTxt — query "cabbage"],
  ),

  figure(
    image("images/query_0005-blip2.png", width: 100%),
    caption: [CLIP + BLIP2 — query "toaster"],
  ),
  figure(
    image("images/query_0005-qwen-llm.png", width: 100%),
    caption: [CLIP + Qwen3-VL (LLM) — query "toaster"],
  ),
)


// ── Bibliography ────────────────────────────────────────────

#pagebreak()
#bibliography("refs.yml", title: "References", style: "ieee")
