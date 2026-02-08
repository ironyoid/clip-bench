# [CLIP COCO 5K Kharpaty Split dataset]

## Baseline captions

### Baseline CLIP (ViT-L-14, OpenAI pretrain)
COCO 5K T2I recalls:
R@1: 0.36   Map@R: 0.32
R@5: 0.61   R-P: 0.42
R@10: 0.71  R@1: 0.73

### CLIP (ViT-L-14, OpenAI pretrain) -> ALBEF (base)
```
R@1: 0.52   Map@R: 0.41
R@5: 0.76   R-P: 0.50
R@10: 0.84  R@1: 0.87
```

### CLIP (ViT-L-14, OpenAI pretrain) -> BLIP2 ITM (pretrain)

### CLIP (ViT-L-14, OpenAI pretrain) -> Qwen3-VL-Embedding (Qwen3-VL-Reranker-2B)
```
R@1: 0.60   Map@R: 0.41
R@5: 0.81   R-P: 0.49
R@10: 0.86  R@1: 0.89
```

# [robotics_kitchen_dataset_v3]

## Baseline captions

### Baseline CLIP (ViT-L-14, OpenAI pretrain)
```
R@1: 0.02  P@1: 0.59  nDCG@1: 0.59
R@5: 0.08  P@5: 0.51  nDCG@5: 0.53
R@10: 0.15  P@10: 0.49  nDCG@10: 0.51
MAP@10: 0.13
```

### CLIP (ViT-L-14, OpenAI pretrain) -> ALBEF (base)
```
R@1: 0.02  P@1: 0.53  nDCG@1: 0.53
R@5: 0.09  P@5: 0.55  nDCG@5: 0.54
R@10: 0.15  P@10: 0.49  nDCG@10: 0.51
MAP@10: 0.13
```

### CLIP (ViT-L-14, OpenAI pretrain) -> BLIP2 ITM (pretrain)
```
R@1: 0.02  P@1: 0.71  nDCG@1: 0.71
R@5: 0.09  P@5: 0.58  nDCG@5: 0.61
R@10: 0.15  P@10: 0.49  nDCG@10: 0.54
MAP@10: 0.14
```

### CLIP (ViT-L-14, OpenAI pretrain) -> Qwen3-VL-Embedding (Qwen3-VL-Reranker-2B)
```
R@1: 0.02  P@1: 0.74  nDCG@1: 0.74
R@5: 0.10  P@5: 0.66  nDCG@5: 0.67
R@10: 0.19  P@10: 0.62  nDCG@10: 0.64
MAP@10: 0.17
```

## LLM (gemma3:4b) enriched captions (N=4, T=2) 

### CLIP (ViT-L-14, OpenAI pretrain) -> BLIP2 ITM (pretrain)
```
R@1: 0.02  P@1: 0.74  nDCG@1: 0.74
R@5: 0.09  P@5: 0.60  nDCG@5: 0.63
R@10: 0.15  P@10: 0.51  nDCG@10: 0.55
MAP@10: 0.14
```

### CLIP (ViT-L-14, OpenAI pretrain) -> Qwen3-VL-Embedding (Qwen3-VL-Reranker-2B)