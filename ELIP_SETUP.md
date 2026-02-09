# ELIP Reranker Setup Guide

## Quick Setup

Create and activate a conda environment:

```bash
conda create -n elip python=3.8 -y
conda activate elip
```

Install required packages (minimal, just for reranking):

```bash
# PyTorch
pip install torch torchvision

# Core dependencies
pip install pillow tqdm ranx

# ELIP/LAVIS (this will pull in transformers, omegaconf, etc.)
pip install salesforce-lavis
pip install timm==0.4.12
pip install opencv-python-headless
```

**Note:** You do NOT need the full `environment.yml` from ELIP repo - that has 200+ packages for development. The above is enough to run reranking.

## Run the Script

```bash
python rerank-elip.py dataset/robotics_kitchen_dataset_v3/clip_output/openclip-output.json
```

## Notes

- The script uses ELIP (BLIP2-based architecture with Q-Former) from the LAVIS framework
- Uses the ELIP checkpoint at `reqs/12.15_v2_2024_12_15-07_14_55-model_ViT-B-16-lr_0.001-b_20-j_8-p_amp-epoch_1.pt`
- ELIP is a BLIP2 model with visual prompt tuning and multi-prompt learning for robust retrieval
- Reranking uses the ITM (Image-Text Matching) head for scoring
- The script follows the same structure as `rerank-blip.py` and outputs the same metrics
- If you need to change the checkpoint path, edit the `CHECKPOINT_PATH` variable in `rerank-elip.py`
