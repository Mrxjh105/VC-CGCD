# VC-CGCD: Virtual Category-Guided Continual Generalized Category Discovery

Official PyTorch implementation of **"Virtual Category-Guided Continual Generalized Category Discovery"** (ECCV 2026).

> Authors: Jiahui Xiong, Qiuxia Lai\*, Hongsong Wang\*  
> \*Corresponding authors  
> Southeast University · Communication University of China

<img width="793" height="243" alt="image" src="https://github.com/user-attachments/assets/9befd892-cb81-42ac-bab5-8b8278e49936" />


## Overview

VC-CGCD addresses **Continual Generalized Category Discovery (C-GCD)**: incrementally discovering novel categories from sequential unlabeled sessions while retaining previously learned ones, without storing past data. The framework introduces **Virtual Category Learning (VCL)** to handle ambiguous unlabeled samples via temporary virtual categories, and **Expanded Neighborhood Contrastive Learning (ENCL)** to improve feature separation through neighbors-of-neighbors mining.

## Project Structure

```
VCCGCD-main/
├── train.py                  # Main training script (offline + online)
├── config.py                 # Dataset and experiment paths
├── data/                     # Dataset loaders
├── models/
│   ├── vision_transformer.py # ViT-B/16 backbone
│   ├── utils_simgcd*.py      # Training utilities (DINOHead, losses, k-means init, ...)
│   ├── utils_proto_aug.py    # Prototype augmentation manager
│   └── vc.py                 # VCHAP: VC classifier + virtual weight generator
├── project_utils/            # Logging, clustering, seeding utilities
└── ssb_splits/               # SSB splits for C-GCD protocol
```

## Requirements

```bash
pip install -r requirements.txt
# Additional: faiss-gpu, tensorboard, openpyxl
```

Python 3.8+ · PyTorch 1.10 · CUDA-enabled GPU (RTX 4090D used in the paper).

## Training

### Stage-0: Offline Pre-training

```bash
python train.py \
    --dataset_name cifar100 \
    --train_session offline \
    --num_old_classes 50 \
    --prop_train_labels 0.8 \
    --epochs_offline 100 \
    --batch_size 128 --lr 0.1 --transform imagenet \
    --continual_session_num 10 \
    --online_novel_unseen_num 200 \
    --online_old_seen_num 25 --online_novel_seen_num 25 \
    --fp16 --seed 0
```

### Stage-1+: Online Continual Learning

```bash
python train.py \
    --dataset_name cifar100 \
    --train_session online \
    --num_old_classes 50 --prop_train_labels 0.8 \
    --epochs_online_per_session 30 \
    --batch_size 128 --lr 0.01 --transform imagenet \
    --warmup_teacher_temp 0.05 --teacher_temp 0.05 --warmup_teacher_temp_epochs 10 \
    --memax_old_new_weight 1 --memax_old_in_weight 1 --memax_new_in_weight 1 \
    --proto_aug_weight 1 --feat_distill_weight 1.2 --radius_scale 1.0 --hardness_temp 0.1 \
    --eval_funcs v2 \
    --continual_session_num 10 \
    --online_novel_unseen_num 200 \
    --online_old_seen_num 25 --online_novel_seen_num 25 \
    --init_new_head --shuffle_classes --fp16 --seed 0 \
    --load_offline_id <offline_checkpoint_id>
```

## Key Arguments

| Argument                      | Description                                            |
| ----------------------------- | ------------------------------------------------------ |
| `--train_session`             | `offline` (stage-0) or `online` (stage-1+)             |
| `--num_old_classes`           | Number of labeled classes in stage-0 (e.g., 50)        |
| `--prop_train_labels`         | Label ratio per init class (e.g., 0.8)                 |
| `--continual_session_num`     | Total number of online sessions (e.g., 5 or 10)       |
| `--online_novel_unseen_num`   | New classes per session                                |
| `--init_new_head`             | Initialize new head with k-means centroids             |
| `--load_offline_id`           | Checkpoint ID from the offline stage                   |
| `--shuffle_classes`           | Shuffle novel class order                              |
| `--fp16`                      | Mixed-precision training                               |

## Citation

```bibtex
@inproceedings{xiong2026vccgcd,
  title     = {Virtual Category-Guided Continual Generalized Category Discovery},
  author    = {Xiong, Jiahui and Lai, Qiuxia and Wang, Hongsong},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
