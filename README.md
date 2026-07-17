# [VC-CGCD: Virtual Category-Guided Continual Generalized Category Discovery](https://arxiv.org/abs/2607.04984)

Official PyTorch implementation of **"Virtual Category-Guided Continual Generalized Category Discovery"** (ECCV 2026).

> Authors: Jiahui Xiong, Qiuxia Lai\*, Hongsong Wang\*  
> \*Corresponding authors  
> Southeast University · Communication University of China

## Overview

Continual Generalized Category Discovery (C-GCD) aims to incrementally identify novel categories from sequential unlabeled data while preserving recognition of known classes, which is an essential capability for open-world visual learning. A major bottleneck lies in ambiguous unlabeled samples that cannot be confidently assigned to known classes nor reliably grouped as novel ones, making pseudo-labeling brittle and often biasing learning toward familiar categories. In this work, we introduce Virtual Category-Guided Continual Generalized Category Discovery by adapting Virtual Category Learning (VCL) to the continual setting. Our method identifies uncertain samples and assigns them to temporary virtual categories, enabling safe and informative learning from unlabeled streams without injecting noisy labels, while improving unlabeled data utilization and mitigating prediction bias. To further stabilize discovery across sessions and enhance class separation, we augment VCL with Expanded Neighborhood Contrastive Learning (ENCL), which exploits extended neighborhood relations and an adaptive margin to learn more discriminative and well-separated representations for both old and emerging classes. Extensive experiments on CIFAR-100, Tiny ImageNet, and ImageNet-100 demonstrate that our approach consistently outperforms state-of-the-art methods, establishing a scalable and effective solution for C-GCD.

<img width="1000" height="316" alt="image" src="https://github.com/user-attachments/assets/bc87fc6d-80df-45db-8058-8b89566b3f65" />


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

## Datasets

- **CIFAR-100**: A 100-class image classification dataset with 600 images per class.
- **Tiny-ImageNet**: A scaled-down version of ImageNet containing 200 classes with 500 images each.
- **ImageNet-100**: A subset of ImageNet with 100 classes and 1200 images per class.

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

## License

This project is released under the MIT License (see [LICENSE](LICENSE)) for **research purposes only**. The code is provided as-is and may change without notice.

## Acknowledgements

Supported by NSFC (62302093, 62306292, 52441503), Jiangsu Natural Science Fund (BK20230833), and the Open Research Fund of the State Key Laboratory of Multimodal Artificial Intelligence Systems. We thank the Big Data Computing Center of Southeast University for computational support.
