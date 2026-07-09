# [Virtual Category-Guided Continual Generalized Category Discovery](https://arxiv.org/abs/2607.04984)

A PyTorch implementation for continual learning with novel class discovery, combining visual concepts (VC) with contrastive learning and knowledge distillation.

## Disclaimer

This code is provided **for research purposes only**. The authors do not guarantee the correctness, completeness, or performance of the code. The code is subject to change without notice as research progresses.

**License**: This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Features

- **Offline Pre-training**: Initial training on labeled old classes
- **Online Continual Learning**: Incremental learning of novel classes across multiple sessions
- **Visual Concept Classifier**: VC-based classification head for handling confusing samples
- **Memory-efficient**: Supports mixed precision training (FP16)

## Project Structure

```
VCCGCD-CIFAR/
├── train.py              # Main training script
├── config.py             # Dataset and experiment paths
├── models/
│   ├── vc.py             # Visual Concept classifier
│   ├── vision_transformer.py  # ViT backbone (DINO)
│   └── utils_simgcd*.py  # Training utilities
├── data/                 # Dataset loaders
└── run_c100.sh          # Example training script
```

## Requirements

```bash
pip install torch torchvision
pip install faiss-gpu
pip install scikit-learn matplotlib pandas tqdm
pip install tensorboard
```

## Datasets

Supported datasets:
- CIFAR-100
- TinyImageNet
- ImageNet-100

Configure dataset paths in `config.py`:
```python
cifar_100_root = '/data/yourpath/CIFAR/'
tiny_imagenet_root = '/data/yourpath/TinyImageNet/'
imagenet_root = '/data2/yourpath/ImageNet2012/'
```

## Training

### Offline Pre-training (Stage 0)

```bash
python train.py \
    --dataset_name cifar100 \
    --batch_size 128 \
    --transform imagenet \
    --lr 0.1 \
    --num_old_classes 50 \
    --prop_train_labels 0.8 \
    --train_session offline \
    --epochs_offline 100 \
    --continual_session_num 10 \
    --online_novel_unseen_num 200 \
    --online_old_seen_num 25 \
    --online_novel_seen_num 25 \
    --fp16 \
    --seed 0
```

### Online Continual Learning (Stage 1+)

```bash
python train.py \
    --dataset_name cifar100 \
    --batch_size 128 \
    --transform imagenet \
    --warmup_teacher_temp 0.05 \
    --teacher_temp 0.05 \
    --warmup_teacher_temp_epochs 10 \
    --lr 0.01 \
    --memax_old_new_weight 1 \
    --memax_old_in_weight 1 \
    --memax_new_in_weight 1 \
    --proto_aug_weight 1 \
    --feat_distill_weight 1.2 \
    --radius_scale 1.0 \
    --hardness_temp 0.1 \
    --eval_funcs v2 \
    --num_old_classes 50 \
    --prop_train_labels 0.8 \
    --train_session online \
    --epochs_online_per_session 30 \
    --continual_session_num 10 \
    --online_novel_unseen_num 200 \
    --online_old_seen_num 25 \
    --online_novel_seen_num 25 \
    --init_new_head \
    --load_offline_id <offline_checkpoint_id> \
    --shuffle_classes \
    --fp16 \
    --seed 0
```

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--dataset_name` | Dataset: cifar100, tiny_imagenet, imagenet_100 |
| `--num_old_classes` | Number of initial labeled classes |
| `--continual_session_num` | Number of continual learning sessions |
| `--prop_train_labels` | Ratio of labeled data from old classes |
| `--train_session` | offline or online |
| `--epochs_online_per_session` | Epochs per online session |
| `--init_new_head` | Initialize new head with k-means centroids |
| `--load_offline_id` | Checkpoint ID to load offline model |
| `--fp16` | Enable mixed precision training |
| `--shuffle_classes` | Shuffle novel class order |

## Output

Checkpoints and logs saved to:
```
<exp_root>_offline/<dataset_name>/<checkpoint_id>/checkpoints/
<exp_root_online>/<dataset_name>/<checkpoint_id>/checkpoints/
```

## Citation

If you find this code useful, please cite:

```bibtex
@article{yourpaper2025,
  title={VC-CGCD: Visual Concept-aware Continual Generalised Class Discovery},
  author={Your Name},
  year={2026}
}
```
