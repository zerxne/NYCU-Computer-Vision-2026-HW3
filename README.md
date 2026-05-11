# NYCU Computer Vision 2026 HW3

- **Student ID**: 314554032
- **Name**: 江怜儀

---

## Introduction
The task involves segmenting four types of cells (class1–class4) from colored medical histology images, evaluated using AP@50 on a held-out test set via CodaBench.

The core approach is based on Mask R-CNN, enhanced with a stronger Swin Transformer backbone, two-stage training (frozen → full fine-tuning), Stochastic Weight Averaging (SWA), and Test-Time Augmentation (TTA) with Soft-NMS at inference. These components collectively aim to push the model beyond the strong baseline of AP@50 ≈ 0.35.

---

## Environment Setup
Step 1 — Install PyTorch with CUDA (adjust `cu118` to match your driver, e.g. `cu121`):
 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
 
Step 2 — Install all other dependencies:
 
```bash
pip install -r requirements.txt
```
 
> CPU-only users: replace the Step 1 command with:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```
---
## Dataset Structure
 
Place the dataset under a single root directory. The expected layout is:
 
```
dataset/
├── train/
│   ├── 0001/
│   │   ├── image.tif
│   │   ├── class1.tif
│   │   ├── class2.tif
│   │   ├── class3.tif
│   │   └── class4.tif
│   ├── 0002/
│   │   └── ...
│   └── ...
└── test_release/
    ├── <image files>
    └── ...
    test_image_name_to_ids.json
```

Each `classN.tif` is a binary mask for foreground instances of class N. Connected components are used to extract individual instances.

---
## Usage

### Training

#### Basic command (ResNet-50, recommended for first run)
 
```bash
python train.py --image-dir ./dataset
```
 
#### Recommended combos
 
```bash
# ResNet-50v2, 1024 px, 40 epochs
python train.py --image-dir ./dataset --model-type res --min-size 1024 --epochs 40
 
# Swin-S, 800 px, 50 epochs (reduce batch size for memory)
python train.py --image-dir ./dataset --model-type swin_s --epochs 50 --batch-size 1
```

### Inference

#### Run Inference

```bash
python test.py \
  --model-type   res \
  --weights      best_model.pth \
  --image-dir    ./dataset \
  --score-thr    0.3 \
  --soft-nms-sigma 0.5
```


## Performance Snapshot
