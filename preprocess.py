import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import cv2
import numpy as np
import tifffile as sio
from pathlib import Path
import random


class InstanceSegmentationDataset(Dataset):

    SCALES = [480, 512, 640, 720, 800]

    def __init__(self, root_dir, augment=False):
        self.root_dir = Path(root_dir)
        self.augment  = augment
        self.samples  = sorted(self.root_dir.iterdir())


    def _load_raw(self, folder):
        image = cv2.imread(str(folder / 'image.tif'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks, labels = [], []
        for class_id in range(1, 5):
            mask_file = folder / f'class{class_id}.tif'
            if not mask_file.exists():
                continue
            class_mask = sio.imread(mask_file)
            n, label_map = cv2.connectedComponents(class_mask.astype(np.uint8))
            for inst_id in range(1, n):
                m = (label_map == inst_id).astype(np.uint8)
                if m.sum() == 0:
                    continue
                masks.append(m)
                labels.append(class_id)
        return image, masks, labels

    @staticmethod
    def _flip(image, masks, code):
        image = cv2.flip(image, code)
        masks = [cv2.flip(m, code) for m in masks]
        return image, masks

    @staticmethod
    def _rot90(image, masks, k):
        image = np.rot90(image, k).copy()
        masks = [np.rot90(m, k).copy() for m in masks]
        return image, masks

    @staticmethod
    def _color_jitter(image):
        alpha = random.uniform(0.7, 1.3)
        beta  = random.randint(-25, 25)
        return np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    @staticmethod
    def _resize_to_scale(image, masks, scale):
        h, w = image.shape[:2]
        if min(h, w) == scale:
            return image, masks
        ratio = scale / min(h, w)
        new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        masks = [cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST) for m in masks]
        return image, masks

    def _copypaste(self, image, masks, labels):
        if len(self.samples) < 2:
            return image, masks, labels

        donor_folder = random.choice(self.samples)
        d_img, d_masks, d_labels = self._load_raw(donor_folder)

        h, w = image.shape[:2]
        d_img   = cv2.resize(d_img,  (w, h), interpolation=cv2.INTER_LINEAR)
        d_masks = [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in d_masks]

        if not d_masks:
            return image, masks, labels

        indices = random.sample(range(len(d_masks)), min(3, len(d_masks)))
        for idx in indices:
            inst_mask = d_masks[idx]
            if inst_mask.sum() == 0:
                continue

            ys, xs = np.where(inst_mask)
            dy = random.randint(-min(int(ys.min()), h // 4), min(h - int(ys.max()) - 1, h // 4))
            dx = random.randint(-min(int(xs.min()), w // 4), min(w - int(xs.max()) - 1, w // 4))

            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_mask = cv2.warpAffine(inst_mask, M, (w, h),
                                          flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            shifted_img  = cv2.warpAffine(d_img,     M, (w, h),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT)
            fg = shifted_mask > 0
            if fg.sum() == 0:
                continue

            image[fg] = shifted_img[fg]
            masks.append(shifted_mask.astype(np.uint8))
            labels.append(d_labels[idx])

        return image, masks, labels


    def __getitem__(self, idx):
        folder = self.samples[idx]
        image, masks, labels = self._load_raw(folder)

        if self.augment and len(masks) > 0:
            if random.random() > 0.5:
                image, masks = self._flip(image, masks, 1)
            if random.random() > 0.5:
                image, masks = self._flip(image, masks, 0)
            k = random.randint(0, 3)
            if k > 0:
                image, masks = self._rot90(image, masks, k)
            image = self._color_jitter(image)
            if random.random() > 0.5:
                image, masks, labels = self._copypaste(image, masks, labels)
            scale = random.choice(self.SCALES)
            image, masks = self._resize_to_scale(image, masks, scale)

        image_tensor = F.to_tensor(image)
        H, W = image_tensor.shape[1:]

        boxes, valid_masks, valid_labels = [], [], []
        for i, mask in enumerate(masks):
            mask_t = torch.as_tensor(mask, dtype=torch.uint8)
            pos    = torch.where(mask_t)
            if pos[0].numel() == 0 or pos[1].numel() == 0:
                continue
            xmin, xmax = pos[1].min().item(), pos[1].max().item()
            ymin, ymax = pos[0].min().item(), pos[0].max().item()
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            valid_masks.append(mask_t)
            valid_labels.append(labels[i])

        if len(boxes) == 0:
            target = {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros((0,),   dtype=torch.int64),
                "masks":    torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area":     torch.tensor([]),
                "iscrowd":  torch.tensor([]),
            }
        else:
            boxes_t  = torch.as_tensor(boxes, dtype=torch.float32)
            masks_t  = torch.stack(valid_masks)
            labels_t = torch.as_tensor(valid_labels, dtype=torch.int64)
            area     = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
            target = {
                "boxes":    boxes_t,
                "labels":   labels_t,
                "masks":    masks_t,
                "image_id": torch.tensor([idx]),
                "area":     area,
                "iscrowd":  torch.zeros(len(valid_labels), dtype=torch.int64),
            }

        return image_tensor, target

    def __len__(self):
        return len(self.samples)