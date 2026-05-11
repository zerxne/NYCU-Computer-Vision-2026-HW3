import math
import torch
import torch.nn.functional as TF
import json
from pathlib import Path
import cv2
import argparse
import torchvision.transforms.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm
from utils import encode_mask
from model import build_model


def load_model(model_type, weight_path, device):
    model = build_model(model_type)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    model.to(device)
    return model



def soft_nms(boxes, scores, sigma=0.5, score_thresh=0.01):
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    scores  = scores.clone().float()
    indices = torch.arange(len(scores), device=boxes.device)
    result  = []

    while indices.numel() > 0:
        best_local = scores[indices].argmax()
        best       = indices[best_local]
        result.append(best.item())

        rest = torch.cat([indices[:best_local], indices[best_local + 1:]])
        if rest.numel() == 0:
            break

        iou   = box_iou(boxes[best].unsqueeze(0), boxes[rest])[0]  
        decay = torch.exp(-(iou ** 2) / sigma)
        scores[rest] *= decay

        keep    = scores[rest] >= score_thresh
        indices = rest[keep]

    return torch.tensor(result, dtype=torch.long, device=boxes.device)


def soft_nms_classwise(boxes, scores, labels, sigma=0.5, score_thresh=0.01):
    keep_all = []
    for cls in labels.unique():
        idx  = (labels == cls).nonzero(as_tuple=False).squeeze(1)
        kept = soft_nms(boxes[idx], scores[idx], sigma=sigma, score_thresh=score_thresh)
        keep_all.append(idx[kept])
    if not keep_all:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    return torch.cat(keep_all)



@torch.no_grad()
def _infer(model, tensor, device):
    return model([tensor.to(device)])[0]



def _scale_output(out, scale, orig_H, orig_W):
    if out['boxes'].numel() == 0:
        return out
    boxes = out['boxes'].clone()
    boxes /= scale                                   
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_H)

    masks = TF.interpolate(
        out['masks'].float(),
        size=(orig_H, orig_W),
        mode='bilinear',
        align_corners=False,
    )
    return {**out, 'boxes': boxes, 'masks': masks}


def _hflip_output(out, W):
    boxes = out['boxes'].clone()
    boxes[:, 0] = W - out['boxes'][:, 2]
    boxes[:, 2] = W - out['boxes'][:, 0]
    return {**out, 'boxes': boxes, 'masks': out['masks'].flip(-1)}


def _vflip_output(out, H):
    boxes = out['boxes'].clone()
    boxes[:, 1] = H - out['boxes'][:, 3]
    boxes[:, 3] = H - out['boxes'][:, 1]
    return {**out, 'boxes': boxes, 'masks': out['masks'].flip(-2)}


def _to_cpu(out):
    return {k: v.cpu() for k, v in out.items()}



@torch.no_grad()
def predict_tta(model, image_rgb, device,
                score_threshold=0.3, soft_nms_sigma=0.5):

    H, W  = image_rgb.shape[:2]
    outputs = []

    for flip_code in [None, 1, 0]: 
        img_f = cv2.flip(image_rgb, flip_code) if flip_code is not None else image_rgb
        t     = F.to_tensor(img_f)
        out   = _to_cpu(_infer(model, t, device))
        torch.cuda.empty_cache()        

        if flip_code == 1:
            out = _hflip_output(out, W)
        elif flip_code == 0:
            out = _vflip_output(out, H)

        outputs.append(out)

    all_boxes  = torch.cat([o['boxes']  for o in outputs], dim=0)
    all_scores = torch.cat([o['scores'] for o in outputs], dim=0)
    all_labels = torch.cat([o['labels'] for o in outputs], dim=0)
    all_masks  = torch.cat([o['masks']  for o in outputs], dim=0)

    keep = soft_nms_classwise(all_boxes, all_scores, all_labels,
                               sigma=soft_nms_sigma, score_thresh=score_threshold)

    if keep.numel() == 0:
        empty = lambda: torch.zeros(0)
        return {'boxes': torch.zeros((0, 4)), 'scores': empty(),
                'labels': torch.zeros(0, dtype=torch.long),
                'masks':  torch.zeros((0, 1, H, W))}

    order = all_scores[keep].argsort(descending=True)
    keep  = keep[order]

    return {
        'boxes':  all_boxes[keep],
        'scores': all_scores[keep],
        'labels': all_labels[keep],
        'masks':  all_masks[keep],
    }



def predict(model, json_input_path, image_dir, output_json_path,
            score_threshold=0.3, soft_nms_sigma=0.5, device='cuda'):

    with open(json_input_path, 'r') as f:
        image_entries = json.load(f)

    results = []
    model.eval()

    for entry in tqdm(image_entries):
        image_path = Path(image_dir) / entry["file_name"]
        image_id   = entry["id"]
        height     = entry["height"]
        width      = entry["width"]

        image_bgr = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        output = predict_tta(model, image_rgb, device,
                             score_threshold=score_threshold,
                             soft_nms_sigma=soft_nms_sigma)

        for i in range(len(output['scores'])):
            x, y, xmax, ymax = output['boxes'][i].tolist()
            binary_mask = output['masks'][i, 0].numpy() > 0.5
            rle         = encode_mask(binary_mask)

            results.append({
                "image_id":    image_id,
                "bbox":        [x, y, xmax - x, ymax - y],
                "score":       float(output['scores'][i]),
                "category_id": int(output['labels'][i]),
                "segmentation": {
                    "size":   [height, width],
                    "counts": rle["counts"],
                },
            })

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions to {output_json_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type',     type=str, choices=['res', 'swin_s'], default='res')
    parser.add_argument('--weights',        type=str, default='best_model.pth')
    parser.add_argument('--image-dir',      type=str, default='./dataset')
    parser.add_argument('--score-thr',      type=float, default=0.3)
    parser.add_argument('--soft-nms-sigma', type=float, default=0.5,
                        help='Gaussian sigma for Soft-NMS (0 = hard NMS)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.model_type, args.weights, device)

    root = args.image_dir
    predict(
        model            = model,
        json_input_path  = f'{root}/test_image_name_to_ids.json',
        image_dir        = f'{root}/test_release',
        output_json_path = 'test-results.json',
        score_threshold  = args.score_thr,
        soft_nms_sigma   = args.soft_nms_sigma,
        device           = device,
    )


if __name__ == '__main__':
    main()