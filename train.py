import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json, os, math, argparse
from preprocess import InstanceSegmentationDataset
from model import build_model, freeze_backbone, unfreeze_backbone
from torch.cuda.amp import GradScaler, autocast


def convert_to_coco_dict(targets, image_ids):
    annotations, images = [], []
    ann_id = 1
    for img_id, t in zip(image_ids, targets):
        h, w = t["masks"].shape[1:]
        images.append({"id": img_id, "height": h, "width": w})
        for i in range(len(t["boxes"])):
            x1, y1, x2, y2 = t["boxes"][i].tolist()
            bbox = [x1, y1, x2 - x1, y2 - y1]
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": int(t["labels"][i]),
                "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
            })
            ann_id += 1
    return {
        "images": images, "annotations": annotations,
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(1, 5)],
    }


def convert_predictions_to_coco(predictions, image_ids):
    results = []
    for img_id, p in zip(image_ids, predictions):
        for i in range(len(p["boxes"])):
            x1, y1, x2, y2 = p["boxes"][i].tolist()
            results.append({
                "image_id": img_id, "category_id": int(p["labels"][i]),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(p["scores"][i]),
            })
    return results


def evaluate_map(model, data_loader, device):
    model.eval()
    coco_targets, coco_preds, image_ids = [], [], []
    for i, (images, targets) in enumerate(tqdm(data_loader, desc="Eval", leave=False)):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)
        img_id = i + 1
        for t in targets:
            t["image_id"] = torch.tensor(img_id)
        image_ids.append(img_id)
        coco_targets.extend(targets)
        coco_preds.extend(outputs)

    gt_dict   = convert_to_coco_dict(coco_targets, image_ids)
    pred_dict = convert_predictions_to_coco(coco_preds, image_ids)
    os.makedirs("tmp_eval", exist_ok=True)
    with open("tmp_eval/gt.json",   "w") as f: json.dump(gt_dict,   f)
    with open("tmp_eval/pred.json", "w") as f: json.dump(pred_dict, f)

    coco_gt   = COCO("tmp_eval/gt.json")
    coco_dt   = coco_gt.loadRes("tmp_eval/pred.json")
    ev = COCOeval(coco_gt, coco_dt, iouType='bbox')
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[1], ev.stats[0]



def cosine_lambda(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    p = (epoch - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir',    type=str,   required=True)
    parser.add_argument('--model-type',   type=str,   choices=['res', 'swin_s'], default='res')
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch-size',   type=int,   default=2)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--min-size',     type=int,   default=800,
                        help='Shorter-side input resolution (try 1024 for high-res data)')
    parser.add_argument('--stage2-start', type=int,   default=10,
                        help='Epoch to unfreeze backbone (0 = no freezing)')
    parser.add_argument('--swa-lr',       type=float, default=5e-6)
    args = parser.parse_args()

    root    = args.image_dir
    collate = lambda x: tuple(zip(*x))

    tr_full = InstanceSegmentationDataset(f'{root}/train', augment=True)
    va_full = InstanceSegmentationDataset(f'{root}/train', augment=False)
    n_val   = int(0.1 * len(tr_full))
    n_train = len(tr_full) - n_val
    idx     = list(range(len(tr_full)))

    tr_set  = Subset(tr_full, idx[:n_train])
    va_set  = Subset(va_full, idx[n_train:])
    bn_set  = Subset(InstanceSegmentationDataset(f'{root}/train', augment=False),
                     idx[:n_train])

    train_loader = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(va_set, batch_size=1, shuffle=False,
                              collate_fn=collate, num_workers=2)
    bn_loader    = DataLoader(bn_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  model: {args.model_type}  |  "
          f"min_size: {args.min_size}  |  train: {n_train}  val: {n_val}")

    model = build_model(args.model_type, min_size=args.min_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params/1e6:.1f}M")

    if args.stage2_start > 0:
        freeze_backbone(model)

    def make_optimizer(model, lr):
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    stage1_lr = args.lr * 5 if args.stage2_start > 0 else args.lr
    optimizer = make_optimizer(model, stage1_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: cosine_lambda(ep, warmup=3, total=args.epochs))

    swa_start = int(0.75 * args.epochs)
    swa_model = AveragedModel(model)
    scaler    = GradScaler()  
    print(f"Stage 1 (frozen backbone): epochs 1–{args.stage2_start}")
    print(f"Stage 2 (full finetune) : epochs {args.stage2_start+1}–{args.epochs}")
    print(f"SWA starts              : epoch {swa_start+1}")

    best_ap50 = 0.0
    in_swa    = False

    for epoch in range(args.epochs):

        if epoch == args.stage2_start and args.stage2_start > 0:
            unfreeze_backbone(model)
            optimizer = make_optimizer(model, args.lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda ep: cosine_lambda(ep, warmup=2, total=args.epochs - args.stage2_start))
            print(f"  → Switched to full-model optimizer (lr={args.lr})")

        if epoch == swa_start and not in_swa:
            in_swa = True
            for pg in optimizer.param_groups:
                pg['lr'] = args.swa_lr
            print(f"\n── SWA phase (lr={args.swa_lr}) ──")

        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for images, targets in loop:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast():                          # fp16 forward pass
                loss_dict = model(images, targets)
                losses    = sum(loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(losses).backward()           # scaled backward
            scaler.unscale_(optimizer)                # unscale before clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += losses.item()
            loop.set_postfix(loss=f"{losses.item():.4f}")

        if not in_swa:
            scheduler.step()

        if in_swa:
            swa_model.update_parameters(model)

        ap50, ap_all = evaluate_map(model, val_loader, device)
        cur_lr = optimizer.param_groups[0]['lr']
        stage  = "[S2-SWA]" if in_swa else ("[S2]" if epoch >= args.stage2_start else "[S1]")
        print(f"Epoch {epoch+1:3d} {stage} | loss={running_loss/len(train_loader):.4f} | "
              f"AP@50={ap50:.4f} | AP@50:95={ap_all:.4f} | lr={cur_lr:.2e}")

        if ap50 > best_ap50:
            best_ap50 = ap50
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ best_model.pth  (AP50={best_ap50:.4f})")

        torch.cuda.empty_cache()

    print("\nUpdating BN stats for SWA model …")

    def update_bn_detection(loader, swa_model, device):
        swa_model.eval()
        bn_layers = [m for m in swa_model.modules()
                     if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm))]
        for bn in bn_layers:
            bn.train()
            bn.reset_running_stats()
        with torch.no_grad():
            for images, _ in tqdm(loader, desc="BN update", leave=False):
                swa_model([img.to(device) for img in images])
        swa_model.eval()

    update_bn_detection(bn_loader, swa_model, device)

    ap50_swa, _ = evaluate_map(swa_model, val_loader, device)
    torch.save(swa_model.module.state_dict(), 'swa_model.pth')
    print(f"SWA AP@50={ap50_swa:.4f}  |  Best single-epoch AP@50={best_ap50:.4f}")
    winner = 'swa_model.pth' if ap50_swa > best_ap50 else 'best_model.pth'
    print(f"→ Use  {winner}  for inference.")


if __name__ == '__main__':
    main()