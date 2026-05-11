import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json, os, math, argparse
import numpy as np                                     
import matplotlib                                     
matplotlib.use('Agg')                              
import matplotlib.pyplot as plt                 
import matplotlib.ticker as ticker              
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



def box_iou(box_a, box_b):
    xa = max(box_a[0], box_b[0]); ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2]); yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0



def accumulate_confusion(targets, predictions, cm, num_classes,
                         iou_thresh=0.5, score_thresh=0.3):

    bg = num_classes   
    for t, p in zip(targets, predictions):
        gt_boxes  = t["boxes"].cpu().numpy()
        gt_labels = t["labels"].cpu().numpy()

        keep = p["scores"].cpu().numpy() >= score_thresh
        pd_boxes  = p["boxes"].cpu().numpy()[keep]
        pd_labels = p["labels"].cpu().numpy()[keep]

        matched_gt  = set()
        matched_pd  = set()

        for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            best_iou, best_pi = 0.0, -1
            for pi, (pb, pl) in enumerate(zip(pd_boxes, pd_labels)):
                iou = box_iou(gb, pb)
                if iou > best_iou:
                    best_iou, best_pi = iou, pi
            if best_iou >= iou_thresh and best_pi not in matched_pd:
                pred_cls = int(pd_labels[best_pi]) - 1  
                gt_cls   = int(gl) - 1
                cm[gt_cls, pred_cls] += 1
                matched_gt.add(gi)
                matched_pd.add(best_pi)
            else:
                cm[int(gl) - 1, bg] += 1

        for pi in range(len(pd_boxes)):
            if pi not in matched_pd:
                cm[bg, int(pd_labels[pi]) - 1] += 1



def evaluate_map(model, data_loader, device,
                 cm=None, num_classes=4):          
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

        if cm is not None:
            accumulate_confusion(targets, outputs, cm, num_classes)

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



def plot_training_curves(history, num_classes, stage2_start, swa_start, out_dir="plots"):

    os.makedirs(out_dir, exist_ok=True)
    epochs   = list(range(1, len(history["loss"]) + 1))
    stage    = history["epoch_stage"]  

    def add_stage_bands(ax, n_epochs):
        s2 = stage2_start
        sw = swa_start + 1
        ne = n_epochs + 1
        ax.axvspan(1,    s2,   alpha=0.06, color="#4e79a7", label="Stage 1 (frozen)")
        ax.axvspan(s2,   sw,   alpha=0.06, color="#f28e2b", label="Stage 2")
        ax.axvspan(sw,   ne,   alpha=0.06, color="#e15759", label="SWA")
        ax.axvline(s2,   color="#4e79a7", lw=1.2, ls="--", alpha=0.6)
        ax.axvline(sw,   color="#e15759", lw=1.2, ls="--", alpha=0.6)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    add_stage_bands(ax, len(epochs))
    ax.plot(epochs, history["loss"], color="#2d6a4f", lw=2.2,
            marker="o", markersize=3.5, label="Train Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    loss_path = os.path.join(out_dir, "training_loss_curve.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ saved {loss_path}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    add_stage_bands(ax, len(epochs))
    ax.plot(epochs, history["ap50"],  color="#1565c0", lw=2.2,
            marker="o", markersize=3.5, label="AP@50")
    ax.plot(epochs, history["ap_all"], color="#6a1b9a", lw=2.2,
            marker="s", markersize=3.5, ls="--", label="AP@50:95")
    best_ep  = int(np.argmax(history["ap50"])) + 1
    best_val = max(history["ap50"])
    ax.axvline(best_ep, color="#1565c0", lw=1.2, ls=":", alpha=0.7)
    ax.annotate(f"best {best_val:.3f}", xy=(best_ep, best_val),
                xytext=(best_ep + 0.5, best_val - 0.03),
                fontsize=8, color="#1565c0",
                arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1))
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP", fontsize=12)
    ax.set_title("Validation Performance (mAP)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    map_path = os.path.join(out_dir, "validation_map_curve.png")
    fig.savefig(map_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ saved {map_path}")

    cm = history["confusion_matrix"]   
    labels = [f"class_{i+1}" for i in range(num_classes)] + ["BG/FP"]

    # normalise rows (GT) so each cell shows recall-like fraction
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm / row_sum, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(labels)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=13, fontweight="bold")

    thresh = 0.5
    for r in range(n):
        for c in range(n):
            val_raw  = int(cm[r, c])
            val_norm = cm_norm[r, c]
            color    = "white" if val_norm > thresh else "black"
            ax.text(c, r, f"{val_norm:.2f}\n({val_raw})",
                    ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ saved {cm_path}")



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
    parser.add_argument('--num-classes',  type=int,   default=4,          
                        help='Number of foreground classes (excluding background)')
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

    history = {
        "loss":             [],
        "ap50":             [],
        "ap_all":           [],
        "epoch_stage":      [],
        "confusion_matrix": np.zeros((args.num_classes + 1,
                                      args.num_classes + 1), dtype=np.int64),
    }

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

            with autocast():                     
                loss_dict = model(images, targets)
                losses    = sum(loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(losses).backward()         
            scaler.unscale_(optimizer)                
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

        is_last_epoch = (epoch == args.epochs - 1)
        cm_this_epoch = history["confusion_matrix"] if is_last_epoch else None

        ap50, ap_all = evaluate_map(model, val_loader, device,
                                    cm=cm_this_epoch,
                                    num_classes=args.num_classes)  

        avg_loss = running_loss / len(train_loader)
        cur_lr   = optimizer.param_groups[0]['lr']
        stage    = "[S2-SWA]" if in_swa else ("[S2]" if epoch >= args.stage2_start else "[S1]")
        print(f"Epoch {epoch+1:3d} {stage} | loss={avg_loss:.4f} | "
              f"AP@50={ap50:.4f} | AP@50:95={ap_all:.4f} | lr={cur_lr:.2e}")

        history["loss"].append(avg_loss)
        history["ap50"].append(ap50)
        history["ap_all"].append(ap_all)
        history["epoch_stage"].append(stage)

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

    cm_swa = np.zeros((args.num_classes + 1, args.num_classes + 1), dtype=np.int64)
    ap50_swa, _ = evaluate_map(swa_model, val_loader, device,
                               cm=cm_swa, num_classes=args.num_classes)

    torch.save(swa_model.module.state_dict(), 'swa_model.pth')
    print(f"SWA AP@50={ap50_swa:.4f}  |  Best single-epoch AP@50={best_ap50:.4f}")
    winner = 'swa_model.pth' if ap50_swa > best_ap50 else 'best_model.pth'
    print(f"→ Use  {winner}  for inference.")

    if ap50_swa > best_ap50:
        history["confusion_matrix"] = cm_swa

    print("\nGenerating plots …")
    plot_training_curves(
        history       = history,
        num_classes   = args.num_classes,
        stage2_start  = args.stage2_start,
        swa_start     = swa_start,
        out_dir       = "plots",
    )
    print("Done. Plots saved to ./plots/")


if __name__ == '__main__':
    main()
    
