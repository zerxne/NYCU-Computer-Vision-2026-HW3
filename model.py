import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.models.swin_transformer import swin_s
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models._utils import IntermediateLayerGetter

NUM_CLASSES = 5 



def _make_swin_fpn_backbone(trainable_layers=5):
    backbone = swin_s(weights='DEFAULT').features

    for i, layer in enumerate(backbone):
        if i < len(backbone) - trainable_layers:
            for p in layer.parameters():
                p.requires_grad_(False)

    return_layers = {'1': '0', '3': '1', '5': '2', '7': '3'}
    in_chs = [96, 192, 384, 768]
    out_ch  = 256

    body = IntermediateLayerGetter(backbone, return_layers=return_layers)
    fpn  = FeaturePyramidNetwork(
        in_channels_list=in_chs,
        out_channels=out_ch,
        extra_blocks=LastLevelMaxPool(),
    )

    class SwinFPN(nn.Module):
        def __init__(self, body, fpn):
            super().__init__()
            self.body = body
            self.fpn  = fpn
            self.out_channels = out_ch

        def forward(self, x):
            feats = self.body(x)
            feats = {k: v.permute(0, 3, 1, 2) for k, v in feats.items()}
            return self.fpn(feats)

    return SwinFPN(body, fpn)



def _replace_heads(model, num_classes=NUM_CLASSES, mask_hidden=256):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_ch_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_ch_mask, mask_hidden, num_classes)



def freeze_backbone(model):
    frozen = 0
    for name, p in model.named_parameters():
        if 'backbone' in name:
            p.requires_grad_(False)
            frozen += p.numel()
    print(f"  Backbone frozen ({frozen/1e6:.1f}M params). Training heads only.")


def unfreeze_backbone(model):
    unfrozen = 0
    for name, p in model.named_parameters():
        if 'backbone' in name:
            p.requires_grad_(True)
            unfrozen += p.numel()
    print(f"  Backbone unfrozen ({unfrozen/1e6:.1f}M params). Full finetuning.")



def build_model(type_, min_size=800, max_size=1333):
    rpn_kwargs = dict(
        rpn_pre_nms_top_n_train  = 3000,
        rpn_pre_nms_top_n_test   = 1500,
        rpn_post_nms_top_n_train = 2000,
        rpn_post_nms_top_n_test  = 1000,
        rpn_nms_thresh           = 0.7,
        rpn_fg_iou_thresh        = 0.7,
        rpn_bg_iou_thresh        = 0.3,
    )
    roi_kwargs = dict(
        box_score_thresh       = 0.01,
        box_nms_thresh         = 0.4,
        box_detections_per_img = 300,
    )
    size_kwargs = dict(min_size=min_size, max_size=max_size)

    if type_ == "res":
        model = maskrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            **rpn_kwargs,
            **roi_kwargs,
            **size_kwargs,
        )
        _replace_heads(model)
        return model

    elif type_ == "swin_s":
        backbone = _make_swin_fpn_backbone(trainable_layers=5)

        anchor_sizes  = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_gen    = AnchorGenerator(anchor_sizes, aspect_ratios)

        model = MaskRCNN(
            backbone,
            num_classes=NUM_CLASSES,
            rpn_anchor_generator=anchor_gen,
            **rpn_kwargs,
            **roi_kwargs,
            **size_kwargs,
        )
        _replace_heads(model)
        return model

    else:
        raise ValueError(f"Unsupported model type: '{type_}'. Expected 'res' or 'swin_s'.")