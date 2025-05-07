import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.models.swin_transformer import swin_s
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models._utils import IntermediateLayerGetter


def _make_swin_fpn_backbone(trainable_layers=5):
    # Load Swin-S backbone features
    backbone = swin_s(weights='DEFAULT').features

    for i, layer in enumerate(backbone):
        if i < len(backbone) - trainable_layers:
            for p in layer.parameters():
                p.requires_grad_(False)

    # Specify which layers to extract
    return_layers = {
        '1': '0',
        '3': '1',
        '5': '2',
        '7': '3'
    }
    in_chs = [96, 192, 384, 768]  # Swin-S (or Swin-T)
    out_ch = 256

    # Wrap backbone for feature extraction
    body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # FPN to merge multi-scale features
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_chs,
        out_channels=out_ch,
        extra_blocks=LastLevelMaxPool()
    )

    class SwinFPN(nn.Module):
        def __init__(self, body, fpn):
            super().__init__()
            self.body = body
            self.fpn = fpn
            self.out_channels = out_ch

        def forward(self, x):
            feats = self.body(x)
            # Convert [B, H, W, C] -> [B, C, H, W]
            feats = {k: v.permute(0, 3, 1, 2) for k, v in feats.items()}
            return self.fpn(feats)

    return SwinFPN(body, fpn)

def build_model(type_):
    if type_ == "res":
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
        num_classes = 5  # 4 classes + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            model.roi_heads.mask_predictor.conv5_mask.in_channels,
            256,
            num_classes
        )
        return model
    elif type_ == "swin_s":
        backbone = _make_swin_fpn_backbone(trainable_layers=5)


        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        # Create Mask R-CNN
        model = MaskRCNN(
            backbone,
            num_classes=5,
            rpn_anchor_generator=anchor_generator
        )

        # Replace heads
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

        in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels_mask, 256, 5)

        return model
    else:
        raise ValueError(f"Unsupported model type: '{type_}'. Expected 'res' or 'swin_s'.")
    
