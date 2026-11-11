"""
DINOv2 dense-feature change detection + SAM region proposals.

Adds two zero-shot, no-training-required signals to the change-detection
pipeline:

  1. dino_change_map()   - per-patch semantic dissimilarity between the
     before/after scene, upsampled to full resolution. More robust than
     raw SSIM/Lab-color to sensor calibration drift, seasonal color shift,
     and other "same content, different pixels" cases, because it compares
     learned semantic embeddings rather than raw pixel statistics.

  2. sam_change_regions() - runs Segment Anything on the 'after' scene and
     scores each resulting segment by its mean change score (DINO or the
     generic heatmap), giving region proposals that follow real object
     boundaries instead of rectangular bounding boxes from contour-finding.

Both are frozen, off-the-shelf models -- no fine-tuning, no labeled
Sentinel-2 data required. This sidesteps the BIT_CD resolution mismatch
(BIT_CD expects ~0.5m/px LEVIR-CD-style imagery; DINO/SAM don't care what
resolution you give them).

Optional deps (only imported lazily, at call time):
    pip install torch torchvision --break-system-packages
    pip install git+https://github.com/facebookresearch/segment-anything.git --break-system-packages

    SAM checkpoint (pick one -- vit_b is the lightest / most CPU-friendly):
        vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
        vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
        vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

    DINOv2 weights download automatically via torch.hub on first use
    (~90MB for dinov2_vits14).
"""
import sys

import cv2
import numpy as np

