"""
inference/model.py

Model loading + prediction, built from the VERIFIED calls in your real
test_NYU.py:

    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    load_checkpoint(model, pretrained_path=cfg.pretrained_path, prefix='student.')
    model.eval()
    pred_3d, _, _, _ = model(**data)

cfg.model.NAME == 'VoxelSSC' (from your cfgs/NYU/voxelSSC.yaml), composed of:
    encoder (mit_b2) -> head (SegFormerHead) -> feature2d, cls2d
    tsdf -> TSDFNet -> tsdf_feat
    (feature2d, mapping2d, tsdf_feat) -> Unet3d -> pred_semantic (== pred_3d)

*** UPDATE: label3d/label_weight/tsdf_CAD placeholders are now CONFIRMED safe,
not just assumed ***
Having seen TSDFNet.forward(self, tsdf), SegFormerHead.forward(self, x, ...),
and Unet3d.forward(self, feature2d, mapping2d, tsdf_feat=None) directly: none
of them reference label3d, label_weight, tsdf_CAD, or mapping anywhere. Only
img, tsdf, and mapping2d actually drive the computation graph. So whatever
VoxelSSC.forward() does with the other kwargs before calling its sub-modules,
it can't be feeding them into the actual math -- they're passed through
harmlessly (most likely just because model(**data) in test_NYU.py blindly
unpacks the whole sample dict, and the eval code needs them from `data`
afterward, not from the model's return value).

Residual, low-probability risk: I haven't seen VoxelSSC's own forward() body,
so I can't rule out an assertion/shape-check on those fields before they're
discarded. If load errors mention label3d/label_weight/tsdf_CAD specifically,
that's the first place to look.
"""

import sys
import torch

# These imports mirror test_NYU.py exactly -- make sure CleanerS's repo root
# (containing the `cleaner` package) is on sys.path before importing this module,
# e.g. by running everything from the repo root, or inserting it below.
from cleaner.models import build_model_from_cfg
from cleaner.utils import load_checkpoint


# *** NEW: image normalization, matching the real pipeline's ImgNormalize step ***
# cfgs/NYU/voxelSSC.yaml's datatransforms.val = [ImgNormalize, ToTensor] -- our
# FrameLoader/model.py were previously sending raw, unnormalized BGR pixel
# values straight to an ImageNet-pretrained encoder (mit_b2), which is almost
# certainly why predictions were collapsing onto one or two dominant classes.
#
# These constants (mean/std, BGR->RGB) are mmsegmentation's standard
# img_norm_cfg default -- a strong guess given the checkpoint-loading log
# explicitly comes from an `mmseg` logger, but NOT 100% verified against the
# actual ImgNormalize transform source. If results are still off after this
# fix, the next thing to check is cleaner/transform.py (or wherever
# ImgNormalize is defined) for the exact mean/std/to_rgb values it uses.
IMG_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
IMG_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


def _normalize_img(img_bgr_hwc):
    """img_bgr_hwc: (H,W,3) float array from cv2.imread (BGR order, 0-255).
    Returns a (3,H,W) normalized tensor, RGB order."""
    img = torch.from_numpy(img_bgr_hwc).float()
    img = img[:, :, [2, 1, 0]]                      # BGR -> RGB
    img = img.permute(2, 0, 1)                        # HWC -> CHW
    img = (img - IMG_MEAN) / IMG_STD
    return img


def load_model(cfg, device):
    """cfg: an EasyConfig object with at least cfg.model and cfg.pretrained_path
    (load it the same way test_NYU.py does -- see run_inference.py)."""
    model = build_model_from_cfg(cfg.model).to(device)
    load_checkpoint(model, pretrained_path=cfg.pretrained_path, prefix='student.')
    model.eval()
    return model


def _build_forward_kwargs(sample, device):
    """sample: dict from FrameLoader.build_tsdf_and_mapping(), with keys
    'img' (H,W,3), 'tsdf' (1,60,36,60), 'mapping' (129600,), 'mapping2d' (H,W).

    Adds a batch dimension and moves to device, and fills in placeholders for
    the training/eval-only fields we don't have (see module docstring)."""
    img = _normalize_img(sample['img']).unsqueeze(0).to(device)
    tsdf = torch.from_numpy(sample['tsdf']).unsqueeze(0).float().to(device)             # (1,1,60,36,60)
    mapping = torch.from_numpy(sample['mapping']).long().unsqueeze(0).to(device)
    mapping2d = torch.from_numpy(sample['mapping2d']).long().unsqueeze(0).to(device)

    # Placeholders -- see module docstring. label3d=0 ('empty' class), 
    # label_weight=all-ones (harmless if truly unused at inference), 
    # tsdf_CAD=same as noisy tsdf (student network's own input; the CAD branch
    # is a teacher-only training input, so this should never actually be read
    # by the student's forward pass -- confirm against your model source).
    label3d = torch.zeros_like(mapping)
    label_weight = torch.ones_like(mapping, dtype=torch.bool)
    tsdf_CAD = tsdf.clone()

    return {
        'img': img,
        'tsdf': tsdf,
        'tsdf_CAD': tsdf_CAD,
        'mapping': mapping,
        'mapping2d': mapping2d,
        'label3d': label3d,
        'label_weight': label_weight,
    }


@torch.no_grad()
def predict(model, sample, device):
    """Returns (pred_label, weight): pred_label (60,36,60) int array of argmax
    class indices, weight (60,36,60) our own observed-voxel mask (not from the
    model -- comes straight from sample, same as test_NYU.py pulling
    label_weight from the loaded data dict rather than the model's output)."""
    data = _build_forward_kwargs(sample, device)

    pred_3d, _, _, _ = model(**data)          # verified call shape from test_NYU.py
    pred_3d = pred_3d.flatten(2).permute(0, 2, 1)   # (B, 12, 60,36,60) -> (B, 129600, 12)
    pred_label = pred_3d[0].argmax(dim=1).view(60, 36, 60).cpu().numpy()

    weight = sample['weight'].reshape(60, 36, 60)   # our real TSDF observed-mask, not a placeholder
    return pred_label, weight