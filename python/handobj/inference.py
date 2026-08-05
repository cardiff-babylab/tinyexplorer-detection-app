"""HandObject Faster R-CNN inference (hands-only).

Encapsulates model construction and single-image hand detection, adapted from
the prototype ``RunHandDetection.py`` (100DOH hand-object detector, res101,
TinyExplorer-tuned). Object detection and hand-object pairing are intentionally
omitted — this build reports hands with per-hand contact state, left/right side
and (for the Tuned checkpoint) own/other ownership.

Importing this module requires ``torch`` and the vendored native extension
``model._C`` (see ``handobj/lib/model``); import it lazily from callers so a
face-only environment does not pay for it.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# The vendored ``model`` package expects ``handobj/lib`` on sys.path (mirrors the
# prototype's _init_paths.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_THIS_DIR, "lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

import torch  # noqa: E402  (after sys.path setup)

from model.faster_rcnn.resnet import resnet  # noqa: E402
from model.roi_layers import nms  # noqa: E402
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes  # noqa: E402
from model.utils.blob import im_list_to_blob  # noqa: E402
from model.utils.config import cfg, cfg_from_file  # noqa: E402

_CFG_FILE = os.path.join(_THIS_DIR, "cfgs", "res101.yml")

# Detection classes for the model; index 2 == "hand".
_PASCAL_CLASSES = np.asarray(["__background__", "targetobject", "hand"])
_HAND_CLASS_IDX = 2

# Contact-state code -> human label (matches the prototype's frame summary).
STATE_LABELS: Dict[int, str] = {
    0: "none",
    1: "self",
    2: "other",
    3: "portable",
    4: "furniture",
}

# Probability above which the ownership head marks a hand as "own".
OWN_IF_PROB = 0.5
# At most this many hands may be labelled "own" per frame (prototype rule).
MAX_OWN_PER_FRAME = 2


@dataclass
class HandDetection:
    """One hand detection in original-image pixel coordinates (x1,y1,x2,y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    state: int
    state_label: str
    side: str  # "left" | "right"
    owner: str  # "own" | "other" | "unknown"


def _get_image_blob(im: np.ndarray):
    """Convert a BGR image into the network input blob (lifted from prototype)."""
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        import cv2

        im_resized = cv2.resize(
            im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        im_scale_factors.append(im_scale)
        processed_ims.append(im_resized)

    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors, dtype=np.float32)


def _enforce_two_own_limit(
    hand_dets: np.ndarray, own_prob_thresh: float, max_own: int = MAX_OWN_PER_FRAME
) -> np.ndarray:
    """Return a boolean mask of hands considered "own", capped at ``max_own``."""
    if hand_dets is None or hand_dets.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    own_probs = hand_dets[:, 10].astype(np.float32)
    is_own = own_probs >= float(own_prob_thresh)

    idx = np.where(is_own)[0]
    if idx.size <= max_own:
        return is_own

    top = idx[np.argsort(-own_probs[idx])[:max_own]]
    out = np.zeros_like(is_own, dtype=bool)
    out[top] = True
    return out


class HandObjectModel:
    """Loads a HandObject checkpoint once and detects hands per image."""

    def __init__(
        self,
        weights_path: str,
        has_ownership: bool,
        use_cuda: Optional[bool] = None,
    ) -> None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"HandObject weights not found: {weights_path}")

        cfg_from_file(_CFG_FILE)
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        cfg.USE_GPU_NMS = self.use_cuda
        np.random.seed(cfg.RNG_SEED)

        self.has_ownership = has_ownership

        net = resnet(_PASCAL_CLASSES, 101, pretrained=False, class_agnostic=False)
        net.create_architecture()

        checkpoint = torch.load(
            weights_path, map_location=("cuda" if self.use_cuda else "cpu")
        )
        net.load_state_dict(checkpoint["model"], strict=False)
        if "pooling_mode" in checkpoint:
            cfg.POOLING_MODE = checkpoint["pooling_mode"]

        if self.use_cuda:
            net.cuda()
        net.eval()
        self.net = net

    @torch.no_grad()
    def detect(
        self,
        im_bgr: np.ndarray,
        thresh_hand: float = 0.5,
        own_prob_thresh: float = OWN_IF_PROB,
    ) -> List[HandDetection]:
        """Detect hands in a single BGR image; returns a list of HandDetection."""
        blobs, im_scales = _get_image_blob(im_bgr)
        assert len(im_scales) == 1, "Only single-scale testing is supported"

        im_info_np = np.array(
            [[blobs.shape[1], blobs.shape[2], im_scales[0]]], dtype=np.float32
        )
        im_data = torch.from_numpy(blobs).permute(0, 3, 1, 2)
        im_info = torch.from_numpy(im_info_np)
        gt_boxes = torch.zeros(1, 1, 5)
        num_boxes = torch.zeros(1)
        box_info = torch.zeros(1, 1, 6)

        if self.use_cuda:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            gt_boxes = gt_boxes.cuda()
            num_boxes = num_boxes.cuda()
            box_info = box_info.cuda()

        (
            rois,
            cls_prob,
            bbox_pred,
            _rpn_loss_cls,
            _rpn_loss_box,
            _rcnn_loss_cls,
            _rcnn_loss_bbox,
            _rois_label,
            loss_list,
        ) = self.net(im_data, im_info, gt_boxes, num_boxes, box_info)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        contact_logits = loss_list[0][0]
        offset_vec = loss_list[1][0].detach()
        lr_logits = loss_list[2][0].detach()

        _, contact_idx = torch.max(contact_logits, 2)
        contact_idx = contact_idx.squeeze(0).unsqueeze(-1).float()
        contact_probs = torch.softmax(contact_logits.detach(), dim=2).squeeze(0)

        lr_prob = torch.sigmoid(lr_logits).squeeze(0).view(-1, 1)

        if self.has_ownership:
            own_logits = loss_list[3][0].detach()
            own_prob = (1.0 - torch.sigmoid(own_logits)).squeeze(0).view(-1, 1)
        else:
            own_prob = torch.full_like(lr_prob, -1.0)

        # Box regression (adapted from 100DOH demo).
        if cfg.TEST.BBOX_REG:
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
                means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                if self.use_cuda:
                    stds, means = stds.cuda(), means.cuda()
                box_deltas = box_deltas.view(-1, 4) * stds + means
                box_deltas = box_deltas.view(1, -1, 4 * len(_PASCAL_CLASSES))
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            pred_boxes = boxes

        pred_boxes /= im_scales[0]

        scores_s = scores.squeeze(0)
        pred_s = pred_boxes.squeeze(0)

        j = _HAND_CLASS_IDX
        inds = torch.nonzero(scores_s[:, j] > thresh_hand).view(-1)
        if inds.numel() <= 0:
            return []

        cls_scores = scores_s[:, j][inds]
        _, order = torch.sort(cls_scores, 0, True)
        cls_boxes_ordered = pred_s[inds][:, j * 4 : (j + 1) * 4][order]
        cls_scores_ordered = cls_scores[order]

        keep = nms(cls_boxes_ordered, cls_scores_ordered, cfg.TEST.NMS).view(-1).long()
        sel = inds[order][keep]

        hand_dets = torch.cat(
            (
                pred_s[sel][:, j * 4 : (j + 1) * 4],
                scores_s[sel, j].unsqueeze(1),
                contact_idx[sel],
                offset_vec.squeeze(0)[sel],  # mag, dx, dy
                lr_prob[sel],
                own_prob[sel],
                contact_probs[sel],  # per-state probabilities 0..4
            ),
            dim=1,
        ).cpu().numpy()

        if self.has_ownership:
            owner_is_own = _enforce_two_own_limit(hand_dets, own_prob_thresh)
        else:
            owner_is_own = np.zeros((hand_dets.shape[0],), dtype=bool)

        results: List[HandDetection] = []
        for hid in range(hand_dets.shape[0]):
            h = hand_dets[hid]
            x1, y1, x2, y2 = (float(v) for v in h[:4])
            state = int(h[5])
            lr_p = float(h[9])
            side = "right" if lr_p >= 0.5 else "left"
            if self.has_ownership:
                owner = "own" if owner_is_own[hid] else "other"
            else:
                owner = "unknown"
            results.append(
                HandDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(h[4]),
                    state=state,
                    state_label=STATE_LABELS.get(state, f"state{state}"),
                    side=side,
                    owner=owner,
                )
            )
        return results
