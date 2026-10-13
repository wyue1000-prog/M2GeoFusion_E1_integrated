from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert

from config import (
    GROUNDED_SAM_ROOT,
    SAM2_CHECKPOINT,
    SAM2_MODEL_CONFIG,
    GROUNDING_DINO_CONFIG,
    GROUNDING_DINO_CHECKPOINT,
    TEXT_PROMPT,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    DILATE_RADIUS,
)
from common import ensure_dir, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=TEXT_PROMPT)
    ap.add_argument("--dilate-radius", type=int, default=DILATE_RADIUS)
    ap.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    ap.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    ap.add_argument("--grounded-sam-root", default=str(GROUNDED_SAM_ROOT))
    ap.add_argument("--sam2-checkpoint", default=str(SAM2_CHECKPOINT))
    ap.add_argument("--sam2-model-config", default=SAM2_MODEL_CONFIG)
    ap.add_argument("--grounding-dino-config", default=str(GROUNDING_DINO_CONFIG))
    ap.add_argument("--grounding-dino-checkpoint", default=str(GROUNDING_DINO_CHECKPOINT))
    args = ap.parse_args()

    lg_path = Path(args.lg).expanduser().resolve()
    out_dir = ensure_dir(args.out)
    grounded_root = Path(args.grounded_sam_root).expanduser().resolve()
    sam2_checkpoint = Path(args.sam2_checkpoint).expanduser().resolve()
    dino_config = Path(args.grounding_dino_config).expanduser().resolve()
    dino_checkpoint = Path(args.grounding_dino_checkpoint).expanduser().resolve()

    for p in [lg_path, grounded_root, sam2_checkpoint, dino_config, dino_checkpoint]:
        if not p.exists():
            raise FileNotFoundError(p)

    # Delayed imports: demo_all.py can provide all local paths centrally.
    sys.path.insert(0, str(grounded_root))
    sys.path.insert(0, str(grounded_root / "grounding_dino"))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

    prompt = args.prompt.strip().lower()
    if not prompt.endswith("."):
        prompt += "."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[E1-01] device = {device}")
    print(f"[E1-01] prompt = {prompt}")

    sam2_model = build_sam2(args.sam2_model_config, str(sam2_checkpoint), device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    grounding_model = load_model(
        model_config_path=str(dino_config),
        model_checkpoint_path=str(dino_checkpoint),
        device=device,
    )

    image_source, image = load_image(str(lg_path))
    sam2_predictor.set_image(image_source)
    boxes, confidences, class_names = predict(
        model=grounding_model,
        image=image,
        caption=prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )

    h, w = image_source.shape[:2]
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("Grounding DINO 没有检测到目标，请调整 prompt 或 threshold。")

    boxes_px = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype)
    input_boxes = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with torch.no_grad(), amp_ctx:
        masks, sam_scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    masks = masks.astype(bool)
    target_mask = np.any(masks, axis=0).astype(np.uint8)

    radius = max(int(args.dilate_radius), 0)
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        need_mask = (cv2.dilate(target_mask * 255, kernel) > 0).astype(np.uint8)
    else:
        need_mask = target_mask.copy()

    cv2.imwrite(str(out_dir / "target_mask.png"), target_mask * 255)
    cv2.imwrite(str(out_dir / "need_mask.png"), need_mask * 255)
    np.save(out_dir / "need_mask.npy", need_mask.astype(np.uint8))
    np.save(out_dir / "target_mask.npy", target_mask.astype(np.uint8))

    img_bgr = cv2.imread(str(lg_path), cv2.IMREAD_COLOR)
    overlay = img_bgr.copy()
    target_bool = target_mask.astype(bool)
    ring_bool = need_mask.astype(bool) & ~target_bool
    green = np.array([0, 255, 0], dtype=np.uint8)
    yellow = np.array([0, 255, 255], dtype=np.uint8)
    overlay[target_bool] = (0.55 * overlay[target_bool] + 0.45 * green).astype(np.uint8)
    overlay[ring_bool] = (0.65 * overlay[ring_bool] + 0.35 * yellow).astype(np.uint8)
    cv2.imwrite(str(out_dir / "overlay.jpg"), overlay)

    conf_list = confidences.detach().cpu().numpy().tolist()
    sam_score_list = np.asarray(sam_scores).reshape(-1).tolist()
    det = []
    for i, (name, conf, box) in enumerate(zip(class_names, conf_list, input_boxes.tolist())):
        det.append({
            "class_name": str(name),
            "grounding_confidence": float(conf),
            "sam_score": float(sam_score_list[i]) if i < len(sam_score_list) else None,
            "bbox_xyxy": [float(x) for x in box],
        })

    save_json(out_dir / "mask_info.json", {
        "lg_path": str(lg_path),
        "prompt": prompt,
        "image_width": int(w),
        "image_height": int(h),
        "dilate_radius_px": radius,
        "target_pixel_count": int(target_mask.sum()),
        "need_pixel_count": int(need_mask.sum()),
        "detections": det,
    })

    print(f"[E1-01] detections = {len(det)}")
    print(f"[E1-01] target pixels = {int(target_mask.sum())}")
    print(f"[E1-01] need pixels   = {int(need_mask.sum())}")
    print(f"[E1-01] saved to: {out_dir}")


if __name__ == "__main__":
    main()
