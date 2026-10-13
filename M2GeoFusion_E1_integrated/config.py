from pathlib import Path

# ============================================================
# M²GeoFusion E1 shared defaults
# Normally you only edit demo_all.py. These are fallback defaults
# for running an individual stage manually.
# ============================================================

VGGT_ROOT = Path("/home/wqy/workspace/vggt")
VGGT_CHECKPOINT = VGGT_ROOT / "checkpoints/model.pt"

GROUNDED_SAM_ROOT = Path("/home/wqy/workspace/Grounded-SAM-2")
SAM2_CHECKPOINT = GROUNDED_SAM_ROOT / "checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = GROUNDED_SAM_ROOT / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = GROUNDED_SAM_ROOT / "gdino_checkpoints/groundingdino_swint_ogc.pth"

TEXT_PROMPT = "wire harness. pipe."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DILATE_RADIUS = 20

VGGT_PREPROCESS_MODE = "crop"
ALIGNMENT_MODE = "scale"  # scale | sim3
DEDUP_FACTOR = 1.5

GROUNDED_SAM_ENV = "grounded_sam2"
VGGT_ENV = "vggt"

# Track visualization defaults
TRACK_MAX_POINTS = 300
TRACK_VIS_THRESHOLD = 0.50
TRACK_CONF_THRESHOLD = 0.50

# Joint-model filtering defaults (demo_all.py can override these).
DIFF_FACTOR = 1.5
SIM3_MAD_FACTOR = 3.5
SIM3_ITERATIONS = 3
SIM3_MAX_POINTS = 100000
SEMANTIC_MASK_KIND = "target"
KEEP_XD_JOINT_LG = True
SHAPE_K = 16
SHAPE_MAD_FACTOR = 3.5
SHAPE_FILTER_ITERATIONS = 1
KNN_CHUNK_SIZE = 20000
