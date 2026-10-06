"""Settings shared by every script. Change them here, not in the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Storage ---------------------------------------------------------------------------
# Everything the pipeline downloads or writes lives under data/ (ignored by git).
DATA = ROOT / "data"
SLIDES = DATA / "slides"          # one slide at a time, deleted as soon as it is tiled
CACHE = DATA / "cache"            # raw tile pixels, deleted once every encoder has run
TILES = DATA / "tiles"            # per-slide tile coordinates, labels, thumbnail (kept, small)
EMBEDDINGS = DATA / "embeddings"  # per-encoder, per-slide embeddings (kept)
ANNOTATIONS = DATA / "annotations"
MANIFEST = ROOT / "slides.csv"

# Hard cap on the size of the whole project folder. No script writes past it.
BUDGET_GB = 50
# Tile pixels held on disk between tiling and embedding, per batch of slides.
CACHE_GB = 8
# Stop if the drive itself has less than this free, whatever the budget says.
DISK_FLOOR_GB = 2

# --- Source ----------------------------------------------------------------------------
# Public bucket from the Registry of Open Data on AWS (no account needed).
BUCKET_URL = "https://camelyon-dataset.s3.us-west-2.amazonaws.com"
PREFIX = "CAMELYON16/"

# --- Tiling ----------------------------------------------------------------------------
TILE = 224           # tile side in pixels at the working magnification
DOWNSAMPLE = 4       # relative to level 0 (40x, about 0.24 um/px), so about 10x, 0.97 um/px
MASK_DOWNSAMPLE = 32  # resolution of the tissue and tumour masks, relative to level 0
MIN_TISSUE = 0.25    # keep a tile if at least this fraction of it is tissue

# --- Conditions ------------------------------------------------------------------------
# Test slides are embedded once per condition. jpeg:10 is never seen in training.
TEST_CONDITIONS = ["clean", "blur:2", "blur:4", "jpeg:30", "jpeg:15", "jpeg:10", "stain:0.3"]
# Training slides are embedded clean, and once more as "mixed": every tile is drawn at
# random from these, as in encoder-or-probe, so both training sets are the same size.
MIX_CONDITIONS = ["clean", "blur:2", "blur:4", "jpeg:30", "jpeg:15", "stain:0.3"]

# --- Encoders --------------------------------------------------------------------------
# name -> (library, hub id, environment). The two environments are the ones built for
# encoder-or-probe; their interpreter paths are read from envs.json (see envs.example.json).
ENCODERS = {
    "phikon": ("hf", "owkin/phikon", "main"),
    "phikon-v2": ("hf", "owkin/phikon-v2", "main"),
    "dinov2-l": ("hf", "facebook/dinov2-large", "main"),
    "kaiko-b16": ("timm", "hf_hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms", "timm"),
}
