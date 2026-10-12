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
# Room kept free for embedding the cached tiles, as a fraction of the cache's size. Their
# embeddings take up to about a third of it (7 conditions, all four encoders), so new tiles
# can never fill the budget in a way that leaves the cache unable to be embedded and cleared.
EMBED_HEADROOM = 0.5

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

# --- Slide-level heads -----------------------------------------------------------------
# Fixed before any test result was seen. Each seed trains on a random 80% of the training
# slides and picks its hyperparameter (C, or the epoch) on the other 20%. The split is the
# same for the clean and the mixed training set, so the two are compared pairwise.
RESULTS = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]
VAL_FRACTION = 0.2
TRAIN_SETS = ["clean", "mixed"]
HEADS = ["mean", "max", "abmil"]
# mean, max: logistic regression on the slide's mean- or max-pooled standardised embedding.
LR_C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
# abmil: gated attention MIL (Ilse et al., 2018) on standardised tile embeddings.
ABMIL = dict(hidden=256, attention=128, dropout=0.25, lr=1e-4, weight_decay=1e-4, epochs=40)
