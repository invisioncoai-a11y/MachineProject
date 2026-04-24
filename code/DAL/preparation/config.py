import os

# =========================
# Project paths
# =========================
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATASET_ZIP = os.environ.get("DATASET_ZIP", "")
EXTRACTED_DATASET_DIR = os.environ.get("EXTRACTED_DATASET_DIR", "")

REPORTS_DIR = os.environ.get(
    "REPORTS_DIR",
    os.path.join(PROJECT_ROOT, "DAL", "reports")
)

ANNOTATIONS_DIR = os.path.join(REPORTS_DIR, "annotations")

# =========================
# Dataset labels
# =========================
ALL_LABELS = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]

# =========================
# Image / patch settings
# =========================
IMG_SIZE = 224
MAX_PATCHES = 6
MIN_LESION_AREA_RATIO = 0.002
LESION_CONTEXT_PAD_RATIO = 0.08

# مهم: بالاتجاه الجديد لا نحفظ crops لكل الداتاست
SAVE_LESION_CROPS = False
USE_SAVED_LESION_CROPS = False

# =========================
# Reproducibility
# =========================
SEED = 42

# =========================
# Image-level split ratios
# =========================
TEST_RATIO = 0.10
VAL_RATIO = 0.10
INITIAL_LABEL_RATIO = 0.05

# =========================
# Training
# =========================
BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5

# =========================
# Active Learning
# =========================
NUM_ROUNDS = 5
QUERY_BATCH_SIZE = 200
TRAIN_ON_SEED_ONLY_AT_ROUND0 = True

ALPHA_UNCERTAINTY = 1.0
BETA_DIVERSITY = 1.0
GAMMA_CLASS_BALANCE = 1.0

USE_CLASS_BALANCED_SAMPLING = True

# =========================
# Interpretability / outputs
# =========================
INTERPRETABILITY_SAMPLE_COUNT = 15