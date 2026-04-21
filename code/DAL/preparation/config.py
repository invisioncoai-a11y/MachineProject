import os

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATASET_ZIP = os.environ.get("DATASET_ZIP", "")
EXTRACTED_DATASET_DIR = os.environ.get("EXTRACTED_DATASET_DIR", "")
REPORTS_DIR = os.environ.get(
    "REPORTS_DIR",
    os.path.join(PROJECT_ROOT, "DAL", "reports")
)

ALL_LABELS = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]

IMG_SIZE = 224
MAX_PATCHES = 6

SEED = 42
TEST_RATIO = 0.10
VAL_RATIO = 0.10
INITIAL_LABEL_RATIO = 0.05