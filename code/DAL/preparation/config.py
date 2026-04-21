import os

# =========================
# Project paths
# =========================
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATASET_ZIP = os.environ.get("DATASET_ZIP", "")
EXTRACTED_DATASET_DIR = os.environ.get("EXTRACTED_DATASET_DIR", "")

# حسب شجرة مشروعك الحالية:
# code/
#   DAL/
#     reports/
#       annotations/
REPORTS_DIR = os.environ.get(
    "REPORTS_DIR",
    os.path.join(PROJECT_ROOT, "DAL", "reports")
)

ANNOTATIONS_DIR = os.path.join(REPORTS_DIR, "annotations")
LESIONS_DIR = os.path.join(REPORTS_DIR, "lesions")
LESION_MANIFEST_DIR = os.path.join(LESIONS_DIR, "manifests")
LESION_CROPS_DIR = os.path.join(LESIONS_DIR, "crops")
LESION_METADATA_DIR = os.path.join(LESIONS_DIR, "metadata")

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
# Image / lesion settings
# =========================
IMG_SIZE = 224
MAX_PATCHES = 6
MIN_LESION_AREA_RATIO = 0.002

# إذا بدك تحفظ crops الناتجة من lesion manifest
SAVE_LESION_CROPS = True

# إذا True، الـ data_loader يقرأ crop_path مباشرة إذا موجود
USE_SAVED_LESION_CROPS = True

# نسبة padding حول صندوق الـ lesion عند القص
LESION_CONTEXT_PAD_RATIO = 0.08

# =========================
# Reproducibility
# =========================
SEED = 42

# =========================
# Image-level split ratios
# أولًا split على مستوى الصور
# بعدها بنبني lesion-level pools
# =========================
TEST_RATIO = 0.10
VAL_RATIO = 0.10
INITIAL_LABEL_RATIO = 0.05

# =========================
# Training
# =========================
BATCH_SIZE = 32
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

# في أول جولة نبدأ من seed set فقط
TRAIN_ON_SEED_ONLY_AT_ROUND0 = True

# score = alpha * U + beta * D + gamma * B
ALPHA_UNCERTAINTY = 1.0
BETA_DIVERSITY = 1.0
GAMMA_CLASS_BALANCE = 1.0

# =========================
# Annotation / manifest behavior
# =========================
# للنسخة الأولى القابلة للتشغيل:
# نستخدم image-level primary label كـ weak bootstrap
# للـ seed lesions
ASSIGN_SEED_ANNOTATIONS = True

# نسمح لـ val/test يكون عندهم weak labels مبدئيًا
# حتى يشتغل الـ pipeline كامل قبل ما يصير عنا lesion annotations حقيقية
ANNOTATE_EVAL_SPLITS_WITH_WEAK_LABELS = True

# إذا False، فقط الصور single-label تأخذ weak lesion labels
ALLOW_MULTILABEL_SEED_ANNOTATION = False

# =========================
# Strategy flags
# =========================
USE_CLASS_BALANCED_SAMPLING = True