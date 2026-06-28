# config.py
import os

# ==========================================
# Base Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

RAW_DATA_PATH = os.path.join(DATA_DIR, "my_training_data.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_training_data.csv")

# Create directories automatically
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# Data Cleaning
# ==========================================
MIN_ARTICLE_WORDS = 30
MIN_SUMMARY_WORDS = 3
MAX_ARTICLE_WORDS = 2000

# ==========================================
# Summarization
# ==========================================
DEFAULT_SUMMARY_SENTENCES = 3

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ==========================================
# Hybrid Deep Learning (PyTorch)
# ==========================================
HYBRID_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "hybrid_model.pt"  # تغيير من .keras إلى .pt
)

HYBRID_EPOCHS = 25
HYBRID_BATCH_SIZE = 64
HYBRID_THRESHOLD = 0.25

HYBRID_RANDOM_STATE = 42

HYBRID_TRAIN_RATIO = 0.70
HYBRID_VALID_RATIO = 0.15
HYBRID_TEST_RATIO = 0.15

HYBRID_TRAINING_SAMPLE_SIZE = 5000
HYBRID_CHUNK_SIZE = 5000

LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 8

# ==========================================
# Evaluation
# ==========================================
ROUGE_METRICS = [
    "rouge1",
    "rouge2",
    "rougeL"
]

EVAL_SAMPLE_SIZE = 500

# ==========================================
# Saved Files
# ==========================================
NUM_FEATURES = 13  # عدد الميزات المستخدمة
USE_SMOTE = True   # استخدام SMOTE لموازنة البيانات
TRAINING_HISTORY_CSV = os.path.join(
    RESULTS_DIR,
    "training_history.csv"
)

METRICS_JSON = os.path.join(
    RESULTS_DIR,
    "metrics.json"
)

CLASSIFICATION_REPORT = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

CONFUSION_MATRIX_IMAGE = os.path.join(
    PLOTS_DIR,
    "confusion_matrix.png"
)

LOSS_CURVE = os.path.join(
    PLOTS_DIR,
    "loss_curve.png"
)

ACCURACY_CURVE = os.path.join(
    PLOTS_DIR,
    "accuracy_curve.png"
)

PRECISION_CURVE = os.path.join(
    PLOTS_DIR,
    "precision_curve.png"
)

RECALL_CURVE = os.path.join(
    PLOTS_DIR,
    "recall_curve.png"
)

F1_CURVE = os.path.join(
    PLOTS_DIR,
    "f1_curve.png"
)