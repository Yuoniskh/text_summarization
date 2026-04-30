# config.py
import os

# مسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'my_training_data.csv')
CLEANED_DATA_PATH = os.path.join(DATA_DIR, 'cleaned_data.csv')

# إنشاء مجلد النماذج إذا لم يكن موجوداً
os.makedirs(MODELS_DIR, exist_ok=True)

# إعدادات التنظيف
MIN_ARTICLE_WORDS = 30
MIN_SUMMARY_WORDS = 3
MAX_ARTICLE_WORDS = 2000  # اختياري

# إعدادات التلخيص
DEFAULT_SUMMARY_SENTENCES = 3
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # نموذج sentence-transformers لـ TextRank

# إعدادات النموذج Hybrid Deep Learning
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, 'hybrid_model.keras')
HYBRID_EPOCHS = 20
HYBRID_BATCH_SIZE = 32
HYBRID_THRESHOLD = 0.5  # عتبة قطع لتصنيف أهمية الجملة
HYBRID_VALIDATION_SPLIT = 0.2
HYBRID_RANDOM_STATE = 42

# إعدادات التقييم
ROUGE_METRICS = ['rouge1', 'rouge2', 'rougeL']
EVAL_SAMPLE_SIZE = 500