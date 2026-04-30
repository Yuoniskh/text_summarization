# src/preprocessing.py
import re
import pandas as pd
from bs4 import BeautifulSoup
from src.utils import normalize_whitespace, remove_news_prefix
import config

# تجميع الأنماط لتحسين الأداء
HTML_PARSER = 'lxml'
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+|ftp://\S+', re.IGNORECASE)
EDITOR_NOTE_PATTERN = re.compile(r"Editor's note:.*?(?=\.\s|$)", re.IGNORECASE)
COPYRIGHT_PATTERN = re.compile(r'Copyright\s+\d{4}\s+.*?All rights reserved\.?', re.IGNORECASE)
PROMO_PATTERN = re.compile(r'Read more\.?|Click here\.?|Watch now\.?', re.IGNORECASE)

def clean_text(text: str, is_summary: bool = False) -> str:
    """تنظيف شامل للنص (مقال أو ملخص)."""
    if pd.isna(text):
        return ""
    text = str(text)

    # إزالة HTML
    soup = BeautifulSoup(text, HTML_PARSER)
    text = soup.get_text(separator=" ")

    # إزالة الروابط
    text = URL_PATTERN.sub(' ', text)

    # تنظيف خاص بالمقالات
    if not is_summary:
        text = remove_news_prefix(text)
        text = EDITOR_NOTE_PATTERN.sub(' ', text)

    text = COPYRIGHT_PATTERN.sub(' ', text)
    text = PROMO_PATTERN.sub(' ', text)

    # إصلاح مسافات بعد علامات الترقيم
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

    # توحيد المسافات
    text = normalize_whitespace(text)
    return text

def load_and_clean_data(input_path: str = None) -> pd.DataFrame:
    """تحميل البيانات من CSV وتنظيفها."""
    input_path = input_path or config.RAW_DATA_PATH
    df = pd.read_csv(input_path)

    # التحقق من الأعمدة
    required_cols = ['article', 'highlights']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")

    # نسخ الأعمدة المطلوبة
    df_clean = df[required_cols].copy()

    # تنظيف النصوص
    df_clean['article'] = df_clean['article'].apply(lambda x: clean_text(x, is_summary=False))
    df_clean['highlights'] = df_clean['highlights'].apply(lambda x: clean_text(x, is_summary=True))

    # إزالة الصفوف الفارغة بعد التنظيف
    df_clean = df_clean[(df_clean['article'].str.strip() != '') & (df_clean['highlights'].str.strip() != '')]

    # فلترة حسب عدد الكلمات
    df_clean['article_word_count'] = df_clean['article'].apply(lambda x: len(str(x).split()))
    df_clean['summary_word_count'] = df_clean['highlights'].apply(lambda x: len(str(x).split()))

    df_clean = df_clean[
        (df_clean['article_word_count'] >= config.MIN_ARTICLE_WORDS) &
        (df_clean['summary_word_count'] >= config.MIN_SUMMARY_WORDS)
    ]
    if config.MAX_ARTICLE_WORDS:
        df_clean = df_clean[df_clean['article_word_count'] <= config.MAX_ARTICLE_WORDS]

    # إزالة التكرارات
    df_clean = df_clean.drop_duplicates(subset=['article']).reset_index(drop=True)

    # حذف أعمدة العد المؤقتة
    df_clean = df_clean.drop(columns=['article_word_count', 'summary_word_count'])

    return df_clean

if __name__ == "__main__":
    df = load_and_clean_data()
    print(f"Cleaned dataset shape: {df.shape}")
    df.to_csv(config.CLEANED_DATA_PATH, index=False, encoding='utf-8')
    print(f"Saved to {config.CLEANED_DATA_PATH}")