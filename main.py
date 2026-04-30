# main.py
import pandas as pd
import os
from src.preprocessing import load_and_clean_data
from src.summarization import ExtractiveSummarizer, batch_summarize
from src.hybrid_deep_model import HybridDeepSummarizer, batch_summarize_hybrid
from src.evaluation import evaluate_model, compare_models
import config

def main():
    # 1. تحميل وتنظيف البيانات
    print("Loading and cleaning data...")
    df = load_and_clean_data()
    print(f"Dataset shape: {df.shape}")

    # 2. إنشاء نماذج التلخيص
    tfidf_summarizer = ExtractiveSummarizer(method='tfidf')
    textrank_summarizer = ExtractiveSummarizer(method='textrank')
    
    # 3. محاولة تحميل نموذج Hybrid (إن كان موجوداً)
    hybrid_summarizer = None
    if os.path.exists(config.HYBRID_MODEL_PATH):
        try:
            print("Loading Hybrid Deep Learning model...")
            hybrid_summarizer = HybridDeepSummarizer.load_model(config.HYBRID_MODEL_PATH)
            print("✓ Hybrid model loaded successfully!")
        except Exception as e:
            print(f"⚠ Could not load Hybrid model: {str(e)}")
            print("  Run: python train_hybrid_model.py")
    else:
        print(f"⚠ Hybrid model not found at {config.HYBRID_MODEL_PATH}")
        print("  Train the model first using: python train_hybrid_model.py")

    # 4. توليد الملخصات (يمكن أخذ عينة للتجربة السريعة)
    sample_df = df.sample(n=min(200, len(df)), random_state=42).copy()
    print("Generating TF-IDF summaries...")
    sample_df['tfidf_summary'] = batch_summarize(sample_df, 'article', tfidf_summarizer)
    print("Generating TextRank summaries...")
    sample_df['textrank_summary'] = batch_summarize(sample_df, 'article', textrank_summarizer)
    
    if hybrid_summarizer:
        print("Generating Hybrid Deep Learning summaries...")
        sample_df['hybrid_summary'] = batch_summarize_hybrid(
            sample_df, 'article', hybrid_summarizer
        )

    # 5. تقييم النماذج
    print("\nEvaluating models...")
    tfidf_scores = evaluate_model(sample_df, 'tfidf_summary', sample_size=None)
    textrank_scores = evaluate_model(sample_df, 'textrank_summary', sample_size=None)

    print("\n--- TF-IDF Scores ---")
    for k, v in tfidf_scores.items():
        print(f"{k}: {v:.4f}")
    print("\n--- TextRank Scores ---")
    for k, v in textrank_scores.items():
        print(f"{k}: {v:.4f}")
    
    if hybrid_summarizer:
        hybrid_scores = evaluate_model(sample_df, 'hybrid_summary', sample_size=None)
        print("\n--- Hybrid Deep Learning Scores ---")
        for k, v in hybrid_scores.items():
            print(f"{k}: {v:.4f}")

    # 6. مقارنة في جدول
    models_to_compare = {
        'TF-IDF': 'tfidf_summary',
        'TextRank': 'textrank_summary'
    }
    if hybrid_summarizer and 'hybrid_summary' in sample_df.columns:
        models_to_compare['Hybrid DL'] = 'hybrid_summary'
    
    comparison = compare_models(sample_df, models_to_compare, 
                           sample_size=len(sample_df))  # استخدام كل البيانات المتاحة
    print("\nComparison Table:")
    print(comparison.to_string(index=False))

    # 7. مثال تلخيص نص مخصص
    custom_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines,
    in contrast to the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of "intelligent agents":
    any device that perceives its environment and takes actions that maximize
    its chance of successfully achieving its goals. Colloquially, the term
    "artificial intelligence" is often used to describe machines that mimic
    "cognitive" functions that humans associate with the human mind, such as
    "learning" and "problem solving".
    """
    print("\n--- Custom Text Summarization ---")
    print("Original text:\n", custom_text)
    print("\nTextRank summary:\n", textrank_summarizer.summarize(custom_text, num_sentences=2))
    print("\nTF-IDF summary:\n", tfidf_summarizer.summarize(custom_text, num_sentences=2))
    
    if hybrid_summarizer:
        try:
            print("\nHybrid DL summary:\n", hybrid_summarizer.summarize(custom_text, num_sentences=2))
        except Exception as e:
            print(f"\n⚠ Hybrid summarization failed: {str(e)}")

if __name__ == "__main__":
    main()