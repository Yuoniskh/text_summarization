import pandas as pd
import os
import numpy as np
from src.preprocessing import load_and_clean_data
from src.summarization import ExtractiveSummarizer, batch_summarize
from src.hybrid_deep_model import HybridDeepSummarizer, batch_summarize_hybrid
from src.evaluation import evaluate_model, compare_models
import config

def main():
    print("=" * 70)
    print("TEXT SUMMARIZATION - MODEL COMPARISON")
    print("=" * 70)
    
    print("\n[1/6] Loading and cleaning data...")
    df = load_and_clean_data()
    print(f"✓ Dataset shape: {df.shape}")
    
    print("\n[2/6] Initializing extractive summarizers...")
    tfidf_summarizer = ExtractiveSummarizer(method='tfidf')
    textrank_summarizer = ExtractiveSummarizer(method='textrank')
    print("✓ TF-IDF summarizer initialized")
    print("✓ TextRank summarizer initialized")
    
    print("\n[3/6] Loading Hybrid Deep Learning model...")
    hybrid_summarizer = None
    if os.path.exists(config.HYBRID_MODEL_PATH):
        try:
            hybrid_summarizer = HybridDeepSummarizer.load_model(config.HYBRID_MODEL_PATH)
            print("✓ Hybrid model loaded successfully!")
        except Exception as e:
            print(f"⚠ Could not load Hybrid model: {str(e)}")
            print("  Run: python train_hybrid_model.py")
    else:
        print(f"⚠ Hybrid model not found at {config.HYBRID_MODEL_PATH}")
        print("  Train the model first using: python train_hybrid_model.py")
    
    print("\n[4/6] Generating summaries...")
    sample_size = min(200, len(df))
    sample_df = df.sample(n=sample_size, random_state=42).copy()
    print(f"  Using sample size: {sample_size}")
    
    print("  Generating TF-IDF summaries...")
    sample_df['tfidf_summary'] = batch_summarize(sample_df, 'article', tfidf_summarizer)
    
    print("  Generating TextRank summaries...")
    sample_df['textrank_summary'] = batch_summarize(sample_df, 'article', textrank_summarizer)
    
    if hybrid_summarizer:
        print("  Generating Hybrid Deep Learning summaries...")
        sample_df['hybrid_summary'] = batch_summarize_hybrid(
            sample_df, 'article', hybrid_summarizer
        )
    
    print("\n[5/6] Evaluating models...")
    
    tfidf_scores = evaluate_model(
        sample_df, 
        'tfidf_summary', 
        sample_size=None  
    )
    
    textrank_scores = evaluate_model(
        sample_df, 
        'textrank_summary', 
        sample_size=None
    )
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    print("\n--- TF-IDF Scores ---")
    for k, v in tfidf_scores.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n--- TextRank Scores ---")
    for k, v in textrank_scores.items():
        print(f"  {k}: {v:.4f}")
    
    hybrid_scores = None
    if hybrid_summarizer and 'hybrid_summary' in sample_df.columns:
        hybrid_scores = evaluate_model(
            sample_df, 
            'hybrid_summary', 
            sample_size=None
        )
        print("\n--- Hybrid Deep Learning Scores ---")
        for k, v in hybrid_scores.items():
            print(f"  {k}: {v:.4f}")
    
    print("\n[6/6] Comparison Table...")
    models_to_compare = {
        'TF-IDF': 'tfidf_summary',
        'TextRank': 'textrank_summary'
    }
    if hybrid_summarizer and 'hybrid_summary' in sample_df.columns:
        models_to_compare['Hybrid DL'] = 'hybrid_summary'
    
    comparison = compare_models(
        sample_df, 
        models_to_compare, 
        sample_size=len(sample_df)
    )
    
    print("\n" + "=" * 50)
    print("COMPARISON TABLE")
    print("=" * 50)
    print(comparison.to_string(index=False))
    
    print("\n" + "=" * 50)
    print("CUSTOM TEXT SUMMARIZATION EXAMPLE")
    print("=" * 50)
    
    custom_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines,
    in contrast to the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of "intelligent agents":
    any device that perceives its environment and takes actions that maximize
    its chance of successfully achieving its goals. Colloquially, the term
    "artificial intelligence" is often used to describe machines that mimic
    "cognitive" functions that humans associate with the human mind, such as
    "learning" and "problem solving". As machines become increasingly capable,
    tasks considered to require intelligence are often removed from the
    definition of AI, a phenomenon known as the AI effect. A quip in Tesler's
    Theorem says "AI is whatever hasn't been done yet."
    """
    
    print("\nOriginal text:")
    print("-" * 40)
    print(custom_text.strip())
    
    print("\nTextRank summary (2 sentences):")
    print("-" * 40)
    print(textrank_summarizer.summarize(custom_text, num_sentences=2))
    
    print("\nTF-IDF summary (2 sentences):")
    print("-" * 40)
    print(tfidf_summarizer.summarize(custom_text, num_sentences=2))
    
    if hybrid_summarizer:
        try:
            print("\nHybrid DL summary (2 sentences):")
            print("-" * 40)
            print(hybrid_summarizer.summarize(custom_text, num_sentences=2))
        except Exception as e:
            print(f"\n⚠ Hybrid summarization failed: {str(e)}")
    
    print("\n" + "=" * 70)
    print("✓ COMPARISON COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()