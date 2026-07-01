import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from typing import Dict, List, Optional
import config

scorer = rouge_scorer.RougeScorer(config.ROUGE_METRICS, use_stemmer=True)

def calculate_rouge_scores(reference: str, prediction: str) -> Dict[str, float]:

    if not reference or not prediction:
        return {metric: 0.0 for metric in config.ROUGE_METRICS}
    try:
        scores = scorer.score(reference, prediction)
        return {metric: scores[metric].fmeasure for metric in config.ROUGE_METRICS}
    except Exception:
        return {metric: 0.0 for metric in config.ROUGE_METRICS}

def evaluate_model(df: pd.DataFrame, pred_col: str, ref_col: str = 'highlights',
                   sample_size: Optional[int] = None, random_state: int = 42) -> Dict[str, float]:
    
    eval_df = df.copy()
    if sample_size and sample_size < len(eval_df):
        eval_df = eval_df.sample(n=sample_size, random_state=random_state)
    
    rouge1, rouge2, rougeL = [], [], []
    
    for _, row in eval_df.iterrows():
        ref = str(row[ref_col]) if pd.notna(row[ref_col]) else ""
        pred = str(row[pred_col]) if pd.notna(row[pred_col]) else ""
        
        if ref and pred:
            scores = calculate_rouge_scores(ref, pred)
            rouge1.append(scores['rouge1'])
            rouge2.append(scores['rouge2'])
            rougeL.append(scores['rougeL'])
    
    if not rouge1:
        return {
            'ROUGE-1': 0.0,
            'ROUGE-2': 0.0,
            'ROUGE-L': 0.0
        }
    
    return {
        'ROUGE-1': np.mean(rouge1),
        'ROUGE-2': np.mean(rouge2),
        'ROUGE-L': np.mean(rougeL)
    }

def compare_models(df: pd.DataFrame, model_predictions: Dict[str, str],
                   sample_size: int = config.EVAL_SAMPLE_SIZE) -> pd.DataFrame:
    
    results = []
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    for model_name, pred_col in model_predictions.items():
        if pred_col not in sample_df.columns:
            print(f"⚠ Column '{pred_col}' not found in DataFrame for model '{model_name}'")
            continue
        scores = evaluate_model(sample_df, pred_col, sample_size=None)
        scores['Model'] = model_name
        results.append(scores)
    
    if not results:
        return pd.DataFrame(columns=['Model', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
    
    return pd.DataFrame(results)