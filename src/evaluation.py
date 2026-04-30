# src/evaluation.py
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from typing import Dict, List, Optional
import config

scorer = rouge_scorer.RougeScorer(config.ROUGE_METRICS, use_stemmer=True)

def calculate_rouge_scores(reference: str, prediction: str) -> Dict[str, float]:
    """حساب مقاييس ROUGE لنصين."""
    scores = scorer.score(reference, prediction)
    return {metric: scores[metric].fmeasure for metric in config.ROUGE_METRICS}

def evaluate_model(df: pd.DataFrame, pred_col: str, ref_col: str = 'highlights',
                   sample_size: Optional[int] = None, random_state: int = 42) -> Dict[str, float]:
    """
    تقييم أداء نموذج تلخيص على مجموعة بيانات.
    
    يدعم جميع النماذج: TF-IDF, TextRank, Hybrid Deep Learning
    
    Args:
        df: DataFrame يحتوي على الملخصات المتوقعة والمرجعية
        pred_col: اسم العمود المتضمن الملخصات المتوقعة
        ref_col: اسم العمود المتضمن الملخصات المرجعية (افتراضي: 'highlights')
        sample_size: عدد العينات للتقييم (None = جميع العينات)
        random_state: seed للعشوائية
        
    Returns:
        قاموس بمقاييس ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
    """
    eval_df = df.copy()
    if sample_size:
        eval_df = eval_df.sample(n=sample_size, random_state=random_state)

    rouge1, rouge2, rougeL = [], [], []
    for _, row in eval_df.iterrows():
        ref = str(row[ref_col])
        pred = str(row[pred_col])
        scores = calculate_rouge_scores(ref, pred)
        rouge1.append(scores['rouge1'])
        rouge2.append(scores['rouge2'])
        rougeL.append(scores['rougeL'])

    return {
        'ROUGE-1': np.mean(rouge1),
        'ROUGE-2': np.mean(rouge2),
        'ROUGE-L': np.mean(rougeL)
    }

def compare_models(df: pd.DataFrame, model_predictions: Dict[str, str],
                   sample_size: int = config.EVAL_SAMPLE_SIZE) -> pd.DataFrame:
    """
    مقارنة عدة نماذج وإرجاع DataFrame بالنتائج.
    
    يدعم مقارنة TF-IDF, TextRank, Hybrid Deep Learning وأي نماذج أخرى
    
    Args:
        df: DataFrame مع أعمدة الملخصات من جميع النماذج
        model_predictions: قاموس بصيغة {'Model Name': 'column_name', ...}
        sample_size: عدد العينات للمقارنة
        
    Returns:
        DataFrame بنتائج التقييم لكل نموذج
        
    Example:
        comparison = compare_models(df, {
            'TF-IDF': 'tfidf_summary',
            'TextRank': 'textrank_summary',
            'Hybrid DL': 'hybrid_summary'
        })
    """
    results = []
    sample_df = df.sample(n=sample_size, random_state=42)
    for model_name, pred_col in model_predictions.items():
        scores = evaluate_model(sample_df, pred_col, sample_size=None)  # لا نأخذ عينة ثانية
        scores['Model'] = model_name
        results.append(scores)
    return pd.DataFrame(results)