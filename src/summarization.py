# src/summarization.py
import numpy as np
import networkx as nx
import pandas as pd
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.utils import split_sentences
import config

class ExtractiveSummarizer:
    """
    ملخص استخراجي باستخدام TF-IDF أو TextRank (مع sentence embeddings).
    """
    def __init__(self, method: str = 'textrank'):
        """
        method: 'tfidf' أو 'textrank'
        """
        self.method = method.lower()
        if self.method == 'textrank':
            self.encoder = SentenceTransformer(config.EMBEDDING_MODEL)
        elif self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
        else:
            raise ValueError("Method must be 'tfidf' or 'textrank'.")

    def summarize(self, text: str, num_sentences: int = None) -> str:
        """توليد ملخص من النص المعطى."""
        num_sentences = num_sentences or config.DEFAULT_SUMMARY_SENTENCES
        sentences = split_sentences(text)

        if not sentences:
            return ""
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # حساب درجات الجمل
        if self.method == 'tfidf':
            scores = self._score_tfidf(sentences)
        else:  # textrank
            scores = self._score_textrank(sentences)

        # اختيار أفضل الجمل مع الحفاظ على الترتيب الأصلي
        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices.sort()
        summary = " ".join([sentences[i] for i in top_indices])
        return summary

    def _score_tfidf(self, sentences: List[str]) -> np.ndarray:
        """حساب درجات TF-IDF للجمل."""
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
        # تطبيع اختياري (تقسيم على طول الجملة لتجنب تفضيل الجمل الطويلة جداً)
        lengths = np.array([len(s.split()) for s in sentences])
        scores = scores / (lengths + 1)  # منع القسمة على صفر
        return scores

    def _score_textrank(self, sentences: List[str]) -> np.ndarray:
        """حساب درجات TextRank باستخدام sentence embeddings."""
        embeddings = self.encoder.encode(sentences)
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)  # لا تربط الجملة بنفسها

        nx_graph = nx.from_numpy_array(sim_matrix)
        scores_dict = nx.pagerank(nx_graph)
        scores = np.array([scores_dict[i] for i in range(len(sentences))])
        return scores

def batch_summarize(df: pd.DataFrame, text_column: str, summarizer: ExtractiveSummarizer,
                    num_sentences: int = None) -> pd.Series:
    """تطبيق التلخيص على عمود كامل في DataFrame."""
    return df[text_column].apply(lambda x: summarizer.summarize(x, num_sentences))