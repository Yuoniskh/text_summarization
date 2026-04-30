# src/hybrid_deep_model.py
"""
Hybrid Extractive Summarization Model using Deep Learning

Pipeline:
1. Extract sentence-level features from text (TF-IDF, TextRank, position, length)
2. Create training dataset using weak supervision (ROUGE-based labeling)
3. Train neural network to predict sentence importance scores
4. Use trained model to generate extractive summaries

Architecture:
    Input features → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) 
    → Dropout(0.2) → Dense(1, sigmoid)
    Output: Probability [0, 1] representing sentence importance
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import os
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.optimizers import Adam

from src.utils import split_sentences, normalize_whitespace
import config


class HybridDeepSummarizer:
    """
    Hybrid Extractive Summarizer using TF-IDF, TextRank, and Deep Learning.
    
    Features used:
    - TF-IDF score: Term frequency-inverse document frequency
    - TextRank score: PageRank-based sentence importance
    - Sentence position: Normalized position in document
    - Sentence length: Word count
    - Normalized sentence length: Word count / avg length
    """
    
    def __init__(self, embedding_model: str = config.EMBEDDING_MODEL):
        """Initialize the hybrid summarizer with embeddings for TextRank."""
        self.embedding_model = embedding_model
        self.encoder = SentenceTransformer(embedding_model)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=1)
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_trained = False
        
    def extract_sentence_features(self, text: str) -> Tuple[List[str], np.ndarray]:
        """
        Extract features for each sentence in the text.
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple of (sentences, features_matrix)
            - sentences: List of sentences
            - features_matrix: Array of shape (num_sentences, num_features)
              Features: [tfidf_score, textrank_score, position_norm, length, norm_length]
        """
        text = normalize_whitespace(text)
        if not text:
            return [], np.array([])
        
        sentences = split_sentences(text)
        if len(sentences) < 2:
            # Handle short texts: return all sentences with features
            features = self._compute_features(sentences)
            return sentences, features
        
        features = self._compute_features(sentences)
        return sentences, features
    
    def _compute_features(self, sentences: List[str]) -> np.ndarray:
        """Compute all features for sentences."""
        num_sentences = len(sentences)
        features = []
        
        # TF-IDF scores
        tfidf_scores = self._compute_tfidf_scores(sentences)
        
        # TextRank scores
        textrank_scores = self._compute_textrank_scores(sentences)
        
        # Position normalization (0 to 1)
        positions = np.arange(num_sentences) / max(num_sentences - 1, 1)
        
        # Sentence length features
        lengths = np.array([len(s.split()) for s in sentences], dtype=float)
        avg_length = np.mean(lengths) if lengths.size > 0 else 1
        normalized_lengths = lengths / max(avg_length, 1)
        
        # Combine all features
        for i in range(num_sentences):
            feat = [
                tfidf_scores[i],           # Feature 0: TF-IDF score
                textrank_scores[i],        # Feature 1: TextRank score
                positions[i],              # Feature 2: Position (normalized)
                lengths[i],                # Feature 3: Sentence length
                normalized_lengths[i]      # Feature 4: Normalized length
            ]
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_tfidf_scores(self, sentences: List[str]) -> np.ndarray:
        """Compute TF-IDF scores for sentences."""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
            scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
            # Normalize by sentence length to avoid bias toward long sentences
            lengths = np.array([len(s.split()) for s in sentences], dtype=float)
            scores = scores / (lengths + 1)  # Prevent division by zero
            # Scale to [0, 1]
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores
        except:
            return np.ones(len(sentences)) / len(sentences)
    
    def _compute_textrank_scores(self, sentences: List[str]) -> np.ndarray:
        """Compute TextRank scores using sentence embeddings."""
        try:
            embeddings = self.encoder.encode(sentences)
            sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(sim_matrix, 0)  # Avoid self-loops
            
            # Create and process graph
            nx_graph = nx.from_numpy_array(sim_matrix)
            scores_dict = nx.pagerank(nx_graph)
            scores = np.array([scores_dict[i] for i in range(len(sentences))])
            
            # Normalize to [0, 1]
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores
        except:
            return np.ones(len(sentences)) / len(sentences)
    
    def create_training_data(self, df: pd.DataFrame, 
                            text_col: str = 'article',
                            summary_col: str = 'highlights',
                            sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training dataset with weak supervision.
        
        Uses ROUGE-based labeling: if similarity between sentence and reference 
        summary is high, label = 1, else label = 0.
        
        Args:
            df: DataFrame with 'article' and 'highlights' columns
            text_col: Column name for articles
            summary_col: Column name for highlights/summaries
            sample_size: Number of samples to use (None = all)
            
        Returns:
            Tuple of (features_array, labels_array)
        """
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=config.HYBRID_RANDOM_STATE)
        
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        
        all_features = []
        all_labels = []
        
        print(f"Creating training data from {len(df)} samples...")
        for idx, row in df.iterrows():
            article = str(row[text_col]).strip()
            summary = str(row[summary_col]).strip()
            
            if not article or not summary:
                continue
            
            try:
                sentences, features = self.extract_sentence_features(article)
                
                if len(sentences) == 0 or features.size == 0:
                    continue
                
                # Generate labels using ROUGE-1 similarity
                labels = []
                for sentence in sentences:
                    # Calculate ROUGE-1 F1 score between sentence and reference summary
                    scores = scorer.score(summary, sentence)
                    rouge_score = scores['rouge1'].fmeasure
                    
                    # Label as 1 if similarity is above threshold (0.3), else 0
                    label = 1 if rouge_score > 0.3 else 0
                    labels.append(label)
                
                all_features.append(features)
                all_labels.extend(labels)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1} samples...")
            
            except Exception as e:
                print(f"  Warning: Error processing sample {idx}: {str(e)}")
                continue
        
        print(f"Created {len(all_labels)} sentence-level training examples")
        
        # Combine features from all articles
        features_array = np.vstack(all_features) if all_features else np.array([])
        labels_array = np.array(all_labels, dtype=np.float32)
        
        # Scale features to [0, 1]
        if features_array.size > 0:
            features_array = self.feature_scaler.fit_transform(features_array)
        
        return features_array, labels_array
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              validation_split: Optional[float] = None,
              verbose: int = 1) -> Dict:
        """
        Train the deep learning model.
        
        Args:
            X_train: Training features of shape (num_samples, num_features)
            y_train: Training labels (binary)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation split ratio
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("Training data is empty")
        
        epochs = epochs or config.HYBRID_EPOCHS
        batch_size = batch_size or config.HYBRID_BATCH_SIZE
        validation_split = validation_split or config.HYBRID_VALIDATION_SPLIT
        
        # Build model
        self.model = self._build_model(input_dim=X_train.shape[1])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        print(f"Training hybrid model for {epochs} epochs...")
        print(f"Training samples: {X_train.shape[0]}, Features per sample: {X_train.shape[1]}")
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            ]
        )
        
        self.is_trained = True
        print("✓ Model training completed!")
        
        return history.history
    
    def _build_model(self, input_dim: int) -> Model:
        """Build the neural network architecture."""
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def summarize(self, text: str, num_sentences: int = None) -> str:
        """
        Generate summary from text using trained model.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            
        Returns:
            Extractive summary
            
        Raises:
            RuntimeError: If model not trained yet
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Please train the model first using train() method.")
        
        num_sentences = num_sentences or config.DEFAULT_SUMMARY_SENTENCES
        text = normalize_whitespace(text)
        
        if not text:
            return ""
        
        sentences, features = self.extract_sentence_features(text)
        
        if len(sentences) == 0:
            return ""
        
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
        
        if features.size == 0:
            return " ".join(sentences[:num_sentences])
        
        # Scale features
        features = self.feature_scaler.transform(features)
        
        # Predict importance scores
        importance_scores = self.model.predict(features, verbose=0).flatten()
        
        # Select top-N sentences while preserving original order
        top_indices = np.argsort(importance_scores)[-num_sentences:]
        top_indices.sort()  # Maintain original order
        
        summary = " ".join([sentences[i] for i in top_indices])
        return summary
    
    def save_model(self, path: str = None) -> None:
        """
        Save trained model and scaler to disk.
        
        Args:
            path: Path to save model (default: config.HYBRID_MODEL_PATH)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save untrained model")
        
        path = path or config.HYBRID_MODEL_PATH
        
        # Save Keras model
        self.model.save(path)
        
        # Save scaler metadata with all necessary attributes
        scaler_path = path.replace('.keras', '_scaler.json')
        scaler_data = {
            'data_min': self.feature_scaler.data_min_.tolist(),
            'data_max': self.feature_scaler.data_max_.tolist(),
            'data_range': self.feature_scaler.data_range_.tolist(),
            'scale': self.feature_scaler.scale_.tolist(),
            'min': self.feature_scaler.min_.tolist(),
            'n_features_in': int(self.feature_scaler.n_features_in_)
        }
        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f)
        
        print(f"✓ Model saved to {path}")
        print(f"✓ Scaler saved to {scaler_path}")
    
    @staticmethod
    def load_model(path: str = None) -> 'HybridDeepSummarizer':
        """
        Load trained model from disk.
        
        Args:
            path: Path to saved model (default: config.HYBRID_MODEL_PATH)
            
        Returns:
            HybridDeepSummarizer instance with loaded model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = path or config.HYBRID_MODEL_PATH
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        # Load Keras model
        keras_model = keras.models.load_model(path)
        
        # Load scaler
        scaler_path = path.replace('.keras', '_scaler.json')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
            scaler = MinMaxScaler()
            # Restore all necessary scaler attributes
            scaler.data_min_ = np.array(scaler_data['data_min'])
            scaler.data_max_ = np.array(scaler_data['data_max'])
            scaler.data_range_ = np.array(scaler_data['data_range'])
            scaler.n_features_in_ = len(scaler.data_min_)
            
            # Try to restore scale_ and min_ from saved data, compute if needed
            if 'scale' in scaler_data:
                scaler.scale_ = np.array(scaler_data['scale'])
            else:
                scaler.scale_ = np.divide(1.0, scaler.data_range_, 
                                         where=scaler.data_range_!=0, 
                                         out=np.zeros_like(scaler.data_range_))
            
            if 'min' in scaler_data:
                scaler.min_ = np.array(scaler_data['min'])
            else:
                scaler.min_ = -scaler.data_min_ * scaler.scale_
        else:
            scaler = MinMaxScaler()
        
        # Create summarizer instance
        summarizer = HybridDeepSummarizer()
        summarizer.model = keras_model
        summarizer.feature_scaler = scaler
        summarizer.is_trained = True
        
        print(f"✓ Model loaded from {path}")
        return summarizer


def batch_summarize_hybrid(df: pd.DataFrame, text_column: str, 
                           summarizer: HybridDeepSummarizer,
                           num_sentences: int = None) -> pd.Series:
    """
    Apply hybrid summarization to entire DataFrame column.
    
    Args:
        df: Input DataFrame
        text_column: Column name containing text to summarize
        summarizer: HybridDeepSummarizer instance (must be trained)
        num_sentences: Number of sentences in summaries
        
    Returns:
        Pandas Series with summaries
    """
    def safe_summarize(text):
        try:
            return summarizer.summarize(text, num_sentences)
        except Exception as e:
            print(f"Warning: {str(e)}")
            return ""
    
    return df[text_column].apply(safe_summarize)
