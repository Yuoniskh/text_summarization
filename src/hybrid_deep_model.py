import os
import json
import gc
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from src.utils import split_sentences, normalize_whitespace
import config


# Focal Loss
class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        probs = torch.sigmoid(inputs)
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        focal_weight = (1 - p_t) ** self.gamma
        
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = -alpha_weight * focal_weight * torch.log(p_t + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Model and Dataset

class HybridDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HybridNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class HybridDeepSummarizer:

    def __init__(self, embedding_model: str = config.EMBEDDING_MODEL):
        self.embedding_model = embedding_model
        self.encoder = SentenceTransformer(embedding_model)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=1)
        self.feature_scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        self.input_dim = None
        self.best_threshold = 0.25
        
        try:
            import nltk
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {'the', 'a', 'an', 'of', 'to', 'for', 'with', 'on', 'at', 'from',
                              'by', 'in', 'as', 'is', 'was', 'were', 'are', 'be', 'been',
                              'that', 'this', 'these', 'those', 'it', 'he', 'she', 'they'}
        
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=['parser'])
        except:
            print("⚠️ spacy not available. NER and POS features will use fallback.")

    # Feature extraction

    def extract_sentence_features(self, text: str) -> Tuple[List[str], np.ndarray]:
        text = normalize_whitespace(text)
        if not text:
            return [], np.array([])
        sentences = split_sentences(text)
        if len(sentences) == 0:
            return [], np.array([])
        features = self._compute_features(text, sentences)
        return sentences, features

    def _compute_features(self, text: str, sentences: List[str]) -> np.ndarray:
        num_sentences = len(sentences)
        
        # 1. TF-IDF scores
        tfidf_scores = self._compute_tfidf_scores(text, sentences)
        
        # 2. TextRank scores
        textrank_scores = self._compute_textrank_scores(sentences)
        
        # 3. Position features
        positions = np.arange(num_sentences) / max(num_sentences - 1, 1)
        
        # 4. Length features
        lengths = np.array([len(s.split()) for s in sentences], dtype=float)
        avg_length = np.mean(lengths) if lengths.size > 0 else 1.0
        normalized_lengths = lengths / max(avg_length, 1.0)
        
        # 5. BM25 scores
        bm25_scores = self._compute_bm25_scores(sentences)
        
        # 6. Sentence centrality
        centrality_scores = self._compute_centrality_scores(sentences)
        
        # 7. Sentence entropy
        entropy_scores = self._compute_entropy_scores(sentences)
        
        # 8. NER scores
        ner_scores = self._compute_ner_scores(sentences)
        
        # 9. POS features
        pos_features = self._compute_pos_features(sentences)
        
        # 10. Position binary
        position_binary = self._compute_position_binary(num_sentences)
        
        # 11. Stopword ratio
        stopword_ratio = self._compute_stopword_ratio(sentences)
        
        # 12. Unique word ratio
        unique_ratio = self._compute_unique_ratio(sentences)
        
        # 13. Embedding features
        embedding_features = self._compute_embedding_features(sentences)

        feats = np.vstack([
            tfidf_scores,           
            textrank_scores,        
            positions,              
            normalized_lengths,    
            bm25_scores,            
            centrality_scores,     
            entropy_scores,        
            ner_scores,            
            pos_features,          
            position_binary,       
            stopword_ratio,        
            unique_ratio,          
            embedding_features     
        ]).T.astype(np.float32)
        
        return feats

    def _compute_tfidf_scores(self, text: str, sentences: List[str]) -> np.ndarray:

        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            sentence_vectors = self.tfidf_vectorizer.transform(sentences)
            scores = np.asarray(sentence_vectors.sum(axis=1)).flatten()
            
            lengths = np.array([len(s.split()) for s in sentences], dtype=float)
            scores = scores / (lengths + 1.0)
            
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores
        except Exception:
            return np.ones(len(sentences), dtype=float) / max(len(sentences), 1)

    def _compute_textrank_scores(self, sentences: List[str]) -> np.ndarray:

        try:
            embeddings = self.encoder.encode(sentences, batch_size=32, show_progress_bar=False)
            sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(sim_matrix, 0)
            nx_graph = nx.from_numpy_array(sim_matrix)
            scores_dict = nx.pagerank(nx_graph, max_iter=100, tol=1e-6)
            scores = np.array([scores_dict[i] for i in range(len(sentences))])
            
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores
        except Exception:
            return np.ones(len(sentences), dtype=float) / max(len(sentences), 1)

    def _compute_bm25_scores(self, sentences: List[str]) -> np.ndarray:

        try:
            from rank_bm25 import BM25Okapi
            import string
            
            tokenized_sentences = [
                sentence.lower().translate(str.maketrans('', '', string.punctuation)).split()
                for sentence in sentences
            ]
            
            bm25 = BM25Okapi(tokenized_sentences)
            scores = []
            
            for i, tokens in enumerate(tokenized_sentences):
                score = bm25.get_scores(tokens)[i]
                scores.append(score)
            
            scores = np.array(scores, dtype=float)
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores
        except Exception:
            return np.ones(len(sentences), dtype=float) / max(len(sentences), 1)

    def _compute_centrality_scores(self, sentences: List[str]) -> np.ndarray:

        try:
            embeddings = self.encoder.encode(sentences, batch_size=32, show_progress_bar=False)
            sim_matrix = cosine_similarity(embeddings)
            centrality = np.mean(sim_matrix, axis=1)
            
            if centrality.max() > 0:
                centrality = centrality / centrality.max()
            return centrality
        except Exception:
            return np.ones(len(sentences), dtype=float) / max(len(sentences), 1)

    def _compute_entropy_scores(self, sentences: List[str]) -> np.ndarray:

        from collections import Counter
        import math
        
        scores = []
        for sentence in sentences:
            words = sentence.lower().split()
            if not words:
                scores.append(0.0)
                continue
            
            word_counts = Counter(words)
            total_words = len(words)
            
            entropy = 0.0
            for count in word_counts.values():
                prob = count / total_words
                entropy -= prob * math.log2(prob)
            
            max_entropy = math.log2(total_words) if total_words > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            scores.append(normalized_entropy)
        
        scores = np.array(scores, dtype=float)
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def _compute_ner_scores(self, sentences: List[str]) -> np.ndarray:

        scores = []
        if self.nlp is None:
            return np.ones(len(sentences), dtype=float) / max(len(sentences), 1)
        
        try:
            for sentence in sentences:
                doc = self.nlp(sentence)
                num_entities = len(doc.ents)
                num_words = max(len(sentence.split()), 1)
                score = num_entities / num_words
                scores.append(score)
            
            scores = np.array(scores, dtype=float)
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores
        except Exception:
            return np.ones(len(sentences), dtype=float) / max(len(sentences), 1)

    def _compute_pos_features(self, sentences: List[str]) -> np.ndarray:

        scores = []
        if self.nlp is None:
            return np.ones(len(sentences), dtype=float) / max(len(sentences), 1)
        
        try:
            for sentence in sentences:
                doc = self.nlp(sentence)
                nouns = sum(1 for token in doc if token.pos_ in ['NOUN', 'PROPN'])
                verbs = sum(1 for token in doc if token.pos_ == 'VERB')
                adj = sum(1 for token in doc if token.pos_ == 'ADJ')
                
                total = max(len(doc), 1)
                score = (nouns + verbs + adj) / total
                scores.append(score)
            
            scores = np.array(scores, dtype=float)
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores
        except Exception:
            return np.ones(len(sentences), dtype=float) / max(len(sentences), 1)

    def _compute_position_binary(self, num_sentences: int) -> np.ndarray:

        if num_sentences <= 1:
            return np.array([0.5] * num_sentences, dtype=float)
        
        positions = np.zeros(num_sentences, dtype=float)
        positions[0] = 1.0  
        positions[-1] = 1.0  
        
        first_third = max(num_sentences // 3, 1)
        last_third = max(2 * num_sentences // 3, 1)
        
        if num_sentences > 3:
            positions[1:first_third] = 0.7
            positions[last_third:-1] = 0.7
            positions[first_third:last_third] = 0.3
        
        return positions

    def _compute_stopword_ratio(self, sentences: List[str]) -> np.ndarray:

        scores = []
        for sentence in sentences:
            words = sentence.lower().split()
            if not words:
                scores.append(0.0)
                continue
            num_stopwords = sum(1 for w in words if w in self.stop_words)
            ratio = num_stopwords / len(words)
            scores.append(ratio)
        
        scores = np.array(scores, dtype=float)
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def _compute_unique_ratio(self, sentences: List[str]) -> np.ndarray:

        scores = []
        for sentence in sentences:
            words = sentence.lower().split()
            if not words:
                scores.append(0.0)
                continue
            unique_words = len(set(words))
            ratio = unique_words / len(words)
            scores.append(ratio)
        
        scores = np.array(scores, dtype=float)
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def _compute_embedding_features(self, sentences: List[str]) -> np.ndarray:

        try:
            embeddings = self.encoder.encode(sentences, batch_size=32, show_progress_bar=False)
            
            features = []
            for emb in embeddings:
                mean_emb = np.mean(emb)
                std_emb = np.std(emb)
                max_emb = np.max(emb)
                min_emb = np.min(emb)
                range_emb = max_emb - min_emb
                features.append([mean_emb, std_emb, max_emb, min_emb, range_emb])
            
            features = np.array(features, dtype=float)
            
            for i in range(features.shape[1]):
                col = features[:, i]
                if col.max() > col.min():
                    features[:, i] = (col - col.min()) / (col.max() - col.min())
            
            scores = np.mean(features, axis=1)
            
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores
            
        except Exception:
            return np.ones(len(sentences), dtype=float) / max(len(sentences), 1)

    # Training data creation with composite score
    
    def create_training_data(self, df: pd.DataFrame, text_col: str = 'article', 
                            summary_col: str = 'highlights',
                            sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=config.HYBRID_RANDOM_STATE)
        
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        threshold = 0.30  # عتبة أعلى قليلاً للـ Score المركب

        all_features = []
        all_labels = []
        
        print(f"Creating training data from {len(df)} samples...")
        print("Using composite score: 0.45*ROUGE + 0.25*TextRank + 0.20*TF-IDF + 0.10*Position")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating training data"):
            article = str(row.get(text_col, '')).strip()
            summary = str(row.get(summary_col, '')).strip()
            
            if not article or not summary:
                continue
            
            try:
                sentences, features = self.extract_sentence_features(article)
                if len(sentences) == 0 or features.size == 0:
                    continue
                
                # 0: TF-IDF, 1: TextRank, 2: Position, 3: Length, ...
                tfidf_scores = features[:, 0]
                textrank_scores = features[:, 1]
                position_scores = features[:, 2]
                
                tfidf_norm = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
                textrank_norm = (textrank_scores - textrank_scores.min()) / (textrank_scores.max() - textrank_scores.min() + 1e-8)
                position_norm = (position_scores - position_scores.min()) / (position_scores.max() - position_scores.min() + 1e-8)
                
                labels = []
                for i, sentence in enumerate(sentences):
                    scores = scorer.score(summary, sentence)
                    rouge_score = scores['rouge1'].fmeasure
                    
                    composite_score = (
                        0.45 * rouge_score +
                        0.25 * textrank_norm[i] +
                        0.20 * tfidf_norm[i] +
                        0.10 * position_norm[i]
                    )
                    
                    label = 1 if composite_score > threshold else 0
                    labels.append(label)
                
                all_features.append(features)
                all_labels.extend(labels)
                
            except Exception as e:
                continue
            finally:
                gc.collect()

        if not all_features:
            return np.array([]), np.array([])

        features_array = np.vstack(all_features)
        labels_array = np.array(all_labels, dtype=np.float32)
        
        return features_array, labels_array

    # Training

    def find_best_threshold(self, val_loader: DataLoader) -> float:

        self.model.eval()
        val_preds_probs = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                probs = torch.sigmoid(outputs)
                val_preds_probs.extend(probs.numpy().flatten())
                val_targets.extend(batch_y.numpy().astype(int).flatten())
        
        thresholds = np.arange(0.05, 0.95, 0.05)
        best_f1 = 0
        best_threshold = 0.25
        
        for threshold in thresholds:
            preds = (np.array(val_preds_probs) >= threshold).astype(int)
            f1 = f1_score(val_targets, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"✓ Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
        return best_threshold

    def train(self, X: np.ndarray, y: np.ndarray, epochs: Optional[int] = None, 
              batch_size: Optional[int] = None, validation_split: Optional[float] = None, 
              verbose: int = 1) -> Dict:

        if X.size == 0 or y.size == 0:
            raise ValueError("Training data is empty")
        
        epochs = epochs or config.HYBRID_EPOCHS
        batch_size = batch_size or config.HYBRID_BATCH_SIZE
        
        self.input_dim = X.shape[1]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=config.HYBRID_RANDOM_STATE, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=config.HYBRID_RANDOM_STATE, stratify=y_temp
        )

        print("\n" + "=" * 60)
        print("Dataset Split")
        print("=" * 60)
        print(f"Training   : {len(X_train):,}")
        print(f"Validation : {len(X_val):,}")
        print(f"Testing    : {len(X_test):,}")
        print(f"Features   : {self.input_dim}")
        print("=" * 60)

        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)

        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=config.HYBRID_RANDOM_STATE, k_neighbors=3)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            print(f"✓ After SMOTE: Pos={np.sum(y_train_resampled)}, Neg={len(y_train_resampled)-np.sum(y_train_resampled)}")
        except:
            print("⚠️ SMOTE not available, using original data")
            X_train_resampled, y_train_resampled = X_train_scaled, y_train

        num_neg = np.sum(y_train_resampled == 0)
        num_pos = np.sum(y_train_resampled == 1)
        pos_ratio = num_pos / (num_pos + num_neg)
        alpha = 1 - pos_ratio  
        print(f"✓ Alpha for Focal Loss: {alpha:.4f}")

        train_ds = HybridDataset(X_train_resampled, y_train_resampled)
        val_ds = HybridDataset(X_val_scaled, y_val)
        test_ds = HybridDataset(X_test_scaled, y_test)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        self.model = HybridNet(input_dim=self.input_dim)
        
        criterion = FocalLoss(alpha=alpha, gamma=2.0, reduction='mean')
        optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        history = {
            "loss": [], "val_loss": [],
            "accuracy": [], "val_accuracy": [],
            "precision": [], "val_precision": [],
            "recall": [], "val_recall": [],
            "f1_score": [], "val_f1_score": []
        }
        
        best_val_loss = float('inf')
        patience = 8  
        patience_counter = 0

        print("\n" + "=" * 60)
        print("Training Started")
        print("=" * 60)

        for epoch in range(epochs):

            self.model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                train_preds.extend((outputs.detach().numpy() > 0.0).astype(int).flatten())
                train_targets.extend(batch_y.numpy().astype(int).flatten())

            train_loss /= len(train_loader.dataset)
            train_acc = accuracy_score(train_targets, train_preds)
            train_prec = precision_score(train_targets, train_preds, zero_division=0)
            train_rec = recall_score(train_targets, train_preds, zero_division=0)
            train_f1 = f1_score(train_targets, train_preds, zero_division=0)

            self.model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    val_preds.extend((outputs.numpy() > 0.0).astype(int).flatten())
                    val_targets.extend(batch_y.numpy().astype(int).flatten())

            val_loss /= len(val_loader.dataset)
            val_acc = accuracy_score(val_targets, val_preds)
            val_prec = precision_score(val_targets, val_preds, zero_division=0)
            val_rec = recall_score(val_targets, val_preds, zero_division=0)
            val_f1 = f1_score(val_targets, val_preds, zero_division=0)

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)
            history["precision"].append(train_prec)
            history["val_precision"].append(val_prec)
            history["recall"].append(train_rec)
            history["val_recall"].append(val_rec)
            history["f1_score"].append(train_f1)
            history["val_f1_score"].append(val_f1)

            scheduler.step(val_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - f1: {train_f1:.4f} | "
                      f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_f1: {val_f1:.4f}")

            # --- Early Stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), config.HYBRID_MODEL_PATH + ".tmp")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if os.path.exists(config.HYBRID_MODEL_PATH + ".tmp"):
            self.model.load_state_dict(torch.load(config.HYBRID_MODEL_PATH + ".tmp"))
            os.remove(config.HYBRID_MODEL_PATH + ".tmp")

        self.is_trained = True

        self.best_threshold = self.find_best_threshold(val_loader)

        print("\n" + "=" * 60)
        print("Final Evaluation on Test Set")
        print("=" * 60)
        
        self.model.eval()
        test_loss = 0.0
        test_preds, test_targets = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= self.best_threshold).int()
                
                test_preds.extend(preds.numpy().astype(int).flatten())
                test_targets.extend(batch_y.numpy().astype(int).flatten())

        test_loss /= len(test_loader.dataset)
        test_acc = accuracy_score(test_targets, test_preds)
        test_prec = precision_score(test_targets, test_preds, zero_division=0)
        test_rec = recall_score(test_targets, test_preds, zero_division=0)
        test_f1 = f1_score(test_targets, test_preds, zero_division=0)

        history["test_loss"] = test_loss
        history["test_accuracy"] = test_acc
        history["test_precision"] = test_prec
        history["test_recall"] = test_rec
        history["test_f1"] = test_f1
        history["best_threshold"] = self.best_threshold

        print(f"Test Loss      : {test_loss:.4f}")
        print(f"Test Accuracy  : {test_acc:.4f}")
        print(f"Test Precision : {test_prec:.4f}")
        print(f"Test Recall    : {test_rec:.4f}")
        print(f"Test F1 Score  : {test_f1:.4f}")
        print(f"Best Threshold : {self.best_threshold:.3f}")
        print("=" * 60)

        self._save_all_results(history, test_targets, test_preds)

        return history

    # Save Results

    def _save_all_results(self, history: Dict, y_true: List[int], y_pred: List[int]):
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(config.PLOTS_DIR, exist_ok=True)

        history_df = pd.DataFrame({
            "epoch": range(1, len(history["loss"]) + 1),
            "train_loss": history["loss"],
            "val_loss": history["val_loss"],
            "train_accuracy": history["accuracy"],
            "val_accuracy": history["val_accuracy"],
            "train_precision": history["precision"],
            "val_precision": history["val_precision"],
            "train_recall": history["recall"],
            "val_recall": history["val_recall"],
            "train_f1": history["f1_score"],
            "val_f1": history["val_f1_score"]
        })
        history_df.to_csv(config.TRAINING_HISTORY_CSV, index=False)
        print(f"✓ Training history saved to {config.TRAINING_HISTORY_CSV}")

        self._save_individual_plots(history)
        self._save_confusion_matrix(y_true, y_pred)
        self._save_metrics_json(history)

        print("✓ All results saved successfully!")

    def _save_individual_plots(self, history: Dict):

        plots = [
            ("loss", "val_loss", "Loss", "Loss", config.LOSS_CURVE),
            ("accuracy", "val_accuracy", "Accuracy", "Accuracy", config.ACCURACY_CURVE),
            ("precision", "val_precision", "Precision", "Precision", config.PRECISION_CURVE),
            ("recall", "val_recall", "Recall", "Recall", config.RECALL_CURVE),
            ("f1_score", "val_f1_score", "F1 Score", "F1 Score", config.F1_CURVE)
        ]

        for train_key, val_key, title, ylabel, filename in plots:
            if train_key in history and val_key in history:
                plt.figure(figsize=(8, 6))
                plt.plot(history[train_key], label="Training", linewidth=2, color='#2E86AB')
                plt.plot(history[val_key], label="Validation", linewidth=2, color='#A23B72')
                plt.xlabel("Epoch")
                plt.ylabel(ylabel)
                plt.title(f"Training and Validation {title}")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(filename, dpi=300)
                plt.close()

        print("✓ Individual plots saved.")

    def _save_confusion_matrix(self, y_true: List[int], y_pred: List[int]):

        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Selected", "Selected"]
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(config.CONFUSION_MATRIX_IMAGE, dpi=300)
        plt.close()

        np.savetxt(
            os.path.join(config.RESULTS_DIR, "confusion_matrix.csv"),
            cm, delimiter=",", fmt="%d"
        )

        report = classification_report(
            y_true, y_pred,
            target_names=["Not Selected", "Selected"]
        )
        with open(config.CLASSIFICATION_REPORT, "w", encoding="utf-8") as f:
            f.write(report)

        print("✓ Confusion matrix saved.")

    def _save_metrics_json(self, history: Dict):

        metrics = {
            "training": {
                "loss": float(history["loss"][-1]),
                "accuracy": float(history["accuracy"][-1]),
                "precision": float(history["precision"][-1]),
                "recall": float(history["recall"][-1]),
                "f1": float(history["f1_score"][-1])
            },
            "validation": {
                "loss": float(history["val_loss"][-1]),
                "accuracy": float(history["val_accuracy"][-1]),
                "precision": float(history["val_precision"][-1]),
                "recall": float(history["val_recall"][-1]),
                "f1": float(history["val_f1_score"][-1])
            },
            "testing": {
                "loss": float(history.get("test_loss", 0)),
                "accuracy": float(history.get("test_accuracy", 0)),
                "precision": float(history.get("test_precision", 0)),
                "recall": float(history.get("test_recall", 0)),
                "f1": float(history.get("test_f1", 0)),
                "best_threshold": float(history.get("best_threshold", 0.5))
            }
        }

        with open(config.METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        print(f"✓ Metrics JSON saved to {config.METRICS_JSON}")

    # Summarization (Inference)

    def summarize(self, text: str, num_sentences: int = None) -> str:

        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Please train or load a trained model.")
        
        num_sentences = num_sentences or config.DEFAULT_SUMMARY_SENTENCES
        text = normalize_whitespace(text)
        
        if not text:
            return ""
        
        sentences = split_sentences(text)
        if len(sentences) == 0:
            return ""
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        _, features = self.extract_sentence_features(text)
        features_scaled = self.feature_scaler.transform(features)

        self.model.eval()
        with torch.no_grad():
            tensor_features = torch.from_numpy(features_scaled).float()
            logits = self.model(tensor_features).numpy().flatten()
            predictions = 1 / (1 + np.exp(-logits))

        top_indices = np.argsort(predictions)[-num_sentences:]
        top_indices.sort()
        return " ".join([sentences[i] for i in top_indices])

    # Save / Load
    
    def save_model(self, path: str = None) -> None:

        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save untrained model")
        
        path = path or config.HYBRID_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save(self.model.state_dict(), path)
        
        scaler_path = path.replace('.pt', '_scaler.json')
        scaler_data = {
            'data_min': self.feature_scaler.data_min_.tolist(),
            'data_max': self.feature_scaler.data_max_.tolist(),
            'data_range': self.feature_scaler.data_range_.tolist(),
            'scale': self.feature_scaler.scale_.tolist(),
            'min': self.feature_scaler.min_.tolist(),
            'n_features_in': int(self.feature_scaler.n_features_in_),
            'input_dim': int(self.input_dim) if self.input_dim else 13,
            'best_threshold': float(self.best_threshold)
        }
        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f)
        
        print(f"✓ Model saved to {path}")
        print(f"✓ Scaler saved to {scaler_path}")

    @classmethod
    def load_model(cls, path: str = None) -> 'HybridDeepSummarizer':

        path = path or config.HYBRID_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        summarizer = cls()
        
        scaler_path = path.replace('.pt', '_scaler.json')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
            
            summarizer.feature_scaler = MinMaxScaler()
            summarizer.feature_scaler.scale_ = np.array(scaler_data['scale'])
            summarizer.feature_scaler.data_min_ = np.array(scaler_data['data_min'])
            summarizer.feature_scaler.data_max_ = np.array(scaler_data['data_max'])
            summarizer.feature_scaler.data_range_ = np.array(scaler_data['data_range'])
            summarizer.feature_scaler.min_ = np.array(scaler_data['min'])
            summarizer.feature_scaler.n_features_in_ = int(scaler_data.get('n_features_in', 13))
            
            input_dim = int(scaler_data.get('input_dim', 13))
            summarizer.best_threshold = float(scaler_data.get('best_threshold', 0.5))
        else:
            input_dim = 13
            summarizer.feature_scaler = MinMaxScaler()

        summarizer.model = HybridNet(input_dim=input_dim)
        try:
            state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        except TypeError:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            
        summarizer.model.load_state_dict(state_dict)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        summarizer.model.to(device)
        summarizer.model.eval()
        
        summarizer.is_trained = True
        summarizer.input_dim = input_dim
        
        print(f"✓ Model loaded from {path}")
        print(f"✓ Best threshold: {summarizer.best_threshold:.3f}")
        return summarizer


def batch_summarize_hybrid(df: pd.DataFrame, text_column: str, 
                           summarizer: HybridDeepSummarizer,
                           num_sentences: int = None) -> pd.Series:

    def safe_summarize(text):
        try:
            return summarizer.summarize(text, num_sentences)
        except Exception as e:
            print(f"Warning: {str(e)}")
            return ""
    
    return df[text_column].apply(safe_summarize)