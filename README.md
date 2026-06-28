# 📝 Text Summarization System

## A Hybrid Deep Learning Approach for Extractive Text Summarization

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Models](#models)
- [Feature Engineering](#feature-engineering)
- [Training Pipeline](#training-pipeline)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 Overview

The **Text Summarization System** is an advanced extractive summarization framework that combines traditional statistical methods with state-of-the-art deep learning techniques. The system leverages a hybrid approach, integrating TF-IDF, TextRank, and a custom PyTorch-based neural network to identify and extract the most important sentences from any given text.

This project was developed as a comprehensive solution for automatic text summarization, suitable for processing news articles, research papers, legal documents, and any other form of textual content.

### Why Hybrid Approach?

Traditional extractive summarization methods (TF-IDF, TextRank) are fast but lack deep semantic understanding. Pure deep learning models require large amounts of labeled data. Our hybrid approach combines the best of both worlds:

- **Statistical Methods** (TF-IDF, BM25): Fast, interpretable, and require no training
- **Graph-Based Methods** (TextRank): Capture sentence relationships and importance
- **Deep Learning**: Learn complex patterns and semantic features from data
- **Rich Feature Set**: 13 engineered features for comprehensive sentence evaluation

---

## ✨ Key Features

### Core Functionality
- **Three Summarization Models**:
  - TF-IDF (Statistical)
  - TextRank (Graph-Based)
  - Hybrid Deep Learning (Neural Network)

- **13 Engineered Features** for sentence scoring:
  1. TF-IDF scores
  2. TextRank scores
  3. Sentence position
  4. Sentence length
  5. BM25 scores
  6. Sentence centrality
  7. Sentence entropy
  8. Named Entity Recognition (NER)
  9. Part-of-Speech (POS) features
  10. Binary position encoding
  11. Stopword ratio
  12. Unique word ratio
  13. Sentence embeddings

- **Interactive Web Interface** built with Streamlit:
  - Real-time summarization
  - Model comparison
  - Performance visualization
  - Training history graphs
  - Confusion matrix display

### Advanced Training Features
- **Weak Supervision**: Automatic label generation using ROUGE scores
- **Composite Scoring**: Multi-metric label creation (ROUGE + TextRank + TF-IDF + Position)
- **SMOTE** for class balancing
- **Focal Loss** for handling imbalanced data
- **Dynamic Threshold Optimization**
- **Early Stopping** with patience
- **Learning Rate Scheduling**

---

## 🏗️ Technical Architecture
┌─────────────────────────────────────────────────────────────────┐
│ Text Summarization System │
├─────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│ │ Streamlit │ │ Training │ │ Model Loading │ │
│ │ Web UI │ │ Script │ │ & Inference │ │
│ └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘ │
│ │ │ │ │
│ └───────────────────┼──────────────────────┘ │
│ │ │
│ ┌──────────────────────────┴──────────────────────────────┐ │
│ │ Core Components │ │
│ ├──────────────────────────────────────────────────────────┤ │
│ │ • Preprocessing Pipeline │ │
│ │ • Feature Extraction (13 Features) │ │
│ │ • Neural Network (PyTorch) │ │
│ │ • Evaluation Module (ROUGE Metrics) │ │
│ │ • Visualization Module │ │
│ └──────────────────────────────────────────────────────────┘ │
│ │
│ ┌────────────────────────────────────────────────────────────┐ │
│ │ Sentence Embeddings (Sentence Transformers) │ │
│ │ • all-MiniLM-L6-v2 │ │
│ │ • 384-dimensional embeddings │ │
│ └────────────────────────────────────────────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────┘

text

---

## 🤖 Models

### 1. TF-IDF Summarizer
**Method**: Statistical keyword extraction
- **Strengths**: Fast, interpretable, no training required
- **Weaknesses**: Lacks semantic understanding
- **Use Case**: Quick summarization of short texts

### 2. TextRank Summarizer
**Method**: Graph-based ranking algorithm
- **Strengths**: Captures sentence relationships, good for longer texts
- **Weaknesses**: Computationally intensive for very long texts
- **Use Case**: Summarization of medium-length documents

### 3. Hybrid Deep Learning Summarizer ⭐
**Method**: Neural network with 13 features
- **Architecture**:
  - 4 hidden layers (256 → 128 → 64 → 32)
  - Batch Normalization
  - Dropout (0.15 - 0.30)
  - ReLU activation
  
- **Training**:
  - Focal Loss (γ=2.0, α=0.5)
  - Adam optimizer (lr=0.001)
  - ReduceLROnPlateau scheduler
  - Early stopping (patience=8)
  - SMOTE for class balancing

- **Performance**: ⭐ **Best performing model**
  - Accuracy: **86.98%**
  - Precision: **78.05%**
  - Recall: **85.19%**
  - F1 Score: **81.46%**

---

## 🧬 Feature Engineering

### Comprehensive Feature Set (13 Features)

| # | Feature | Description | Importance |
|---|---------|-------------|------------|
| 1 | **TF-IDF** | Statistical word importance | High |
| 2 | **TextRank** | Graph-based sentence ranking | High |
| 3 | **Position** | Sentence position (0-1 normalized) | Medium |
| 4 | **Length** | Normalized sentence length | Medium |
| 5 | **BM25** | Enhanced TF-IDF variant | Medium |
| 6 | **Centrality** | Sentence similarity centrality | Medium |
| 7 | **Entropy** | Word diversity measure | Low |
| 8 | **NER** | Named entity density | Low |
| 9 | **POS** | Linguistic part-of-speech features | Medium |
| 10 | **Position Binary** | First/last sentence importance | High |
| 11 | **Stopword Ratio** | Function word proportion | Low |
| 12 | **Unique Ratio** | Lexical diversity | Low |
| 13 | **Embedding** | Semantic vector representation | High |

### Composite Label Generation

Instead of relying solely on ROUGE scores, we use a weighted composite:
Composite Score = 0.45 × ROUGE-1 + 0.25 × TextRank + 0.20 × TF-IDF + 0.10 × Position

text

This creates more accurate and robust training labels.

---

## 🔄 Training Pipeline
┌────────────────────────────────────────────────────────────────────┐
│ Training Pipeline │
├────────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│ │ Raw Data │───▶│ Cleaning & │───▶│ Feature │ │
│ │ (CSV) │ │ Preprocessing│ │ Extraction │ │
│ └──────────────┘ └──────────────┘ └──────────────────┘ │
│ │ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│ │ Training │◀───│ Weak Label │◀───│ ROUGE + │ │
│ │ Data │ │ Generation │ │ Composite │ │
│ └──────────────┘ └──────────────┘ └──────────────────┘ │
│ │ │
│ ▼ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│ │ SMOTE │───▶│ Train/Val/ │───▶│ Model │ │
│ │ Balancing │ │ Test Split │ │ Training │ │
│ └──────────────┘ └──────────────┘ └──────────────────┘ │
│ │ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│ │ Model │◀───│ Threshold │◀───│ Focal Loss + │ │
│ │ Evaluation │ │ Optimization│ │ Scheduler │ │
│ └──────────────┘ └──────────────┘ └──────────────────┘ │
│ │ │
│ ▼ │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Results: Metrics, Plots, Confusion Matrix, Reports │ │
│ └──────────────────────────────────────────────────────────────┘ │
│ │
└────────────────────────────────────────────────────────────────────┘

text

---

## 📊 Performance Metrics

### Model Comparison

| Metric | TF-IDF | TextRank | Hybrid DL (Ours) |
|--------|--------|----------|------------------|
| **Accuracy** | ~72% | ~75% | **86.98%** |
| **Precision** | ~25% | ~28% | **78.05%** |
| **Recall** | ~72% | ~74% | **85.19%** |
| **F1 Score** | ~37% | ~41% | **81.46%** |

### Training Performance

| Phase | Loss | Accuracy | Precision | Recall | F1 Score |
|-------|------|----------|-----------|--------|----------|
| **Training** | 0.0407 | 86.10% | 85.93% | 86.33% | 86.13% |
| **Validation** | 0.0411 | 84.94% | 71.67% | 91.25% | 80.28% |
| **Testing** | **0.0377** | **86.98%** | **78.05%** | **85.19%** | **81.46%** |

### Key Achievements
- ✅ **86.98%** Test Accuracy
- ✅ **81.46%** Test F1 Score
- ✅ **96%** Reduction in Loss (0.9023 → 0.0377)
- ✅ **286%** Improvement in Precision (0.20 → 0.78)
- ✅ **156%** Improvement in F1 Score (0.32 → 0.81)

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for faster training)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/text-summarization.git
cd text-summarization
Step 2: Create Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Step 3: Install Dependencies
bash
pip install -r requirements.txt
Step 4: Download NLTK & SpaCy Models
bash
python -c "import nltk; nltk.download('stopwords')"
python -m spacy download en_core_web_sm
Step 5: Set Environment Variables
bash
# Optional: Set Hugging Face token for faster downloads
export HF_TOKEN=your_token_here
Step 6: Verify Installation
bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
💻 Usage
1. Train the Hybrid Model
bash
python train_hybrid_model.py --epochs 25 --batch_size 64 --sample_size 50000
Parameters:

--epochs: Number of training epochs (default: 25)

--batch_size: Batch size for training (default: 64)

--sample_size: Number of samples to use (default: 50000)

2. Launch Streamlit Web Interface
bash
streamlit run streamlit_app.py
Access the interface at: http://localhost:8501

3. Run Comparative Evaluation
bash
python main.py
4. Generate Single Summary
python
from src.hybrid_deep_model import HybridDeepSummarizer

# Load trained model
summarizer = HybridDeepSummarizer.load_model('models/hybrid_model.pt')

# Summarize text
text = "Your long text goes here..."
summary = summarizer.summarize(text, num_sentences=3)
print(summary)
5. Batch Processing
python
import pandas as pd
from src.hybrid_deep_model import HybridDeepSummarizer, batch_summarize_hybrid

# Load model and data
summarizer = HybridDeepSummarizer.load_model('models/hybrid_model.pt')
df = pd.read_csv('data/articles.csv')

# Generate summaries
df['summary'] = batch_summarize_hybrid(df, 'article', summarizer, num_sentences=3)
df.to_csv('data/summarized.csv', index=False)
📁 Project Structure
text
text_summarization/
│
├── 📂 data/
│   ├── my_training_data.csv          # Raw training data
│   └── cleaned_training_data.csv     # Preprocessed data
│
├── 📂 models/
│   ├── hybrid_model.pt               # Trained PyTorch model
│   └── hybrid_model_scaler.json      # Feature scaler
│
├── 📂 results/
│   ├── training_history.csv          # Training logs
│   ├── metrics.json                  # Evaluation metrics
│   ├── classification_report.txt     # Detailed classification report
│   └── 📂 plots/
│       ├── loss_curve.png
│       ├── accuracy_curve.png
│       ├── precision_curve.png
│       ├── recall_curve.png
│       ├── f1_curve.png
│       └── confusion_matrix.png
│
├── 📂 src/
│   ├── __init__.py
│   ├── preprocessing.py              # Data cleaning pipeline
│   ├── summarization.py              # TF-IDF & TextRank models
│   ├── hybrid_deep_model.py          # PyTorch neural network
│   ├── evaluation.py                 # ROUGE metrics & evaluation
│   └── utils.py                      # Helper functions
│
├── 📂 tests/
│   ├── test_preprocessing.py
│   ├── test_summarization.py
│   └── test_hybrid_model.py
│
├── 📄 streamlit_app.py               # Web interface
├── 📄 train_hybrid_model.py          # Training script
├── 📄 main.py                        # Comparison script
├── 📄 config.py                      # Configuration
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # This file
└── 📄 LICENSE                        # MIT License
📈 Results Visualization
Sample Outputs
1. Training Curves
https://results/plots/training_curves.png
Loss, Accuracy, Precision, Recall, and F1 curves over training epochs

2. Confusion Matrix
https://results/plots/confusion_matrix.png
Confusion matrix showing model classification performance

3. Model Comparison
text
Comparison Table:
┌─────────────┬──────────┬──────────┬──────────┐
│ Model       │ ROUGE-1  │ ROUGE-2  │ ROUGE-L  │
├─────────────┼──────────┼──────────┼──────────┤
│ TF-IDF      │ 0.259    │ 0.091    │ 0.185    │
│ TextRank    │ 0.287    │ 0.087    │ 0.185    │
│ Hybrid DL   │ 0.333    │ 0.129    │ 0.213    │
└─────────────┴──────────┴──────────┴──────────┘
🎯 Future Work
Planned Improvements
Abstractive Summarization - Integrating T5 or BART for abstractive generation

Multi-Document Summarization - Extending to handle multiple documents

Real-time Processing - Optimizing for streaming data

More Languages - Adding support for Arabic, French, and other languages

Active Learning - Interactive improvement with user feedback

Research Directions
Transformer-based Hybrid Models - Combining Transformer architectures with extractive methods

Few-shot Learning - Reducing training data requirements

Reinforcement Learning - Optimizing summaries using ROUGE as reward

Explainable AI - Adding interpretability features

🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository

Create a feature branch

bash
git checkout -b feature/amazing-feature
Commit your changes

bash
git commit -m 'Add amazing feature'
Push to the branch

bash
git push origin feature/amazing-feature
Open a Pull Request

Development Guidelines
Follow PEP 8 style guide

Add docstrings for all functions

Write unit tests for new features

Update documentation accordingly

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

text
MIT License

Copyright (c) 2024 Text Summarization System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
📞 Contact & Support
Author
Name: Jalal Ibrahum

Email: jalaleb432@gmail.com


Support
Issues: GitHub Issues

Discussions: GitHub Discussions

🙏 Acknowledgments
Sentence Transformers for excellent pre-trained embeddings

Hugging Face for transformer models and datasets

PyTorch for the deep learning framework

Streamlit for the interactive web interface

All open-source contributors whose libraries made this project possible

📚 References
TextRank: Bringing Order into Text - Mihalcea & Tarau (2004)

ROUGE: A Package for Automatic Evaluation of Summaries - Lin (2004)

Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks - Reimers & Gurevych (2019)

Focal Loss for Dense Object Detection - Lin et al. (2017)

SMOTE: Synthetic Minority Over-sampling Technique - Chawla et al. (2002)

<div align="center"> <strong>⭐ If you found this project useful, please give it a star! ⭐</strong> <br> <br> <sub>Built with ❤️ by the Text Summarization Team</sub> </div> ```
📋 Quick Reference Card
🚀 Quick Start
bash
# Train
python train_hybrid_model.py

# Launch Web Interface
streamlit run streamlit_app.py

# Run Comparison
python main.py
📊 Key Metrics
Test Accuracy: 86.98%

Test F1 Score: 81.46%

Test Precision: 78.05%

Test Recall: 85.19%

🔧 Configuration (config.py)
python
HYBRID_EPOCHS = 25
HYBRID_BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 8
LEARNING_RATE = 1e-3
📦 Dependencies
txt
torch >= 1.9.0
streamlit >= 1.10.0
sentence-transformers >= 2.2.0
scikit-learn >= 0.24.0
This README provides a comprehensive overview of the Text Summarization System. For detailed API documentation, please refer to the source code docstrings.
