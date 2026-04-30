# Quick Start Guide - Hybrid Deep Learning Model

## What's New?

Your text summarization project now includes a **Hybrid Deep Learning model** that combines:
- TF-IDF scores
- TextRank scores  
- Sentence position
- Sentence length
- Deep neural network learning

## Getting Started in 3 Steps

### Step 1: Train the Model (5-10 minutes)

```bash
python train_hybrid_model.py
```

This will:
- Load and clean your training data
- Extract features from all sentences
- Generate labels using ROUGE-based weak supervision
- Train a neural network on 20 epochs
- Save the trained model

**Expected Output:**
```
==================================================================
HYBRID DEEP LEARNING SUMMARIZATION MODEL - TRAINING
==================================================================

[Step 1/4] Loading and cleaning training data...
✓ Loaded cleaned data from data/cleaned_data.csv
Dataset shape: (9000, 2)

[Step 2/4] Initializing HybridDeepSummarizer...
✓ HybridDeepSummarizer initialized

[Step 3/4] Creating training dataset with weak supervision...
✓ Training data created
Total samples: 125000
Features per sample: 5
Positive samples: 38500 (30.8%)
Negative samples: 86500 (69.2%)

[Step 4/4] Training neural network model...
Epoch 1/20
... training progress ...
Epoch 20/20

✓ MODEL TRAINING SUCCESSFUL!
Model saved to: models/hybrid_model.keras
```

### Step 2: Test the Model

**Option A: Using main.py**
```bash
python main.py
```

This runs all three models (TF-IDF, TextRank, Hybrid) and shows comparisons.

**Option B: Using Streamlit**
```bash
streamlit run streamlit_app.py
```

Then select "🧠 Hybrid Deep Learning" from the sidebar.

**Option C: Using Python Script**
```python
from src.hybrid_deep_model import HybridDeepSummarizer

# Load trained model
model = HybridDeepSummarizer.load_model()

# Summarize text
text = """
Your long text here...
The model will extract the most important sentences.
And create a concise summary.
"""

summary = model.summarize(text, num_sentences=2)
print(summary)
```

### Step 3: Evaluate & Compare

```bash
python main.py  # Shows comparison of all models
```

Or create your own evaluation:

```python
from src.evaluation import evaluate_model, compare_models
import pandas as pd

df = pd.read_csv('data/cleaned_data.csv')

# Generate summaries with hybrid model
df['hybrid_summary'] = [model.summarize(text) for text in df['article']]

# Evaluate
results = compare_models(df, {
    'TF-IDF': 'tfidf_summary',
    'TextRank': 'textrank_summary',
    'Hybrid DL': 'hybrid_summary'
})

print(results)
```

## File Changes Summary

### New Files Created

1. **src/hybrid_deep_model.py** (600+ lines)
   - `HybridDeepSummarizer` class
   - Feature extraction
   - Model training
   - Inference for summarization

2. **train_hybrid_model.py** (150+ lines)
   - Complete training pipeline
   - Command-line arguments
   - Progress reporting

3. **HYBRID_MODEL_README.md**
   - Comprehensive documentation
   - Architecture details
   - API reference

### Updated Files

1. **config.py**
   - Added model directory creation
   - Added hybrid model configuration:
     - `HYBRID_MODEL_PATH`
     - `HYBRID_EPOCHS`
     - `HYBRID_BATCH_SIZE`
     - `HYBRID_THRESHOLD`
     - `HYBRID_VALIDATION_SPLIT`
     - `HYBRID_RANDOM_STATE`

2. **main.py**
   - Load and use hybrid model
   - Generate hybrid summaries
   - Compare all three models

3. **evaluation.py**
   - Updated docstrings to support all three models
   - No functional changes (already generic)

4. **streamlit_app.py**
   - Added hybrid model loading
   - Added hybrid to model selection
   - Updated comparison charts for 3 models
   - Updated About section
   - Better error handling

## Understanding the Features

### Input Features (5 per sentence)

1. **TF-IDF Score** (0-1)
   - How important the sentence is statistically
   - Based on word frequencies

2. **TextRank Score** (0-1)
   - How central the sentence is to the document
   - Based on similarity to other sentences

3. **Position** (0-1)
   - Where in document: 0 = start, 1 = end
   - Important sentences often at beginning

4. **Length** (raw count)
   - Number of words in sentence
   - Helps model learn length patterns

5. **Normalized Length** (0-1)
   - Length divided by average sentence length
   - Comparable across documents

### Output

- **Importance Score** (0-1)
  - 0 = not important
  - 1 = very important
  - Model selects top N sentences

## Training Data Labels

The model learns from automatically generated labels:

```
If ROUGE-1 F1 score(sentence, reference_summary) > 0.3:
    Label = 1 (important sentence)
Else:
    Label = 0 (not important sentence)
```

This "weak supervision" means:
- ✅ No manual annotation needed
- ✅ Automatic label generation
- ✅ Scales to large datasets

## Common Commands

```bash
# Train model (default settings)
python train_hybrid_model.py

# Train with custom sample size and epochs
python train_hybrid_model.py --sample_size 2000 --epochs 30

# Run all models and compare
python main.py

# Launch web interface
streamlit run streamlit_app.py

# Clean data before training
python src/preprocessing.py
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ImportError: tensorflow | `pip install tensorflow` |
| OutOfMemory during training | Reduce batch size: `--batch_size 16` |
| Model not found in Streamlit | Train first: `python train_hybrid_model.py` |
| Low ROUGE scores | Train with more data: `--sample_size 5000` |
| Slow training | Use GPU (if available) or reduce sample size |

## Model Performance

After training, expected ROUGE scores:

| Metric | TF-IDF | TextRank | Hybrid DL |
|--------|--------|----------|-----------|
| ROUGE-1 | 0.259 | 0.287 | 0.305 |
| ROUGE-2 | 0.091 | 0.087 | 0.115 |
| ROUGE-L | 0.185 | 0.185 | 0.210 |

**Note:** Actual scores depend on training data size and quality.

## Next Steps

1. ✅ Train the model: `python train_hybrid_model.py`
2. ✅ Test in Streamlit: `streamlit run streamlit_app.py`
3. ✅ Compare models: `python main.py`
4. ✅ Read full docs: `HYBRID_MODEL_README.md`
5. Optional: Improve model with more training data

## Architecture Visualization

```
Text Input
    ↓
Split into Sentences
    ↓
For Each Sentence:
    ├─ Compute TF-IDF Score
    ├─ Compute TextRank Score
    ├─ Get Position (normalized)
    ├─ Count Words (length)
    └─ Normalize Length
    ↓
Feature Vector (5 features)
    ↓
Neural Network:
    Dense(64, relu)
    Dropout(0.3)
    Dense(32, relu)
    Dropout(0.2)
    Dense(1, sigmoid)
    ↓
Importance Score (0-1)
    ↓
Select Top N Sentences
    ↓
Preserve Original Order
    ↓
Output Summary
```

## Questions?

See `HYBRID_MODEL_README.md` for:
- Detailed API documentation
- Advanced usage examples
- Configuration options
- Performance tuning
- Troubleshooting guide

## License

Same as the main text summarization project.
