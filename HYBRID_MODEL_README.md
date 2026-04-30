# Hybrid Deep Learning Summarization Model

## Overview

The **Hybrid Deep Learning Summarizer** is an advanced extractive summarization model that combines multiple feature extraction techniques with a deep neural network. It uses TF-IDF, TextRank, and sentence features to train a model that predicts sentence importance scores.

## Architecture

### Feature Extraction
For each sentence in the input text, the model extracts 5 features:

1. **TF-IDF Score**: Term frequency-inverse document frequency score
   - Captures statistical importance of words in sentences
   - Normalized by sentence length to avoid bias toward long sentences

2. **TextRank Score**: PageRank-based importance score
   - Computed from sentence similarity graph
   - Identifies sentences similar to other important sentences

3. **Position Score**: Normalized position in document (0 to 1)
   - Useful feature as important sentences often appear at document start/end
   - `position = sentence_index / (total_sentences - 1)`

4. **Sentence Length**: Word count of the sentence
   - Raw feature that captures sentence size

5. **Normalized Length**: Length normalized by average sentence length
   - Allows the model to learn length preferences

### Neural Network Architecture

```
Input Layer (5 features)
    ↓
Dense(64, relu)
    ↓
Dropout(0.3)
    ↓
Dense(32, relu)
    ↓
Dropout(0.2)
    ↓
Dense(1, sigmoid)
    ↓
Output (probability 0-1)
```

- **Input**: Features vector of length 5
- **Hidden layers**: 64 → 32 neurons with ReLU activation
- **Dropout**: 30% → 20% for regularization
- **Output**: Sigmoid activation for binary classification (0 = not important, 1 = important)

### Training Process

#### Weak Supervision Labels
The model is trained with automatically generated labels using ROUGE-1 F1 score:

- **Label = 1**: If ROUGE-1 F1-score(sentence, reference_summary) > 0.3
- **Label = 0**: If ROUGE-1 F1-score(sentence, reference_summary) ≤ 0.3

This approach doesn't require manual annotation of important sentences.

#### Loss Function
- **Binary Crossentropy**: Appropriate for sentence importance binary classification
- **Optimizer**: Adam with learning rate 0.001
- **Metrics**: Accuracy, AUC

#### Training Configuration (config.py)
```python
HYBRID_EPOCHS = 20                    # Number of training epochs
HYBRID_BATCH_SIZE = 32                # Batch size
HYBRID_THRESHOLD = 0.5                # Threshold for importance
HYBRID_VALIDATION_SPLIT = 0.2         # 20% validation split
HYBRID_MODEL_PATH = 'models/hybrid_model.keras'
```

## Installation

1. Ensure all dependencies are installed:
```bash
pip install tensorflow tensorflow_hub keras
```

2. The following are already in your requirements:
   - scikit-learn
   - sentence-transformers
   - rouge-score
   - networkx
   - pandas
   - numpy

## Training the Model

### Step 1: Prepare Training Data
Ensure you have a cleaned CSV file with 'article' and 'highlights' columns:
```bash
python src/preprocessing.py
```

This creates `data/cleaned_data.csv`

### Step 2: Train the Model
```bash
python train_hybrid_model.py
```

**Optional arguments:**
```bash
python train_hybrid_model.py --sample_size 1000 --epochs 20 --batch_size 32
```

**Output:**
- Trained model saved to: `models/hybrid_model.keras`
- Scaler metadata saved to: `models/hybrid_model_scaler.json`
- Training history (accuracy, loss, etc.)

### Training Time Estimate
- ~3-5 minutes on CPU for 1000 samples
- ~1-2 minutes on GPU for 1000 samples

## Usage

### Method 1: Using in Python Scripts

```python
from src.hybrid_deep_model import HybridDeepSummarizer

# Load trained model
summarizer = HybridDeepSummarizer.load_model('models/hybrid_model.keras')

# Summarize text
text = "Your long text here..."
summary = summarizer.summarize(text, num_sentences=3)
print(summary)
```

### Method 2: Using with main.py

```bash
python main.py
```

This automatically:
1. Loads all three models (TF-IDF, TextRank, Hybrid)
2. Generates summaries
3. Evaluates all models
4. Shows comparison

### Method 3: Using Streamlit App

```bash
streamlit run streamlit_app.py
```

Then:
1. Select "🧠 Hybrid Deep Learning (متقدم)" from sidebar
2. Adjust "عدد جمل الملخص" (number of sentences)
3. Click "🚀 توليد الملخص" (generate summary)

### Method 4: Batch Processing

```python
from src.hybrid_deep_model import HybridDeepSummarizer, batch_summarize_hybrid
import pandas as pd

# Load model
summarizer = HybridDeepSummarizer.load_model()

# Load data
df = pd.read_csv('data/cleaned_data.csv')

# Generate summaries for entire DataFrame
df['hybrid_summary'] = batch_summarize_hybrid(df, 'article', summarizer, num_sentences=3)

# Save results
df.to_csv('results_with_hybrid.csv', index=False)
```

## API Reference

### HybridDeepSummarizer Class

#### Constructor
```python
__init__(embedding_model: str = config.EMBEDDING_MODEL)
```

#### Methods

##### extract_sentence_features(text: str)
Extracts feature vectors for all sentences in text.

**Parameters:**
- `text` (str): Input text

**Returns:**
- `Tuple[List[str], np.ndarray]`: Sentences and feature matrix

##### create_training_data(df, text_col='article', summary_col='highlights', sample_size=None)
Creates training dataset with weak supervision.

**Parameters:**
- `df` (pd.DataFrame): Training data
- `text_col` (str): Column name for texts
- `summary_col` (str): Column name for reference summaries
- `sample_size` (int, optional): Limit number of samples

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Feature matrix and labels

##### train(X_train, y_train, epochs=None, batch_size=None, validation_split=None, verbose=1)
Trains the neural network model.

**Parameters:**
- `X_train` (np.ndarray): Feature matrix
- `y_train` (np.ndarray): Binary labels
- `epochs` (int): Number of epochs
- `batch_size` (int): Batch size
- `validation_split` (float): Validation split ratio
- `verbose` (int): Verbosity level

**Returns:**
- `Dict`: Training history

##### summarize(text: str, num_sentences: int = None)
Generates extractive summary from text.

**Parameters:**
- `text` (str): Input text to summarize
- `num_sentences` (int): Number of sentences in summary

**Returns:**
- `str`: Extractive summary

**Raises:**
- `RuntimeError`: If model not trained

##### save_model(path: str = None)
Saves trained model and scaler to disk.

**Parameters:**
- `path` (str): Save path (default: config.HYBRID_MODEL_PATH)

##### load_model(path: str = None) [Static]
Loads trained model from disk.

**Parameters:**
- `path` (str): Load path (default: config.HYBRID_MODEL_PATH)

**Returns:**
- `HybridDeepSummarizer`: Loaded instance

**Raises:**
- `FileNotFoundError`: If model file not found

## Performance

### ROUGE Scores (Estimated)
When trained on sufficient data:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| TF-IDF | 0.259 | 0.091 | 0.185 |
| TextRank | 0.287 | 0.087 | 0.185 |
| Hybrid DL | 0.305 | 0.115 | 0.210 |

### Speed
- **TF-IDF**: ~0.05 seconds per article
- **TextRank**: ~0.2 seconds per article
- **Hybrid DL**: ~0.1 seconds per article

## Troubleshooting

### Issue: "Model not trained. Please train the model first"
**Solution:** Train the model using `train_hybrid_model.py`

### Issue: ImportError for tensorflow
**Solution:** Install TensorFlow:
```bash
pip install tensorflow
```

### Issue: Out of Memory
**Solution:** Reduce batch size or sample size:
```bash
python train_hybrid_model.py --batch_size 16 --sample_size 500
```

### Issue: Model not loading in Streamlit
**Solution:** 
1. Check that `models/hybrid_model.keras` exists
2. Check that `models/hybrid_model_scaler.json` exists
3. Retrain the model if files are corrupted

### Issue: Low ROUGE scores
**Solution:**
- Train with more data
- Increase number of epochs
- Adjust ROUGE threshold in config (currently 0.3)

## Configuration

Edit `config.py` to customize:

```python
# Model path
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, 'hybrid_model.keras')

# Training hyperparameters
HYBRID_EPOCHS = 20
HYBRID_BATCH_SIZE = 32
HYBRID_VALIDATION_SPLIT = 0.2

# Inference threshold
HYBRID_THRESHOLD = 0.5

# Random seed for reproducibility
HYBRID_RANDOM_STATE = 42
```

## Advanced Usage

### Custom Feature Extraction
```python
from src.hybrid_deep_model import HybridDeepSummarizer
import numpy as np

summarizer = HybridDeepSummarizer()
text = "Your text..."
sentences, features = summarizer.extract_sentence_features(text)

print(f"Extracted {len(sentences)} sentences")
print(f"Features shape: {features.shape}")
print(f"Features: {features}")
```

### Custom Training
```python
from src.hybrid_deep_model import HybridDeepSummarizer
import pandas as pd

summarizer = HybridDeepSummarizer()
df = pd.read_csv('my_data.csv')

# Create training data
X_train, y_train = summarizer.create_training_data(df, sample_size=2000)

# Train with custom parameters
history = summarizer.train(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.15
)

# Save
summarizer.save_model('my_hybrid_model.keras')
```

### Evaluation
```python
from src.evaluation import evaluate_model
import pandas as pd

df = pd.read_csv('data.csv')
df['hybrid_summary'] = [summarizer.summarize(text) for text in df['article']]

scores = evaluate_model(df, 'hybrid_summary')
print(scores)
```

## Project Structure

```
text_summarization/
├── src/
│   ├── hybrid_deep_model.py      # Main hybrid model implementation
│   ├── summarization.py          # TF-IDF and TextRank models
│   ├── preprocessing.py          # Data cleaning
│   ├── evaluation.py             # ROUGE evaluation
│   └── utils.py                  # Utility functions
├── models/
│   ├── hybrid_model.keras        # Trained model (after training)
│   └── hybrid_model_scaler.json  # Scaler metadata
├── data/
│   ├── my_training_data.csv      # Raw data
│   └── cleaned_data.csv          # Cleaned data
├── config.py                     # Configuration
├── train_hybrid_model.py         # Training script
├── main.py                       # Main demo
├── streamlit_app.py              # Web interface
└── HYBRID_MODEL_README.md        # This file
```

## Future Improvements

- [ ] Abstractive summarization using seq2seq
- [ ] Multi-lingual support
- [ ] Fine-tuning on domain-specific data
- [ ] Ensemble methods combining all three models
- [ ] Real-time model updates
- [ ] GPU optimization
- [ ] Export to ONNX format

## Contributing

To improve the model:

1. Prepare more diverse training data
2. Experiment with different ROUGE thresholds
3. Try different neural network architectures
4. Add more hand-crafted features
5. Test on out-of-domain texts

## References

- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)
- [TextRank: Bringing Order into Texts](http://www.aclweb.org/anthology/W04-3252)
- [TF-IDF Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)

## License

This project is available under the same license as the main text summarization project.
