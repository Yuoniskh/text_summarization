# train_hybrid_model.py
"""
Training script for the Hybrid Deep Learning Summarization Model.

This script:
1. Loads cleaned training data
2. Creates HybridDeepSummarizer instance
3. Creates training dataset with weak supervision (ROUGE-based labels)
4. Trains the neural network model (PyTorch)
5. Saves the trained model to disk

Usage:
    python train_hybrid_model.py [--sample_size 1000] [--epochs 20] [--batch_size 64]
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
from src.preprocessing import load_and_clean_data
from src.hybrid_deep_model import HybridDeepSummarizer
import config


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Deep Learning Summarizer")
    parser.add_argument('--sample_size', type=int, 
                       default=config.HYBRID_TRAINING_SAMPLE_SIZE,
                       help=f'Number of samples to train on (default: {config.HYBRID_TRAINING_SAMPLE_SIZE:,})')
    parser.add_argument('--epochs', type=int, default=config.HYBRID_EPOCHS,
                       help=f'Number of training epochs (default: {config.HYBRID_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=config.HYBRID_BATCH_SIZE,
                       help=f'Batch size (default: {config.HYBRID_BATCH_SIZE})')
    parser.add_argument('--use_smote', action='store_true', 
                   default=config.USE_SMOTE,
                   help='Use SMOTE for data balancing')
    args = parser.parse_args()
    
    print("=" * 70)
    print("HYBRID DEEP LEARNING SUMMARIZATION MODEL - TRAINING (PyTorch)")
    print("=" * 70)
    
    # Step 1: Load cleaned data
    print("\n[Step 1/5] Loading and cleaning training data...")
    try:
        if os.path.exists(config.CLEANED_DATA_PATH):
            df = pd.read_csv(config.CLEANED_DATA_PATH)
            print(f"✓ Loaded cleaned data from {config.CLEANED_DATA_PATH}")
        else:
            print(f"⚠ Cleaned data not found at {config.CLEANED_DATA_PATH}")
            print("  Generating cleaned data from raw data...")
            df = load_and_clean_data()
            df.to_csv(config.CLEANED_DATA_PATH, index=False, encoding='utf-8')
            print(f"✓ Saved cleaned data to {config.CLEANED_DATA_PATH}")
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        sys.exit(1)
    
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    if args.sample_size and args.sample_size < len(df):
        print(f"  Using sample size: {args.sample_size:,} ({100*args.sample_size/len(df):.1f}% of data)")
    
    if 'article' not in df.columns or 'highlights' not in df.columns:
        print("✗ Dataset must have 'article' and 'highlights' columns")
        sys.exit(1)
    
    # Step 2: Initialize summarizer
    print("\n[Step 2/5] Initializing HybridDeepSummarizer...")
    try:
        summarizer = HybridDeepSummarizer()
        print("✓ HybridDeepSummarizer initialized")
        print(f"  Embedding model: {config.EMBEDDING_MODEL}")
    except Exception as e:
        print(f"✗ Error initializing summarizer: {str(e)}")
        sys.exit(1)
    
    # Step 3: Create training data
    print("\n[Step 3/5] Creating training dataset with weak supervision...")
    print(f"  Using ROUGE-1 score > 0.3 as positive label")
    try:
        X_train, y_train = summarizer.create_training_data(
            df,
            sample_size=args.sample_size
        )
        
        if X_train.size == 0 or y_train.size == 0:
            print("✗ No training data generated")
            sys.exit(1)
        
        print(f"✓ Training data created")
        print(f"  Total samples: {X_train.shape[0]:,}")
        print(f"  Features per sample: {X_train.shape[1]}")
        print(f"  Positive samples: {np.sum(y_train)} ({100*np.mean(y_train):.1f}%)")
        print(f"  Negative samples: {len(y_train) - np.sum(y_train)} ({100*(1-np.mean(y_train)):.1f}%)")
    
    except Exception as e:
        print(f"✗ Error creating training data: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Train model
    print("\n[Step 4/5] Training neural network model...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    
    try:
        history = summarizer.train(
            X_train, y_train, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            verbose=1
        )
        # عرض النتائج النهائية
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        
        # نتائج التدريب
        if 'accuracy' in history and history['accuracy']:
            print(f"\nTraining Results (Last Epoch):")
            print(f"  Accuracy  : {history['accuracy'][-1]:.4f}")
            print(f"  Precision : {history['precision'][-1]:.4f}")
            print(f"  Recall    : {history['recall'][-1]:.4f}")
            print(f"  F1-Score  : {history['f1_score'][-1]:.4f}")
            print(f"  Loss      : {history['loss'][-1]:.4f}")
        
        # نتائج التحقق
        if 'val_accuracy' in history and history['val_accuracy']:
            print(f"\nValidation Results (Last Epoch):")
            print(f"  Accuracy  : {history['val_accuracy'][-1]:.4f}")
            print(f"  Precision : {history['val_precision'][-1]:.4f}")
            print(f"  Recall    : {history['val_recall'][-1]:.4f}")
            print(f"  F1-Score  : {history['val_f1_score'][-1]:.4f}")
            print(f"  Loss      : {history['val_loss'][-1]:.4f}")
        
        # نتائج الاختبار
        if 'test_accuracy' in history:
            print(f"\nTest Results:")
            print(f"  Accuracy  : {history['test_accuracy']:.4f}")
            print(f"  Precision : {history['test_precision']:.4f}")
            print(f"  Recall    : {history['test_recall']:.4f}")
            print(f"  F1-Score  : {history['test_f1']:.4f}")
            print(f"  Loss      : {history['test_loss']:.4f}")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"✗ Error training model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Save model
    print("\n[Step 5/5] Saving trained model...")
    try:
        summarizer.save_model(config.HYBRID_MODEL_PATH)
        print("\n" + "=" * 70)
        print("✓ MODEL TRAINING SUCCESSFUL!")
        print("=" * 70)
        print(f"Model saved to: {config.HYBRID_MODEL_PATH}")
        print(f"Scaler saved to: {config.HYBRID_MODEL_PATH.replace('.pt', '_scaler.json')}")
        print(f"\nSaved files:")
        print(f"  - Training history: {config.TRAINING_HISTORY_CSV}")
        print(f"  - Metrics JSON: {config.METRICS_JSON}")
        print(f"  - Classification report: {config.CLASSIFICATION_REPORT}")
        print(f"  - Confusion matrix: {config.CONFUSION_MATRIX_IMAGE}")
        print(f"  - Loss curve: {config.LOSS_CURVE}")
        print(f"  - Accuracy curve: {config.ACCURACY_CURVE}")
        print(f"  - Precision curve: {config.PRECISION_CURVE}")
        print(f"  - Recall curve: {config.RECALL_CURVE}")
        print(f"  - F1 curve: {config.F1_CURVE}")
        
        print("\nYou can now use the model with:")
        print("  from src.hybrid_deep_model import HybridDeepSummarizer")
        print(f"  summarizer = HybridDeepSummarizer.load_model('{config.HYBRID_MODEL_PATH}')")
        print("  summary = summarizer.summarize(text)")
    
    except Exception as e:
        print(f"✗ Error saving model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ ALL DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()