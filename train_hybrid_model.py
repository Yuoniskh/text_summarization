# train_hybrid_model.py
"""
Training script for the Hybrid Deep Learning Summarization Model.

This script:
1. Loads cleaned training data
2. Creates HybridDeepSummarizer instance
3. Creates training dataset with weak supervision (ROUGE-based labels)
4. Trains the neural network model
5. Saves the trained model to disk

Usage:
    python train_hybrid_model.py [--sample_size 1000] [--epochs 20]
"""

import argparse
import pandas as pd
import numpy as np
import os
from src.preprocessing import load_and_clean_data
from src.hybrid_deep_model import HybridDeepSummarizer
import config


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Deep Learning Summarizer")
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to train on (default: all)')
    parser.add_argument('--epochs', type=int, default=config.HYBRID_EPOCHS,
                       help=f'Number of training epochs (default: {config.HYBRID_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=config.HYBRID_BATCH_SIZE,
                       help=f'Batch size (default: {config.HYBRID_BATCH_SIZE})')
    args = parser.parse_args()
    
    print("=" * 70)
    print("HYBRID DEEP LEARNING SUMMARIZATION MODEL - TRAINING")
    print("=" * 70)
    
    # Step 1: Load cleaned data
    print("\n[Step 1/4] Loading and cleaning training data...")
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
        return
    
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    if 'article' not in df.columns or 'highlights' not in df.columns:
        print("✗ Dataset must have 'article' and 'highlights' columns")
        return
    
    # Step 2: Initialize summarizer
    print("\n[Step 2/4] Initializing HybridDeepSummarizer...")
    try:
        summarizer = HybridDeepSummarizer()
        print("✓ HybridDeepSummarizer initialized")
        print(f"  Embedding model: {config.EMBEDDING_MODEL}")
    except Exception as e:
        print(f"✗ Error initializing summarizer: {str(e)}")
        return
    
    # Step 3: Create training data
    print("\n[Step 3/4] Creating training dataset with weak supervision...")
    print(f"  Using ROUGE-1 score > 0.3 as positive label")
    try:
        X_train, y_train = summarizer.create_training_data(
            df,
            sample_size=args.sample_size
        )
        
        if X_train.size == 0 or y_train.size == 0:
            print("✗ No training data generated")
            return
        
        print(f"✓ Training data created")
        print(f"  Total samples: {X_train.shape[0]}")
        print(f"  Features per sample: {X_train.shape[1]}")
        print(f"  Positive samples: {np.sum(y_train)} ({100*np.mean(y_train):.1f}%)")
        print(f"  Negative samples: {len(y_train) - np.sum(y_train)} ({100*(1-np.mean(y_train)):.1f}%)")
    
    except Exception as e:
        print(f"✗ Error creating training data: {str(e)}")
        return
    
    # Step 4: Train model
    print("\n[Step 4/4] Training neural network model...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Validation split: {config.HYBRID_VALIDATION_SPLIT}")
    
    try:
        history = summarizer.train(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=config.HYBRID_VALIDATION_SPLIT,
            verbose=1
        )
        
        # Print training statistics
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        
        if 'accuracy' in history:
            final_acc = history['accuracy'][-1]
            final_val_acc = history['val_accuracy'][-1]
            print(f"Final Training Accuracy: {final_acc:.4f}")
            print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        
        if 'loss' in history:
            final_loss = history['loss'][-1]
            final_val_loss = history['val_loss'][-1]
            print(f"Final Training Loss: {final_loss:.4f}")
            print(f"Final Validation Loss: {final_val_loss:.4f}")
        
    except Exception as e:
        print(f"✗ Error training model: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Save model
    print("\n[Step 5/5] Saving trained model...")
    try:
        summarizer.save_model(config.HYBRID_MODEL_PATH)
        print("=" * 70)
        print("✓ MODEL TRAINING SUCCESSFUL!")
        print("=" * 70)
        print(f"Model saved to: {config.HYBRID_MODEL_PATH}")
        print("\nYou can now use the model with:")
        print("  from src.hybrid_deep_model import HybridDeepSummarizer")
        print(f"  summarizer = HybridDeepSummarizer.load_model('{config.HYBRID_MODEL_PATH}')")
        print("  summary = summarizer.summarize(text)")
    
    except Exception as e:
        print(f"✗ Error saving model: {str(e)}")
        return


if __name__ == "__main__":
    main()
