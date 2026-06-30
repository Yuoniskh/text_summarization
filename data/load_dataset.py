from datasets import load_dataset
import pandas as pd


def load_cnn_dataset():
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    return dataset

def save_to_csv(dataset, filename):
    df = pd.DataFrame(dataset['train']).sample(n = 10000).reset_index(drop=True) 
    df.to_csv(filename, index=False)
    return df

dataset = load_cnn_dataset()
df = save_to_csv(dataset, './data/my_training_data.csv')
print(df.head)