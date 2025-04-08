import pandas as pd
from sklearn.model_selection import train_test_split

def random_split(df):
    # Split dataset into test set (20%) and remaining (80%)
    train_val, test_set = train_test_split(df, test_size=0.2, random_state=42)
    
    # Split remaining data into 5 train-validation sets (each validation set = 20% of train_val)
    train_val_splits = []
    val_splits = []
    
    for i in range(5):
        train_split, val_split = train_test_split(train_val, test_size=0.2, random_state=42 + i)
        train_val_splits.append(train_split)
        val_splits.append(val_split)
    
    return train_val_splits, val_splits, test_set

