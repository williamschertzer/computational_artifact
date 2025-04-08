import pandas as pd
from sklearn.model_selection import train_test_split

def logo_split(df):
    group_columns = [
        "smiles1", "smiles2", "smiles3", "c1", "c2", "c3",
        "Temp(C)", "RH(%)", "solvent", "solvent_conc(M)", "stab_temp",
        'additive_smiles1', 'additive_smiles2', 
        'additive_smiles3', 'additivec1', 'additivec2', 'additivec3',
    ]

    # Identify groups where "time(h)" varies (i.e., has multiple time points)
    time_varying_groups = df.groupby(group_columns).filter(lambda g: g["time(h)"].nunique() > 1)

    # Initialize list to store splits
    splits = []

    # Iterate over unique time-varying groups
    for name, test_group in time_varying_groups.groupby(group_columns, group_keys=False):
            
        # Remaining data after removing the test group
        train_val = df.drop(test_group.index, errors='ignore')

        # Split train_val into train and validation sets (validation set = 20% of train_val)
        train_split, val_split = train_test_split(train_val, test_size=0.2, random_state=42)

        # Store results for this iteration
        splits.append({
            "train": train_split,
            "val": val_split,
            "test": test_group
        })

    return splits  # Returns a list of dictionaries containing train, val, and test sets for each group
