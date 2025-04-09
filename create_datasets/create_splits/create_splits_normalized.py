import pandas as pd
from sklearn.model_selection import train_test_split

def normalized_split(df):
    static_columns = [
        "smiles1", "smiles2", "smiles3", "c1", "c2", "c3",
        "Temp(C)", "RH(%)", "stab_temp"
    ]

    # Dynamically find all additive_ and solvent_ columns
    additive_columns = [col for col in df.columns if col.startswith('additive_')]
    solvent_columns = [col for col in df.columns if col.startswith('solvent_')]

    # Combine into final group_columns
    group_columns = static_columns + additive_columns + solvent_columns

    # Identify groups where "time(h)" varies (i.e., has multiple time points)
    time_varying_groups = df.groupby(group_columns).filter(lambda g: g["time(h)"].nunique() > 1)

    # Initialize list to store splits
    splits = []

    # Iterate over unique time-varying groups
    for name, test_group in time_varying_groups.groupby(group_columns, group_keys=False):
            
        # Training and validation on all data
        train_split = df 
        val_split = df

        # Store results for this iteration
        splits.append({
            "train": train_split,
            "val": val_split,
            "test": test_group
        })

    return splits  # Returns a list of dictionaries containing train, val, and test sets for each group
