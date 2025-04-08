import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from create_datasets.fix_raw_data.fix_raw_data import fix_raw
from create_datasets.prepare_datasets.prepare_data import prepare_dataset
from create_datasets.create_splits.create_splits_random_val import split_random_val
from create_datasets.create_splits.create_splits import split
from create_datasets.create_splits.random_splits import random_split
from create_datasets.create_splits.create_splits_logo import logo_split
from create_datasets.create_splits.create_splits_normalized import normalized_split
from create_datasets.fingerprinting.fingerprint_pg import fingerprint_pg
import os

# settings
# Path to the JSON file
json_file_path = "process_dataset.json"

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# read in raw data, fix raw data
df  = pd.read_csv(data['raw_data'])
props = data['props']
df = fix_raw(df, props)

# calculate and add theoretical IEC, shift time data to start at max
log_time = data['log_time'] # whether or not to log scale time
df = prepare_dataset(df, log=log_time)
df = df.reset_index(drop=True)

# create splits
output = data['output']
test_output = output + 'test.csv'
splits = normalized_split(df)

# extract the first split
split = splits[0]

train_set = split['train']
val_set = split['val']
test_set = split['test']

# save un-fingerprinted datasets
train_set.to_csv(output + f'/train.csv')


# fingerprint datasets
fp_scaler_path = output + f'/fp_scaler.pkl'
prop_scaler_path = output + f'/prop_scaler.pkl'

fp_train_df, fp_val_df, fp_test_df = fingerprint_pg(train_df=train_set, val_df = val_set,test_df = test_set, fp_scaler_path=fp_scaler_path, prop_scaler_path=prop_scaler_path)

# save fingerprinted datasets
train_output = output + f'/fp_train.csv'
fp_train_df.to_csv(train_output)

