import pandas as pd
import json
from create_datasets.fix_raw_data.fix_raw_data import fix_raw
from create_datasets.create_splits.create_splits_normalized import normalized_split
from create_datasets.fingerprinting.fingerprint_pg import fingerprint_pg

# settings
# Path to the JSON file
json_file_path = "process_dataset.json"

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# read in raw data, fix raw datacalculate and add theoretical IEC, shift time data to start at max
df  = pd.read_csv(data['raw_data'])
props = data['props']
log_time = data['log_time'] # whether or not to log scale time
df = fix_raw(df, props, log_time)
df = df.reset_index(drop=True)

# output location from json file
output = data['output']

# save informatics-ready dataset
df.to_csv(output + f'/dataset.csv')

# create splits
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

