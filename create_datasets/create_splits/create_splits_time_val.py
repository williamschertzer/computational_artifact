import pandas as pd
import numpy as np
import os
import networkx as nx


def time_based_mask(df_subset):
    return (df_subset['time (h)'] != 0) | ((df_subset['time (h)'] == 0) & (df_subset['solvent'].notna()))


# import dataset
df = pd.read_csv(os.path.join("..", 'shared_datasets', 'dataset_2_7_25.csv'))
df = df.reset_index(inplace=False)
smiles_columns = ['smiles1', 'smiles2', 'smiles3', 'additive_smiles1', 'additive_smiles2']

# Count total time-dependent datapoints in the full dataset
total_time_dep = df[time_based_mask(df)].shape[0]
print("Total time-dependent datapoints in dataset:", total_time_dep)

# Create an empty bipartite graph
B = nx.Graph()

# Add row nodes (using the DataFrame index)
B.add_nodes_from(df.index, bipartite='rows')

# For each row, add SMILES nodes (if not already present) and connect the row to each SMILES string
for idx, row in df.iterrows():
    for col in smiles_columns:
        smile = row[col]
        if pd.notnull(smile) and (smile != '*CC*'):
            # Add the SMILES node with a label for the bipartite set
            B.add_node(smile, bipartite='smiles')
            # Add an edge between the row and the SMILES
            B.add_edge(idx, smile)

# Compute the connected components of the bipartite graph. Each component is an independent set containing both rows and SMILES.
components = list(nx.connected_components(B))

# --------------------- Step 1. Split Off the Test Set ---------------------

# For each component, extract the row nodes only
independent_row_sets = [set(comp).intersection(df.index) for comp in components if set(comp).intersection(df.index)]

# We want to accumulate entire independent sets until the union contains at least 10% of all time-dependent datapoints.
test_threshold = 0.10 * total_time_dep
selected_test_sets = []
cumulative_test_count = 0


for row_set in independent_row_sets:
    df_temp = df.loc[list(row_set)]
    count_temp = df_temp[time_based_mask(df_temp)].shape[0]
    selected_test_sets.append(row_set)
    cumulative_test_count += count_temp
    if cumulative_test_count >= test_threshold:
        break

print("Number of independent sets selected for test:", len(selected_test_sets))
print("Cumulative time-dependent count in selected test sets:", cumulative_test_count)

# Create the union of all indices from the selected independent sets.
selected_test_indices = set()
for row_set in selected_test_sets:
    selected_test_indices.update(row_set)

# In the selected sets, choose as test set those rows that meet the time-based condition.
df_selected_test = df.loc[list(selected_test_indices)]
initial_test_set = df_selected_test[time_based_mask(df_selected_test)]
print("Initial Test Set size:", initial_test_set.shape[0])

# The remaining rows in the selected sets (those that do NOT meet the time-based condition) will be added to the train set.
train_candidates_from_selected = df_selected_test[~time_based_mask(df_selected_test)]
remaining_indices = set(df.index) - selected_test_indices

# Overall train set is the union of the "train candidates" from the selected sets plus all rows from remaining sets.
overall_train_set = pd.concat([train_candidates_from_selected, df.loc[list(remaining_indices)]])
overall_train_set = overall_train_set.drop_duplicates()  # In case of any overlap
print("Overall Train Set size:", overall_train_set.shape[0])

# --------------------- Step 2. Split Off the Validation Set from the Overall Train Set ---------------------
# First, count time-dependent datapoints in overall_train_set and set a validation threshold (10% of train time-dependent data)
total_time_dep_train = total_time_dep
val_threshold = 0.10 * total_time_dep_train
print("Total time-dependent datapoints in overall train set:", total_time_dep_train)

# Recompute independent sets for the overall_train_set:
B_train = nx.Graph()
train_idx_list = overall_train_set.index.tolist()
B_train.add_nodes_from(train_idx_list, bipartite='rows')
for idx in train_idx_list:
    row = overall_train_set.loc[idx]
    for col in smiles_columns:
        smile = row[col]
        if pd.notnull(smile):
            B_train.add_node(smile, bipartite='smiles')
            B_train.add_edge(idx, smile)
components_train = list(nx.connected_components(B_train))
independent_train_sets = [set(comp).intersection(set(train_idx_list)) for comp in components_train if set(comp).intersection(set(train_idx_list))]


# Accumulate independent sets from the overall_train_set until the cumulative time-dependent count reaches the validation threshold
selected_val_sets = []
cumulative_val_count = 0
for row_set in independent_train_sets:
    df_temp = overall_train_set.loc[list(row_set)]
    count_temp = df_temp[time_based_mask(df_temp)].shape[0]
    selected_val_sets.append(row_set)
    cumulative_val_count += count_temp
    if cumulative_val_count >= val_threshold:
        break

print("Number of independent sets selected for validation:", len(selected_val_sets))
print("Cumulative time-dependent count in selected validation sets:", cumulative_val_count)



# Create the union of indices from the selected validation sets.
selected_val_indices = set()
for row_set in selected_val_sets:
    selected_val_indices.update(row_set)

# In these sets, the rows meeting the time-based condition become the validation set.
df_selected_val = overall_train_set.loc[list(selected_val_indices)]
validation_set = df_selected_val[time_based_mask(df_selected_val)]
print("Validation Set size:", validation_set.shape[0])

# The remaining rows in the selected validation sets (that do NOT meet the time-based condition)
# plus all rows not in the selected validation sets form the final training set.
train_candidates_from_val = df_selected_val[~time_based_mask(df_selected_val)]
remaining_train_indices = set(overall_train_set.index) - selected_val_indices
final_train_set = pd.concat([train_candidates_from_val, overall_train_set.loc[list(remaining_train_indices)]])
final_train_set = final_train_set.drop_duplicates()




# --------------------- Summary of Splits ---------------------
print("\n--- Summary ---")
print("Initial Test Set size:", initial_test_set.shape[0])
print("Validation Set size:", validation_set.shape[0])
print("Final Training Set size:", final_train_set.shape[0])

# Save to CSV
data_dir = '../shared_datasets/time'
os.makedirs(data_dir, exist_ok=True)
final_train_set.to_csv(f'{data_dir}/2_7_25_train.csv', index=False)
initial_test_set.to_csv(f'{data_dir}/2_7_25_test.csv', index=False)
validation_set.to_csv(f'{data_dir}/2_7_25_val.csv', index=False)
