import pandas as pd
import numpy as np
import networkx as nx

def split(df, train_output, val_output, test_output):

    def time_based_mask(df_subset):
        return (df_subset['time (h)'] != 0) | ((df_subset['time (h)'] == 0) & (df_subset['solvent'].notna()))

    # Reset index
    df = df.reset_index(inplace=False)
    smiles_columns = ['smiles1', 'smiles2', 'smiles3', 'additive_smiles1', 'additive_smiles2']

    # Count total time-dependent datapoints
    total_time_dep = df[time_based_mask(df)].shape[0]
    print("Total time-dependent datapoints in dataset:", total_time_dep)

    # Create an empty bipartite graph
    B = nx.Graph()
    B.add_nodes_from(df.index, bipartite='rows')

    # Add SMILES nodes
    for idx, row in df.iterrows():
        for col in smiles_columns:
            smile = row[col]
            if pd.notnull(smile) and (smile != '*CC*'):
                B.add_node(smile, bipartite='smiles')
                B.add_edge(idx, smile)

    # Compute connected components
    components = list(nx.connected_components(B))
    independent_row_sets = [set(comp).intersection(df.index) for comp in components if set(comp).intersection(df.index)]

    # --------------------- Step 1: Test Set ---------------------
    test_threshold = 0.20 * total_time_dep
    selected_test_sets = []
    cumulative_test_count = 0

    for row_set in independent_row_sets:
        df_temp = df.loc[list(row_set)]
        count_temp = df_temp[time_based_mask(df_temp)].shape[0]
        selected_test_sets.append(row_set)
        cumulative_test_count += count_temp
        if cumulative_test_count >= test_threshold:
            break

    selected_test_indices = set()
    for row_set in selected_test_sets:
        selected_test_indices.update(row_set)

    df_selected_test = df.loc[list(selected_test_indices)]
    initial_test_set = df_selected_test[time_based_mask(df_selected_test)]
    train_candidates_from_selected = df_selected_test[~time_based_mask(df_selected_test)]
    remaining_indices = set(df.index) - selected_test_indices
    overall_train_set = pd.concat([train_candidates_from_selected, df.loc[list(remaining_indices)]])
    overall_train_set = overall_train_set.drop_duplicates()
    
    print("Initial Test Set size:", initial_test_set.shape[0])
    print("Overall Train Set size:", overall_train_set.shape[0])

    # --------------------- Step 2: Four Validation Sets ---------------------
    remaining_time_dep = overall_train_set[time_based_mask(overall_train_set)].shape[0]
    val_threshold = 0.20 * remaining_time_dep  # Each validation set needs ~20% of the remaining time-dependent data
    validation_sets = []
    
    remaining_train_set = overall_train_set.copy()

    for i in range(4):  # Create the first 4 validation sets
        if remaining_train_set.shape[0] < val_threshold:
            print(f"Not enough data left for validation set {i+1}, stopping early.")
            break
        
        # Create a bipartite graph for the remaining data
        B_val = nx.Graph()
        B_val.add_nodes_from(remaining_train_set.index, bipartite='rows')

        for idx, row in remaining_train_set.iterrows():
            for col in smiles_columns:
                smile = row[col]
                if pd.notnull(smile) and (smile != '*CC*'):
                    B_val.add_node(smile, bipartite='smiles')
                    B_val.add_edge(idx, smile)

        components_val = list(nx.connected_components(B_val))
        independent_row_sets_val = [set(comp).intersection(remaining_train_set.index) 
                                    for comp in components_val if set(comp).intersection(remaining_train_set.index)]
        
        selected_val_sets = []
        cumulative_val_count = 0

        for row_set in independent_row_sets_val:
            df_temp = remaining_train_set.loc[list(row_set)]
            count_temp = df_temp[time_based_mask(df_temp)].shape[0]
            selected_val_sets.append(row_set)
            cumulative_val_count += count_temp
            if cumulative_val_count >= val_threshold:
                break

        selected_val_indices = set()
        for row_set in selected_val_sets:
            selected_val_indices.update(row_set)

        validation_set = remaining_train_set.loc[list(selected_val_indices)]
        validation_sets.append(validation_set)

        # Remove validation data from remaining train set
        remaining_train_set = remaining_train_set.drop(selected_val_indices)

        print(f"Validation Set {i+1} size:", validation_set.shape[0])

    # --------------------- Step 3: Fifth Validation Set (All Remaining Time-Dependent Data) ---------------------
    remaining_time_dep_set = remaining_train_set[time_based_mask(remaining_train_set)]
    validation_sets.append(remaining_time_dep_set)
    remaining_train_set = remaining_train_set.drop(remaining_time_dep_set.index)  # Remove from training set

    print(f"Validation Set 5 size (all remaining time-dependent data):", remaining_time_dep_set.shape[0])

    # --------------------- Step 4: Generate Training Set for Each Fold ---------------------
    final_train_sets = []

    for i in range(len(validation_sets)):  # Iterate through validation sets
        validation_indices = validation_sets[i].index
        train_set_for_fold = overall_train_set.drop(validation_indices)  # Exclude current validation set
        final_train_sets.append(train_set_for_fold)
        print(f"Training Set {i+1} size (excluding Validation Set {i+1}):", train_set_for_fold.shape[0])

    # --------------------- Save Outputs ---------------------
    test_output = test_output + '.csv'
    initial_test_set.to_csv(test_output)

    for i, (train_set, val_set) in enumerate(zip(final_train_sets, validation_sets)):
        train_filename = train_output + f"_{i}.csv"
        val_filename = val_output + f"_{i}.csv"
        
        train_set.to_csv(train_filename)
        val_set.to_csv(val_filename)
        
        print(f"Saved Training Set {i+1} to {train_filename}")
        print(f"Saved Validation Set {i+1} to {val_filename}")

    print("\n--- Summary ---")
    print("Initial Test Set size:", initial_test_set.shape[0])
    for i, (train_set, val_set) in enumerate(zip(final_train_sets, validation_sets)):
        print(f"Training Set {i+1} size:", train_set.shape[0])
        print(f"Validation Set {i+1} size:", val_set.shape[0])
