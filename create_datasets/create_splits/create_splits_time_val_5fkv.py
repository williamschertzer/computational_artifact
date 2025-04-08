import pandas as pd
import numpy as np
import os
import networkx as nx

def split_5_fold_independent(df, output_dir):
    """ 
    Performs 5-fold cross-validation ensuring:
    - Test set contains only time-dependent data (20% of total time-dependent data).
    - Remaining data is split into 5 roughly equal sets using a bipartite graph.
    - Each validation set contains only time-dependent data from one of the 5 sets.
    - Each training set contains all other available data.
    - Connected components remain intact, preventing data leakage.
    """

    def time_based_mask(df_subset):
        """ Returns a boolean mask for time-dependent data points. """
        return (df_subset['time (h)'] != 0) | ((df_subset['time (h)'] == 0) & (df_subset['solvent'].notna()))

    def create_bipartite_graph(df, smiles_columns):
        """ Creates a bipartite graph where rows are connected to their SMILES components. """
        B = nx.Graph()
        B.add_nodes_from(df.index, bipartite='rows')

        for idx, row in df.iterrows():
            for col in smiles_columns:
                smile = row[col]
                if pd.notnull(smile) and (smile != '*CC*'):
                    B.add_node(smile, bipartite='smiles')
                    B.add_edge(idx, smile)

        return B

    def select_independent_sets(df, independent_row_sets, num_splits):
        """ Splits data into roughly equal independent sets. """
        sets = [set() for _ in range(num_splits)]
        set_sizes = [0] * num_splits
        sorted_components = sorted(independent_row_sets, key=lambda x: len(x), reverse=True)

        for component in sorted_components:
            smallest_idx = set_sizes.index(min(set_sizes))
            sets[smallest_idx].update(component)
            set_sizes[smallest_idx] += len(component)

        return sets

    # Reset index
    df = df.reset_index(drop=True)
    smiles_columns = ['smiles1', 'smiles2', 'smiles3', 'additive_smiles1', 'additive_smiles2']
    total_time_dep = df[time_based_mask(df)].shape[0]

    # Create bipartite graph and find independent components
    B = create_bipartite_graph(df, smiles_columns)
    components = list(nx.connected_components(B))
    independent_row_sets = [set(comp).intersection(df.index) for comp in components if set(comp).intersection(df.index)]

    # --------------------- Step 1: Extract Test Set (20%) ---------------------
    test_threshold = 0.20 * total_time_dep
    selected_test_sets = []
    test_indices = set()
    cumulative_count = 0

    for row_set in independent_row_sets:
        df_subset = df.loc[list(row_set)]
        time_dependent_count = df_subset[time_based_mask(df_subset)].shape[0]
        
        selected_test_sets.append(row_set)
        test_indices.update(row_set)
        cumulative_count += time_dependent_count
        
        if cumulative_count >= test_threshold:
            break

    # Extract only time-dependent data for the test set
    df_selected_test = df.loc[list(test_indices)]
    test_set = df_selected_test[time_based_mask(df_selected_test)]
    
    # The remaining data consists of all non-time-dependent data and unselected data
    train_candidates_from_test = df_selected_test[~time_based_mask(df_selected_test)]
    remaining_indices = set(df.index) - test_indices
    remaining_data = pd.concat([train_candidates_from_test, df.loc[list(remaining_indices)]], ignore_index=True)

    print(f"Test Set size: {test_set.shape[0]}, Remaining Data: {remaining_data.shape[0]}")

    # Save the test set
    os.makedirs(output_dir, exist_ok=True)
    test_set.to_csv(os.path.join(output_dir, "test_set.csv"), index=False)

    # --------------------- Step 2: Split Remaining Data into 5 Equal Sets ---------------------
    B_remaining = create_bipartite_graph(remaining_data, smiles_columns)
    components_remaining = list(nx.connected_components(B_remaining))
    independent_remaining_sets = [set(comp).intersection(remaining_data.index) for comp in components_remaining if set(comp).intersection(remaining_data.index)]

    five_sets = select_independent_sets(remaining_data, independent_remaining_sets, 5)

    # --------------------- Step 3: Create 5-Fold Cross-Validation ---------------------
    for fold in range(5):
        val_indices = five_sets[fold]
        df_val_set = remaining_data.loc[list(val_indices)]
        val_set = df_val_set[time_based_mask(df_val_set)]  # Extract only time-dependent data
        
        # Train set is all remaining data excluding validation set
        train_set = remaining_data.drop(val_indices)

        # Save the train and validation splits
        train_set.to_csv(os.path.join(output_dir, f"train_fold_{fold + 1}.csv"), index=False)
        val_set.to_csv(os.path.join(output_dir, f"val_fold_{fold + 1}.csv"), index=False)

        print(f"Fold {fold + 1} - Train size: {train_set.shape[0]}, Validation size: {val_set.shape[0]}")

    print("\n5-Fold Cross Validation Splits Completed!")
