import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import copy
from sklearn.model_selection import GroupShuffleSplit
import joblib
import random
import networkx as nx
from collections import defaultdict
import os
from pathlib import Path

# Set the number of threads to use
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"


# Load the dataset
df_file = ('')
# df = df.sample(frac = 1)

props_to_consider = ['OHCond(mS/cm)']
prop_names = ['OH',]
splits = ['temperature']
random_seeds = [123, 152, 923, 816, 229]
test_sizes = [0.2]
num_splits = 5
date = ''


columns = pd.read_pickle('')
columns = columns.append(pd.Index(['Temp(C)', 'RH(%)', 'IEC']))


# function to split dataset so test set is known polymers at unknown temepratures
def temperature_split(df_prop, random_seed):

    # Group by chemistry and filter out those with at least 5 different temperatures
    chemistry_groups = df_prop.groupby('pid_composition').filter(lambda x: x['Temp(C)'].nunique() >= 5)

    # Randomly select chemistries for the test set
    test_chemistries = chemistry_groups['pid_composition'].drop_duplicates().sample(1, random_state=random_seed)
    print(test_chemistries)

    test_df_list = []
    train_df_list = []

    for chemistry in test_chemistries:
        chemistry_df = df_prop[df_prop['pid_composition'] == chemistry]

        # Calculate the median temperature for the current chemistry
        median_temp = chemistry_df['Temp(C)'].median()

        # Split the data points based on the median temperature
        test_samples = chemistry_df[chemistry_df['Temp(C)'] > median_temp]
        train_samples = chemistry_df[chemistry_df['Temp(C)'] <= median_temp]

        test_df_list.append(test_samples)
        train_df_list.append(train_samples)

    test_df = pd.concat(test_df_list).reset_index(drop=True)
    train_df = pd.concat(train_df_list).reset_index(drop=True)

    # Add the rest of the chemistries to the training set
    remaining_train_df = df_prop[~df_prop['pid_composition'].isin(test_chemistries)].reset_index(drop=True)
    train_df = pd.concat([train_df, remaining_train_df]).reset_index(drop=True)

    return train_df, test_df

# function to split dataset so test set is known monomers at new copolyme compositions
def composition_split(df_prop, random_seed, test_size, k):
    groups = np.array(df_prop.pid_cpid)
    gss = GroupShuffleSplit(test_size=test_size, random_state=random_seed)
    training_ind, test_ind = next(gss.split(df_prop['Temp(C)'], df_prop[props_to_consider[k]], groups=groups))

    training_df = df_prop.iloc[training_ind].reset_index(drop=True)
    test_df = df_prop.iloc[test_ind].reset_index(drop=True)

    return training_df, test_df

# function to split dataset so test set is unseen monomers
def polymer_split(df_prop, random_seed, test_size):

    def test_no_smiles_overlap_for_same_prop(val_df, train_df):
        # Exclude *CC* from consideration
        exclude_smiles = '*CC*'
        smiles_overlap = []
        # Check if any smiles in val_df are in train_df when 'prop' is the same
        for smiles_col in ['smiles1', 'smiles2', 'smiles3', 'smiles4', 'smiles5']:
            for smiles_col2 in ['smiles1', 'smiles2', 'smiles3', 'smiles4', 'smiles5']:
                for prop in val_df['prop'].unique():
                    train_prop_smiles = set(train_df[(train_df['prop'] == prop) & (train_df[smiles_col] != exclude_smiles)][smiles_col])
                    val_prop_smiles = set(val_df[(val_df['prop'] == prop) & (val_df[smiles_col] != exclude_smiles)][smiles_col2])
                    smiles_overlap.append(train_prop_smiles & val_prop_smiles)                
        return smiles_overlap

    def get_datasets_from_dict(val_dict, test_size):
        # get dataset and scale values
        datasets = []
        datasets_final = []

        # get test_df
        test_df = pd.DataFrame()
        for i in range(0,int((test_size*10))):
            test_df = pd.concat([test_df, val_dict[i]])
        
        # get training_df
        training_df = pd.DataFrame()
        for i in range(int((test_size*10)), 10):
            training_df = pd.concat([training_df, val_dict[i]])


        # print final overlap for test and training
        smiles_overlap = test_no_smiles_overlap_for_same_prop(test_df, training_df)
        print('-------smiles overlap == ', smiles_overlap)


        # removing smiles overlap
        # Identify rows containing the specific value:
        for set in smiles_overlap:
            for smile in set:
                mask = test_df.isin([smile]).any(axis=1)

                # Extract the rows containing the specific value:
                rows_to_move = test_df[mask]

                # Remove these rows from the original DataFrame:
                test_df = test_df[~mask]


                #Add these rows to the other DataFrame:

                training_df = pd.concat([training_df, rows_to_move], ignore_index=True)

        # checking that removeing overlap works
        print('----final overlap after removing----')
        smiles_overlap = test_no_smiles_overlap_for_same_prop(test_df, training_df)
        print(smiles_overlap)
        for set in smiles_overlap:
            for smile in set:
                print('this is smiles overlap remaining: ', smile)
                
        return training_df, test_df

    val_dict = {}
    for i in range(0, 10):
        val_dict[i] = pd.DataFrame()

    smiles_columns = ['smiles1', 'smiles2', 'smiles3', 'smiles4', 'smiles5']
    smiles_df = df_prop[smiles_columns]
    unique_smiles = list(pd.unique(df_prop[smiles_columns].values.ravel('K')))
    random.Random(random_seed).shuffle(unique_smiles)

    # Ensure '*CC*' is in unique_smiles before attempting to remove it
    if '*CC*' in unique_smiles:
        unique_smiles.remove('*CC*')

    # Initialize a graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(unique_smiles)

    # Add edges between SMILES that appear in the same copolymer
    for index, row in smiles_df.iterrows():
        smiles_set = set(smile for smile in row.dropna())
        smiles_set.discard('*CC*')  # discard drops if it is there but does not raise an error if it is not
        for smile1 in smiles_set:
            for smile2 in smiles_set:
                if smile1 != smile2:
                    G.add_edge(smile1, smile2)


    # Number of independent sets desired
    num_indep_sets = 10

    # Function to find independent sets with balanced datapoints
    def find_balanced_independent_sets(graph, num_indep_sets, df, smiles_columns):
        nodes = list(graph.nodes)
        random.Random(random_seed).shuffle(nodes)  # Shuffle nodes to randomize the process
        unique_smiles = list(pd.unique(df[smiles_columns].values.ravel('K')))
        
        independent_sets = [[] for _ in range(num_indep_sets)]
        set_datapoint_count = [0] * num_indep_sets
        
        # Count how many times each monomer appears in the data
        monomer_counts = defaultdict(int)
        for col in smiles_columns:
            for smile in df[col]:
                if smile in unique_smiles:
                    monomer_counts[smile] += 1
        
        assigned_nodes = set()
        
        for node in nodes:
            if node in assigned_nodes:
                continue
            
            # Find the set with the minimum datapoints that can still accept the node
            min_set_index = -1
            min_datapoint_count = float('inf')
            
            for i in range(num_indep_sets):
                if all(neighbor not in independent_sets[i] for neighbor in graph.neighbors(node)):
                    potential_datapoint_count = set_datapoint_count[i] + monomer_counts[node]
                    if potential_datapoint_count < min_datapoint_count:
                        min_set_index = i
                        min_datapoint_count = potential_datapoint_count
            
            if min_set_index != -1:
                independent_sets[min_set_index].append(node)
                set_datapoint_count[min_set_index] = min_datapoint_count
                assigned_nodes.add(node)
                # Ensure that neighbors of the assigned node are not assigned to any set
                for neighbor in graph.neighbors(node):
                    assigned_nodes.add(neighbor)
        
        return independent_sets, set_datapoint_count

    # Get the independent sets
    independent_sets, set_datapoint_count = find_balanced_independent_sets(G, num_indep_sets, smiles_df, smiles_columns)

    # Create DataFrame for each set
    dfs = []
    for independent_set in independent_sets:
        mask = smiles_df.apply(lambda row: any(smile in independent_set for smile in row), axis=1)
        set_df = df_prop[mask]
        dfs.append(set_df)

    # Output the independent sets and their corresponding DataFrames
    for i, (independent_set, count) in enumerate(zip(independent_sets, set_datapoint_count)):
        print(f"Set {i+1}: Number of datapoints: {count}")
        # print(independent_set)
    
    for i in range(len(val_dict)):
        val_dict[i] = dfs[i]

    training_df, test_df = get_datasets_from_dict(val_dict, test_size)

    return training_df, test_df

# Spliting based on only one property
def split_st(df_file, props_to_consider, splits, random_seeds, num_splits, test_sizes, date):

    for k, prop in enumerate(props_to_consider):
        for split in splits:
            for test_size in test_sizes:
                for num_split in range(num_splits):
                    
                    print('prop: ', prop_names[k], 'split: ', split)
                    save_file = f'/data/wschertzer/AEM_PEM_Fuell_cells/analysis/results/GPR/PG/splits/{date}/{prop_names[k]}/{split}/ST/{int(100.0 - test_size * 100)}_{int(test_size*100)}/{num_split}'
                    isexist = os.path.exists(save_file)
                    if not isexist:

                        Path(save_file).mkdir(parents=True, exist_ok=True)

                        random_seed = random_seeds[num_split]

                        df = pd.read_csv(df_file)
                        props = df.prop_name.unique().tolist()
                        target_property_scaler = MinMaxScaler
                        # features_scaler = MinMaxScaler()  # has () because it is an instance of the MinMaxScaler class


                        # Format the dataset
                        df_prop, df_not_prop = format(df, prop)

                        # Split the dataset into train and test by split type

                        if split == 'temperature':
                            # Split the dataset by temperature
                            training_df, test_df = temperature_split(df_prop, random_seed)


                            property_scaler = {}
                            for prop in training_df.prop.unique():
                                property_scaler[prop] = target_property_scaler()

                                # fit transform train
                                cond_train = training_df[training_df['prop'] == prop].index
                                val_train = training_df.loc[cond_train, ['value']]
                                print(val_train)
                                test_train = property_scaler[prop].fit_transform(val_train)
                                training_df.loc[cond_train, ['value']] = test_train
                                
                                # transform test
                                cond_test = test_df[test_df['prop'] == prop].index
                                if len(cond_test) > 0:
                                    val_test = test_df.loc[cond_test, ['value']]
                                    test_test = property_scaler[prop].transform(val_test)  # Use transform, not fit_transform
                                    test_df.loc[cond_test, ['value']] = test_test

                            training_df['test_train'] = 'train'
                            training_df = training_df.reset_index(drop=True)
                            test_df['test_train'] = 'test'
                            test_df = test_df.reset_index(drop = True)

                            df = pd.concat([training_df, test_df], axis = 0)
                            df['true'] = 'yes'

                            
                            df_list = []
                            # Create 30 copies with incremented temperature
                            for i in range(-10,21):
                                temp_df = test_df.copy()
                                temp_df['Temp(C)'] = i * 10
                                df_list.append(temp_df)

                            # Concatenate all DataFrames in the list
                            new_df = pd.concat(df_list, ignore_index=True)
                            new_df['test_train'] = 'unseen_temp'
                            new_df['true'] = 'no'

                            new_df = pd.concat([df, new_df]).reset_index(drop=True)
                            columns_drop = df.columns.drop(['test_train', 'true'])
                            new_df = new_df.drop_duplicates(subset = columns_drop, keep = 'first')
                            new_df = new_df.reset_index(drop=True)

                            print(new_df, new_df.true.unique())

                            # Save the datasets
                            test_df.to_csv(save_file + '/test_df_predict.csv')
                            training_df.to_csv(save_file + '/train_df_predict.csv')
                            new_df.to_csv(save_file + '/training_pg.csv')

                            property_scaler_filename = save_file + '/property_scaler.save'
                            joblib.dump(property_scaler, property_scaler_filename)

                        if split == 'composition':

                            training_df, test_df = composition_split(df_prop, random_seed, test_size, k)
                            print('----training_df---', training_df)
                            print('---df_not_prop---', df_not_prop)
                            training_df = training_df.reset_index(drop=True)
                            print('----training_df----', training_df)


                            property_scaler = {}
                            for prop in training_df.prop.unique():
                                property_scaler[prop] = target_property_scaler()

                                # fit transform train
                                cond_train = training_df[training_df['prop'] == prop].index
                                val_train = training_df.loc[cond_train, ['value']]
                                print(val_train)
                                test_train = property_scaler[prop].fit_transform(val_train)
                                training_df.loc[cond_train, ['value']] = test_train
                                
                                # transform test
                                cond_test = test_df[test_df['prop'] == prop].index
                                if len(cond_test) > 0:
                                    val_test = test_df.loc[cond_test, ['value']]
                                    test_test = property_scaler[prop].transform(val_test)  # Use transform, not fit_transform
                                    test_df.loc[cond_test, ['value']] = test_test
                            


                            training_df = training_df.reset_index(drop=True)
                            test_df = test_df.reset_index(drop=True)

                            training_df['id'] = training_df.index
                            test_df['id'] = test_df.index


                            print('----training_df----', training_df)
                            print('----test_df----', test_df)

                            training_pg = pd.concat([training_df, test_df], axis = 0, ignore_index=True)
                            training_pg['id'] = training_pg.index
                            print('----training_pg----', training_pg)

                            training_df.to_csv(save_file + '/training_df.csv')
                            test_df.to_csv(save_file + '/test_df.csv')
                            training_pg.to_csv(save_file + '/training_pg.csv')

                            property_scaler_filename = save_file + '/property_scaler.save'
                            joblib.dump(property_scaler, property_scaler_filename)    

                        if split == 'polymer':
                            training_df, test_df = polymer_split(df_prop, random_seed, test_size)

                            print('----training_df---', training_df)
                            print('---df_not_prop---', df_not_prop)
                            training_df = training_df.reset_index(drop=True)
                            print('----training_df----', training_df)

                            property_scaler = {}
                            for prop in training_df.prop.unique():
                                property_scaler[prop] = target_property_scaler()

                                # fit transform train
                                cond_train = training_df[training_df['prop'] == prop].index
                                val_train = training_df.loc[cond_train, ['value']]
                                test_train = property_scaler[prop].fit_transform(val_train)
                                training_df.loc[cond_train, ['value']] = test_train
                                
                                # transform test
                                cond_test = test_df[test_df['prop'] == prop].index
                                if len(cond_test) > 0:
                                    val_test = test_df.loc[cond_test, ['value']]
                                    test_test = property_scaler[prop].transform(val_test)  # Use transform, not fit_transform
                                    test_df.loc[cond_test, ['value']] = test_test
                            


                            training_df = training_df.reset_index(drop=True)
                            test_df = test_df.reset_index(drop=True)

                            training_df['id'] = training_df.index
                            test_df['id'] = test_df.index


                            training_pg = pd.concat([training_df, test_df], axis = 0, ignore_index=True)
                            training_pg['id'] = training_pg.index

                            training_df.to_csv(save_file + '/training_df.csv')
                            test_df.to_csv(save_file + '/test_df.csv')
                            training_pg.to_csv(save_file + '/training_pg.csv')

                            property_scaler_filename = save_file + '/property_scaler.save'
                            joblib.dump(property_scaler, property_scaler_filename)    


# Extending the Single Task (ST) splitting to Multi-Task (MT) setting:
# In the MT case, after creating the train and test split for a given property, the remaining data for all other properties is added to the training set.
def split_mt(df_file, props_to_consider, splits, random_seeds, num_splits, test_sizes, date):

    for k, prop in enumerate(props_to_consider):
        for split in splits:
            for test_size in test_sizes:
                for num_split in range(num_splits):
                    
                    print('prop: ', prop_names[k], 'split: ', split)
                    save_file = f'/data/wschertzer/AEM_PEM_Fuell_cells/analysis/results/GPR/PG/splits/{date}/{prop_names[k]}/{split}/MT/{int(100.0 - test_size * 100)}_{int(test_size*100)}/{num_split}'
                    isexist = os.path.exists(save_file)
                    if not isexist:

                        Path(save_file).mkdir(parents=True, exist_ok=True)

                        random_seed = random_seeds[num_split]

                        df = pd.read_csv(df_file)
                        props = df.prop_name.unique().tolist()
                        target_property_scaler = MinMaxScaler
                        # features_scaler = MinMaxScaler()  # has () because it is an instance of the MinMaxScaler class


                        # Format the dataset
                        df_prop, df_not_prop = format(df, prop)

                        # Split the dataset into train and test by split type

                        if split == 'temperature':
                            # Split the dataset by temperature
                            training_df, test_df = temperature_split(df_prop, random_seed)

                            training_df = pd.concat([df_not_prop, training_df], axis=0, ignore_index=True)
                            training_df = training_df.reset_index(drop=True)
                            training_df['true'] = 'yes'


                            property_scaler = {}
                            for prop in training_df.prop.unique():
                                property_scaler[prop] = target_property_scaler()

                                # fit transform train
                                cond_train = training_df[training_df['prop'] == prop].index
                                val_train = training_df.loc[cond_train, ['value']]
                                print(val_train)
                                test_train = property_scaler[prop].fit_transform(val_train)
                                training_df.loc[cond_train, ['value']] = test_train
                                
                                # transform test
                                cond_test = test_df[test_df['prop'] == prop].index
                                if len(cond_test) > 0:
                                    val_test = test_df.loc[cond_test, ['value']]
                                    test_test = property_scaler[prop].transform(val_test)  # Use transform, not fit_transform
                                    test_df.loc[cond_test, ['value']] = test_test

                            training_df['test_train'] = 'train'
                            training_df = training_df.reset_index(drop=True)
                            test_df['test_train'] = 'test'
                            test_df = test_df.reset_index(drop=True)

                            df = pd.concat([training_df, test_df], axis = 0)
                            df['true'] = 'yes'

                            
                            df_list = []
                            # Create 30 copies with incremented temperature
                            for i in range(-10,21):
                                temp_df = test_df.copy()
                                temp_df['Temp(C)'] = i * 10
                                df_list.append(temp_df)

                            # Concatenate all DataFrames in the list
                            new_df = pd.concat(df_list, ignore_index=True)
                            new_df['test_train'] = 'unseen_temp'
                            new_df['true'] = 'no'

                            new_df = pd.concat([df, new_df]).reset_index(drop=True)
                            columns_drop = df.columns.drop(['test_train', 'true'])
                            new_df = new_df.drop_duplicates(subset = columns_drop, keep = 'first')
                            new_df = new_df.reset_index(drop=True)

                            print(new_df, new_df.true.unique())

                            # Save the datasets
                            test_df.to_csv(save_file + '/test_df_predict.csv')
                            training_df.to_csv(save_file + '/train_df_predict.csv')
                            new_df.to_csv(save_file + '/training_pg.csv')

                            property_scaler_filename = save_file + '/property_scaler.save'
                            joblib.dump(property_scaler, property_scaler_filename)

                        if split == 'composition':

                            training_df, test_df = composition_split(df_prop, random_seed, test_size, k)
                            print('----training_df---', training_df)
                            print('---df_not_prop---', df_not_prop)
                            training_df = training_df.reset_index(drop=True)
                            training_df = pd.concat([df_not_prop, training_df], axis=0, ignore_index=True)
                            training_df = training_df.reset_index(drop=True)
                            training_df['true'] = 'yes'
                            print('----training_df----', training_df)


                            property_scaler = {}
                            for prop in training_df.prop.unique():
                                property_scaler[prop] = target_property_scaler()

                                # fit transform train
                                cond_train = training_df[training_df['prop'] == prop].index
                                val_train = training_df.loc[cond_train, ['value']]
                                print(val_train)
                                test_train = property_scaler[prop].fit_transform(val_train)
                                training_df.loc[cond_train, ['value']] = test_train
                                
                                # transform test
                                cond_test = test_df[test_df['prop'] == prop].index
                                if len(cond_test) > 0:
                                    val_test = test_df.loc[cond_test, ['value']]
                                    test_test = property_scaler[prop].transform(val_test)  # Use transform, not fit_transform
                                    test_df.loc[cond_test, ['value']] = test_test
                            


                            training_df = training_df.reset_index(drop=True)
                            test_df = test_df.reset_index(drop=True)

                            training_df['id'] = training_df.index
                            test_df['id'] = test_df.index


                            print('----training_df----', training_df)
                            print('----test_df----', test_df)

                            training_pg = pd.concat([training_df, test_df], axis = 0, ignore_index=True)
                            training_pg['id'] = training_pg.index
                            print('----training_pg----', training_pg)

                            training_df.to_csv(save_file + '/training_df.csv')
                            test_df.to_csv(save_file + '/test_df.csv')
                            training_pg.to_csv(save_file + '/training_pg.csv')

                            property_scaler_filename = save_file + '/property_scaler.save'
                            joblib.dump(property_scaler, property_scaler_filename)    

                        if split == 'polymer':
                            training_df, test_df = polymer_split(df_prop, random_seed, test_size)

                            print('----training_df---', training_df)
                            print('---df_not_prop---', df_not_prop)
                            training_df = training_df.reset_index(drop=True)
                            print('----training_df----', training_df)

                            training_df = pd.concat([df_not_prop, training_df], axis=0, ignore_index=True)
                            training_df = training_df.reset_index(drop=True)
                            training_df['true'] = 'yes'

                            property_scaler = {}
                            for prop in training_df.prop.unique():
                                property_scaler[prop] = target_property_scaler()

                                # fit transform train
                                cond_train = training_df[training_df['prop'] == prop].index
                                val_train = training_df.loc[cond_train, ['value']]
                                test_train = property_scaler[prop].fit_transform(val_train)
                                training_df.loc[cond_train, ['value']] = test_train
                                
                                # transform test
                                cond_test = test_df[test_df['prop'] == prop].index
                                if len(cond_test) > 0:
                                    val_test = test_df.loc[cond_test, ['value']]
                                    test_test = property_scaler[prop].transform(val_test)  # Use transform, not fit_transform
                                    test_df.loc[cond_test, ['value']] = test_test
                            


                            training_df = training_df.reset_index(drop=True)
                            test_df = test_df.reset_index(drop=True)

                            training_df['id'] = training_df.index
                            test_df['id'] = test_df.index


                            training_pg = pd.concat([training_df, test_df], axis = 0, ignore_index=True)
                            training_pg['id'] = training_pg.index

                            training_df.to_csv(save_file + '/training_df.csv')
                            test_df.to_csv(save_file + '/test_df.csv')
                            training_pg.to_csv(save_file + '/training_pg.csv')

                            property_scaler_filename = save_file + '/property_scaler.save'
                            joblib.dump(property_scaler, property_scaler_filename)    


split_st(df_file, props_to_consider, splits, random_seeds, num_splits, test_sizes, date)
split_mt(df_file, props_to_consider, splits, random_seeds, num_splits, test_sizes, date)
