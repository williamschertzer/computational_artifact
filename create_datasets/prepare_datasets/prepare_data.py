import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler
import pickle


def prepare_dataset(df, log):

    # Add theoretical IEC as as feature
    def calculate_iec(smile, c):
        if pd.isna(smile) or pd.isna(c):
            return 0, 0
        mol = Chem.MolFromSmiles(smile.split(".")[0])
        mw = Descriptors.ExactMolWt(mol)
        num_ions = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0)
        return num_ions * c, mw * c

    def calculate_theoretical_iec_for_polymer(row):
        """
        row: a single row (or dictionary-like object) containing
            smiles1, smiles2, smiles3, c1, c2, c3, ...
        Returns: the theoretical IEC in meq/g for that combination
        """
        # Calculate partial IEC for the main monomers
        num_ions_smiles1, mw_smiles1 = calculate_iec(row['smiles1'], row['c1'])
        num_ions_smiles2, mw_smiles2 = calculate_iec(row['smiles2'], row['c2'])
        num_ions_smiles3, mw_smiles3 = calculate_iec(row['smiles3'], row['c3'])
        
        # If you want to include additive contributions, do similarly:
        num_ions_add1, mw_add1 = calculate_iec(row['additive_smiles1'], row['additivec1'])
        num_ions_add2, mw_add2 = calculate_iec(row['additive_smiles2'], row['additivec2'])
        num_ions_add3, mw_add3 = calculate_iec(row['additive_smiles3'], row['additivec3'])
        
        # Sum them up
        total_num_ions = (num_ions_smiles1 + num_ions_smiles2 + num_ions_smiles3 + 
                        num_ions_add1   + num_ions_add2   + num_ions_add3)
        
        total_mw = (mw_smiles1 + mw_smiles2 + mw_smiles3 +
                    mw_add1    + mw_add2    + mw_add3)
        
        # Theoretical IEC in meq g^-1, if we assume:
        #    ( # of charges / molecular weight ) * 1000
        if total_mw > 0:
            iec = (total_num_ions / total_mw) * 1000
        else:
            iec = 0.0  # or numpy.nan if you prefer
        
        iec = round(iec, 2)

        return iec

    # Function to adjust time-dependent data
    def adjust_time_dependent_data(group):
        if isinstance(group, tuple):  # Ensure it's a DataFrame
            print("Unexpected tuple format in group:", group)
            return None  # Or handle it accordingly
        max_idx = group["value"].idxmax()  # Find index of max value
        max_time = group.loc[max_idx, "time(h)"]  # Time corresponding to max value

        # Keep only data from max_time
        group = group.loc[group["time(h)"] >= max_time].copy()
        
        # Reset time so that max value corresponds to time = 0
        group["time(h)"] = group["time(h)"] - max_time

        return group

    def compute_known_values(df, group_columns):
        time_varying_groups = df.groupby(group_columns).filter(lambda g: g["time(h)"].nunique() > 1)
        # Iterate over each group
        for name, group in time_varying_groups.groupby(group_columns, group_keys=False):
            # known_A: value where time(h) == 0
            known_A_value = group.loc[group['time(h)'] == 0, 'value'].values
            if len(known_A_value) > 0:
                df.loc[group.index, 'known_A'] = known_A_value[0]

            # Check if group has at least 5 data points and compute known_B
            if len(group) >= 5:
                sorted_group = group.sort_values('time(h)')
                last_value = sorted_group['value'].iloc[-1]
                second_last_value = sorted_group['value'].iloc[-2]
                value_diff = abs(last_value - second_last_value)

                if value_diff <= 0.01:
                    # known_B: value at the maximum time(h)
                    max_time_value = sorted_group.loc[sorted_group['time(h)'].idxmax(), 'value']
                    df.loc[group.index, 'known_B'] = max_time_value


    df['theor_IEC'] = df.apply(calculate_theoretical_iec_for_polymer, axis=1)

    # fix OhCond
    df['prop'] = df['prop'].replace('OhCond(mS/cm)', 'OHCond(mS/cm)')

    # filter for certain props
    props_to_use = ['OHCond(mS/cm)']
    df = df[df['prop'].isin(props_to_use)]

    subset_columns = ['c1', 'prop', 'value']
    df = df.dropna(subset=subset_columns, how = 'any')
    # df = df.drop('theor_IEC', axis=1)

    # Convert 'value' column to numeric, coercing errors to NaN
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['value'] = round(np.log10(df['value']), 3)
    df = df.reset_index(drop = True)

    # Group time-dependent data by unique identifiers
    group_columns = [
        "smiles1", "smiles2", "smiles3", "c1", "c2", "c3",
        "Temp(C)", "RH(%)", "solvent", "solvent_conc(M)", 
        'additive_smiles1', 'additive_smiles2', 
        'additive_smiles3', 'additivec1', 'additivec2', 'additivec3', 'stab_temp'
    ]

    # Identify time-varying groups
    # time_varying_columns = df.groupby(group_columns).filter(lambda g: g["time(h)"].nunique() > 1)

    time_varying_mask = df.groupby(group_columns)["time(h)"].transform("nunique") > 1
    # Split the dataframe into time-dependent and non-time-dependent parts
    time_df = df[time_varying_mask].copy()
    df = df[~time_varying_mask].copy()

    # Modify each time-dependent group
    time_varying_groups = time_df.groupby(group_columns, group_keys=False)
    modified_time_dfs = []

    for group_key, group_df in time_varying_groups:
        fix_df = adjust_time_dependent_data(group_df)
        modified_time_dfs.append(fix_df)

    if modified_time_dfs:
        modified_time_df = pd.concat(modified_time_dfs, ignore_index=True)
        # Merge the modified time-dependent data back into df
        df = pd.concat([df, modified_time_df], ignore_index=True)

    # Apply log10 to time values across all data
    df['time(h)'] = pd.to_numeric(df['time(h)'], errors='coerce')
    if log:
        df['time(h)'] = df['time(h)'].apply(lambda x: np.log10(x) if x > 0 else x)

    # Compute known_A and known_B for both training and validation datasets
    # Add known_A and known_B columns with default None values
    df['known_A'] = None
    df['known_B'] = None
    compute_known_values(df, group_columns)

    
    df['time(h)'] = df['time(h)'].astype(str)    # Convert the 'time(h)' column back to string
    df['value'] = df['value'].astype(str)    # Convert the 'value' column back to string

    return df