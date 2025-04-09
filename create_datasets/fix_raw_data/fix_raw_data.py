import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler
import re


def fix_raw(df, props, log_time):

    def calculate_iec(smile, c):
        if pd.isna(smile) or pd.isna(c) or not isinstance(smile, str):
            return 0, 0
        mol = Chem.MolFromSmiles(smile)
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

    def assign_additive_content(row):
        # Count real additive names
        name1 = row['additive_name_1']
        name2 = row['additive_name_2']

        real_names = [n for n in [name1, name2] if n != 'N/A']
        real_smiles = [
            s for s in [row['additive_smiles1'], row['additive_smiles2'], row['additive_smiles3']]
            if s != '*CC*'
        ]

        additive_contributions = {f"additive_{a}": 0.0 for a in unique_additives}

        if len(real_names) == 1:
            # One additive present → sum all additive concentrations
            additive_contributions[f"additive_{real_names[0]}"] = row['additivec1'] + row['additivec2'] + row['additivec3']

        elif len(real_names) == 2 and len(real_smiles) == 2:
            # Two names, two valid SMILES → use c1 and c2
            additive_contributions[f"additive_{real_names[0]}"] = row['additivec1']
            additive_contributions[f"additive_{real_names[1]}"] = row['additivec2']

        elif len(real_names) == 2 and len(real_smiles) == 3:
            # Ambiguous case → record and skip assignment
            ambiguous_rows.append(row)
            
            return pd.Series(additive_contributions)

        return pd.Series(additive_contributions)


    df = df.dropna(how='all')


    # fill missing values with defaults
    df['Backbone'] = df['Backbone'].fillna('N/A') 
    df['Cation'] = df['Cation'].fillna('N/A') 
    df['Sample'] = df['Sample'].fillna('N/A') 
    df['smiles1'] = df['smiles1'].fillna('*CC*')
    df['smiles2'] = df['smiles2'].fillna('*CC*')
    df['smiles3'] = df['smiles3'].fillna('*CC*')
    df['c1'] = df['c1'].fillna(0)
    df['c2'] = df['c2'].fillna(0)
    df['c3'] = df['c3'].fillna(0)
    df['additive_name_1'] = df['additive_name_1'].fillna('N/A')
    df['additive_name_2'] = df['additive_name_2'].fillna('N/A')
    df['additive_smiles1'] = df['additive_smiles1'].fillna('*CC*')
    df['additive_smiles2'] = df['additive_smiles2'].fillna('*CC*')
    df['additive_smiles3'] = df['additive_smiles3'].fillna('*CC*')
    df['additivec1'] = df['additivec1'].fillna(0)
    df['additivec2'] = df['additivec2'].fillna(0)
    df['additivec3'] = df['additivec3'].fillna(0)
    df['solvent'] = df['solvent'].fillna('N/A')
    df['solvent_conc(M)'] = df['solvent_conc(M)'].fillna(0)
    df['stab_temp'] = df['stab_temp'].fillna(25)
    df['RH(%)'] = df['RH(%)'].fillna(100)
    df['lw'] = df['lw'].fillna(1)
    df['Temp(C)'] = df['Temp(C)'].fillna(25)
    df['time(h)'] = df['time(h)'].fillna(0)


    # Define columns to clean
    smiles_cols = [
        'smiles1', 'smiles2', 'smiles3',
        'additive_smiles1', 'additive_smiles2', 'additive_smiles3'
    ]

    # Apply regex to remove all ".x" fragments (dot followed by any alphanumeric or bracketed group)
    for col in smiles_cols:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r"\.\[.*?\]|\.\w+", "", x))

    subset_columns = ['smiles1', 'smiles2', 'smiles3', 'c1', 'c2', 'c3', 
                    'additive_smiles1', 'additive_smiles2', 'additive_smiles3', 
                    'additivec1', 'additivec2', 'additivec3', 'solvent', 'solvent_conc(M)', 
                    'stab_temp', 'RH(%)', 'Temp(C)', 'RH(%)', 'time(h)', 'prop', 'value']

    # Create a mask for duplicates (including the first occurrence)
    duplicate_mask = df.duplicated(subset=subset_columns, keep=False)

    # Create a mask for rows where EXP_IEC is NaN
    no_exp_iec_mask = df['EXP_IEC'].isna()

    # Remove rows that are duplicates AND have no value for EXP_IEC
    df = df[~(duplicate_mask & no_exp_iec_mask)]

    # fix OhCond
    df['prop'] = df['prop'].replace('OhCond(mS/cm)', 'OHCond(mS/cm)')


    # filter for certain props
    props_to_use = props
    df = df[df['prop'].isin(props_to_use)]

    # remove rows that do not contain values for imperative columns
    subset_columns = ['c1', 'prop', 'value']
    df = df.dropna(subset=subset_columns, how = 'any')

    # add theoretical IEC column
    if 'theor_IEC' in df.columns:
        df = df.drop('theor_IEC', axis=1)
    df['theor_IEC'] = df.apply(calculate_theoretical_iec_for_polymer, axis=1)

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

    # Apply log10 to time values across all data if log_time is true
    df['time(h)'] = pd.to_numeric(df['time(h)'], errors='coerce')
    if log_time:
        df['time(h)'] = df['time(h)'].apply(lambda x: np.log10(x) if x > 0 else x)


    df['time(h)'] = df['time(h)'].astype(str)    # Convert the 'time(h)' column back to string
    df['value'] = df['value'].astype(str)    # Convert the 'value' column back to string



    # Add a column for each unique solvent
    unique_solvents = [s for s in df['solvent'].unique() if s != 'N/A']
    for solvent in unique_solvents:
        df[f"solvent_{solvent}"] = df.apply(
            lambda row: row['solvent_conc(M)'] if row['solvent'] == solvent else 0,
            axis=1
        )

    # Special case handling: combine glutaraldehyde and halloysite_nanotube
    condition = (
        (df['additive_name_1'] == 'glutaraldehyde') & 
        (df['additive_name_2'] == 'halloysite_nanotube')
    )

    df.loc[condition, 'additive_smiles2'] = (
        df.loc[condition, 'additive_smiles2'] + ',' + df.loc[condition, 'additive_smiles3']
    )
    df.loc[condition, 'additivec2'] = (
        df.loc[condition, 'additivec1'] + df.loc[condition, 'additivec2']
    )
    df.loc[condition, 'additivec1'] = 0
    df.loc[condition, 'additive_smiles3'] = '*CC*'

    # Add a column for each unique additive name
    unique_additives = [s for s in pd.unique(df[['additive_name_1', 'additive_name_2']].values.ravel('K')) if s != 'N/A']
    for additive in unique_additives:
        df[f"additive_{additive}"] = 0.0  # Initialize all to zero

    # List to track problematic rows
    ambiguous_rows = []


    # Apply and update df
    df.update(df.apply(assign_additive_content, axis=1))

    # Print problematic rows at the end
    if ambiguous_rows:
        print("\n Ambiguous rows with 2 additive names and 3 real SMILES:")
        print(pd.DataFrame(ambiguous_rows)[[
            'additive_name_1', 'additive_name_2', 
            'additive_smiles1', 'additive_smiles2', 'additive_smiles3', 
            'additivec1', 'additivec2', 'additivec3'
        ]])




    # drop unnecessary columns
    df = df.drop(['Backbone', 'Cation', 'Sample', 'additive_name_1', 'additive_name_2', 'additive_smiles1', 
                  'additive_smiles2', 'additive_smiles3', 'additivec1', 'additivec2', 'additivec3',
                  'solvent', 'solvent_conc(M)', 'lw', 'EXP_IEC', 'DOI', 'additive_*CC*'], axis = 1)

    print(df, df.columns)



    return df
