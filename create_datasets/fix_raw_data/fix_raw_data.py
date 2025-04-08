import pandas as pd
import numpy as np


def fix_raw(df, props):

    df = df.dropna(how='all')


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
    df['stab_temp'] = df['stab_temp'].fillna('N/A')
    df['RH(%)'] = df['RH(%)'].fillna(100)
    df['lw'] = df['lw'].fillna(1)
    df['Temp(C)'] = df['Temp(C)'].fillna(20)
    df['time(h)'] = df['time(h)'].fillna(0)

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
    df = df[df['prop'].isin(props)]

    # remove rows with imperative columns
    subset_columns = ['c1', 'prop', 'value']
    df = df.dropna(subset=subset_columns, how = 'any')
    
    if 'theor_IEC' in df.columns:
        df = df.drop('theor_IEC', axis=1)

    df['stab_temp'] = df['stab_temp'].replace('N/A', 25)


    return df
