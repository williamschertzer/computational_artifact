#!/usr/bin/env python
import pandas as pd
import numpy as np
from pgfingerprinting import fp
from sklearn.preprocessing import MinMaxScaler
import pickle
import argparse
import os


def fingerprint_pg(train_df, val_df, test_df,fp_scaler_path, prop_scaler_path):

    numeric_train_df = train_df.apply(pd.to_numeric, errors='coerce')
    train_df = numeric_train_df.combine_first(train_df)
    train_df = train_df.reset_index(drop=True)

    numeric_val_df = val_df.apply(pd.to_numeric, errors='coerce')
    val_df = numeric_val_df.combine_first(val_df)
    val_df = val_df.reset_index(drop = True)

    numeric_test_df = test_df.apply(pd.to_numeric, errors='coerce')
    test_df = numeric_test_df.combine_first(test_df)
    test_df = test_df.reset_index(drop = True)


    # File to store cached fingerprints
    FINGERPRINT_CACHE_FILE = "fingerprint_cache.pkl"

    # Load existing cache if available
    if os.path.exists(FINGERPRINT_CACHE_FILE):
        with open(FINGERPRINT_CACHE_FILE, "rb") as f:
            fingerprint_cache = pickle.load(f)
    else:
        fingerprint_cache = {}

    def save_fingerprint_cache():
        """Save the computed fingerprints to a file."""
        with open(FINGERPRINT_CACHE_FILE, "wb") as f:
            pickle.dump(fingerprint_cache, f)

    def get_fingerprint(smiles, params):
        """Retrieve fingerprint from cache or compute and store it."""
        if smiles not in fingerprint_cache:
            fingerprint_cache[smiles] = fp.fingerprint_from_smiles(smiles, params)
        return fingerprint_cache[smiles]

    def compute_weighted_fingerprint(row, params):
        """Compute a weighted fingerprint for a given row using cached fingerprints."""
        s1 = row['smiles1'] if pd.notnull(row['smiles1']) else '*CC*'
        s2 = row['smiles2'] if pd.notnull(row['smiles2']) else '*CC*'
        s3 = row['smiles3'] if pd.notnull(row['smiles3']) else '*CC*'
        
        fp1_raw = get_fingerprint(s1, params)
        fp2_raw = get_fingerprint(s2, params)
        fp3_raw = get_fingerprint(s3, params)

        def convert_to_dict(fp_val):
            if isinstance(fp_val, dict):
                return fp_val
            elif isinstance(fp_val, np.ndarray):
                return {i: fp_val[i] for i in range(fp_val.shape[0])}
            else:
                print(row)
                raise ValueError("Fingerprint must be either a dict or a numpy array.")
        
        fp1 = convert_to_dict(fp1_raw)
        fp2 = convert_to_dict(fp2_raw)
        fp3 = convert_to_dict(fp3_raw)

        w1 = row['c1']/100 if pd.notnull(row['c1']) else 0
        w2 = row['c2']/100 if pd.notnull(row['c2']) else 0
        w3 = row['c3']/100 if pd.notnull(row['c3']) else 0

        all_keys = set(fp1.keys()) | set(fp2.keys()) | set(fp3.keys())
        
        return {key: (w1 * fp1.get(key, 0) + w2 * fp2.get(key, 0) + w3 * fp3.get(key, 0)) for key in all_keys}
    
    def process_dataset(df, params):
        """
        Reads the CSV file into a DataFrame and computes a final fingerprint for each row.
        Returns both the original DataFrame and a list of fingerprint dictionaries.
        """
        # (For testing purposes, process only the first X rows.)
        # X = 100
        # df = df.iloc[:X]
        fingerprints = []
        i = 0
        for idx, row in df.iterrows():
            fp_vec = compute_weighted_fingerprint(row, params)
            fingerprints.append(fp_vec)
            if i%500 == 0:
                print(i)
            i+=1
        return df, fingerprints

    def scale_fingerprints(fp_df, fp_scaler_path='fp_scaler.pkl', fit=True, columns_to_scale=None):
        """
        Scales specified columns using MinMaxScaler.
        
        - If `fit=True`, it fits and transforms the specified columns and saves the scaler.
        - If `fit=False`, it loads the saved scaler and transforms the specified columns.

        Parameters:
            fp_df (pd.DataFrame): The DataFrame containing fingerprint features.
            fp_scaler_path (str): Path to save or load the scaler.
            fit (bool): Whether to fit a new scaler (`True`) or use an existing one (`False`).
            columns_to_scale (list or None): List of columns to scale. If None, all columns are scaled.

        Returns:
            pd.DataFrame: DataFrame with scaled specified columns.
            MinMaxScaler: The scaler used for transformation.
        """
        if columns_to_scale is None:
            columns_to_scale = fp_df.columns  # Default to scaling all columns if none are specified

        if fit:
            # Fit a new scaler on the specified columns
            scaler = MinMaxScaler()
            fp_df[columns_to_scale] = scaler.fit_transform(fp_df[columns_to_scale])
            
            # Save the fitted scaler
            with open(fp_scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        else:
            # Load the existing scaler
            with open(fp_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Transform only the specified columns
            fp_df[columns_to_scale] = scaler.transform(fp_df[columns_to_scale])

        return scaler, fp_df.fillna(0)

    def scale_property_by_prop(df, scaler_dict=None, is_train=True, prop_scaler_path='prop_scalers.pkl'):
        """
        Scales the property column 'value' for each unique 'prop' independently.
        A new column 'scaled_value' is added to df.
        """
        if is_train:
            prop_scalers = {}
            scaled_values = pd.Series(index=df.index, dtype=float)
            for prop in df['prop'].unique():
                mask = df['prop'] == prop
                values = df.loc[mask, 'value'].values.reshape(-1, 1)
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(values).flatten()
                scaled_values.loc[mask] = scaled
                prop_scalers[prop] = scaler
            with open(prop_scaler_path, 'wb') as f:
                pickle.dump(prop_scalers, f)
        else:
            prop_scalers = scaler_dict
            scaled_values = pd.Series(index=df.index, dtype=float)
            for prop in df['prop'].unique():
                mask = df['prop'] == prop
                values = df.loc[mask, 'value'].values.reshape(-1, 1)
                if prop in prop_scalers:
                    scaler = prop_scalers[prop]
                    scaled = scaler.transform(values).flatten()
                    scaled_values.loc[mask] = scaled
                else:
                    scaled_values.loc[mask] = df.loc[mask, 'value']
        df['scaled_value'] = scaled_values
        return df, prop_scalers

    def add_additional_features(df, fp_df):

        """
        Appends additional features from the original dataset (df) to the fingerprint DataFrame (fp_df).
        
        Features added:
        - RH(%)     : fill with 100 if missing.
        - Temp (C)  : fill with 20 if missing.
        - Theor IEC : error if missing
        - additives from 'additive_{}' and 'additive_name_2'.
        - solvents from 'solvent_{}'.
        - stab_temp : fill with 0 if missing.
        - One-hot encoded prop.
        - time(h)   : from 'time(h)' or 'time'.
        - known_A, known_B
        """

        # RH(%)
        if 'RH(%)' in df.columns:
            rh = df['RH(%)'].fillna(100)
        else:
            print("NO RH")
            quit()
            
        # Temp (C)
        if 'Temp(C)' in df.columns:
            temp = df['Temp(C)'].fillna(20)
        else:
            print('NO TEMP')
            quit()

        if 'theor_IEC' in df.columns:
            theor_IEC = df['theor_IEC']
        else:
            print('NO THEOR IEC')
            quit()
            
        # Use additive_{name} and solvent_{name} columns directly
        additive_cols = [col for col in df.columns if col.startswith('additive_')]
        solvent_cols = [col for col in df.columns if col.startswith('solvent_')]

        additive_data = df[additive_cols] if additive_cols else pd.DataFrame(index=df.index)
        solvent_data = df[solvent_cols] if solvent_cols else pd.DataFrame(index=df.index)

    
        # stab_temp:
        if 'stab_temp' in df.columns:
            stab_temp = df['stab_temp'].fillna(25)
        else:
            print('NO STAB TEMP')
            quit()
        
        # One-hot encoding for prop:
        if 'prop' in df.columns:
            prop_dummies = pd.get_dummies(df['prop'], prefix='prop')
            # Ensure binary structure (0 or 1) for solvent
            prop_dummies = (prop_dummies > 0).astype(int)
        else:
            print('NO PROP')
            quit()
            
        # time(h)
        if 'time(h)' in df.columns:
            time_h = df['time(h)']
        else:
            print("NO TIME")
            quit()
        # # known_A
        # if 'known_A' in df.columns:
        #     known_A = df['known_A']
        # else:
        #     print("NO KNOWN_A")
        #     quit()
        # # known_B
        # if 'known_B' in df.columns:
        #     known_B = df['known_B']
        # else:
        #     print("NO KNOWN B")
        #     quit()
        
        

        # print(rh, temp, additive_dummies, solvent_dummies)
        # quit()


        # Combine all additional features.
        additional_features = pd.concat([rh, temp, theor_IEC, additive_data, solvent_data, 
                                        stab_temp, prop_dummies, time_h], axis=1)
        # join with the fingerprint DataFrame.
        combined_df = fp_df.join(additional_features)


        return combined_df

    def main(train_df,val_df,test_df,fp_scaler_path, prop_scaler_path):

        # Parameters for fingerprinting
        params = {
        "fp_identifier": "fp_",
        "write_property": 0,
        "col_property": "",
        "normalize_a": 1,
        "normalize_b": 1,
        "normalize_m": 1,
        "normalize_e": 1,
        "block_list_version": "20201210",
        "ismolecule": 0,
        "polymer_fp_type": ["aS", "aT", "bT", "m", "e"],
        "calculate_side_chain": 1,
        "use_chirality": 1,
        }

        # --- Process the training set ---
        print("Processing training set...")

        train_df, train_fps = process_dataset(train_df, params)
        train_fp_df = pd.DataFrame(train_fps)
        train_fp_df = add_additional_features(train_df, train_fp_df)

        
        # Save the training feature columns for later use.
        training_feature_columns = train_fp_df.columns

        columns_to_scale = ['RH(%)', 'Temp(C)', 'theor_IEC', 'stab_temp', 'time(h)']
        columns_to_scale += [col for col in train_fp_df.columns if col.startswith('additive_') or col.startswith('solvent_')]
        scaler, scaled_train_df = scale_fingerprints(train_fp_df, 'fp_scaler.pkl', fit=True, columns_to_scale=columns_to_scale)

        

        # reset index and add id and value columns
        final_train_df = scaled_train_df.reset_index(drop = True)
        final_train_df.insert(0, 'id', final_train_df.index)
        final_train_df['value'] = train_df['value']

        if val_df is not None:
            # --- Process the val set ---
            print("Processing val set...")
            val_df, val_fps = process_dataset(val_df, params)
            val_fp_df = pd.DataFrame(val_fps)
            val_fp_df = add_additional_features(val_df, val_fp_df)

            # Reindex val features to match the training feature columns.
            val_fp_df = val_fp_df.reindex(columns=training_feature_columns, fill_value=0)
            val_fp_df = val_fp_df.fillna(0)
            print(val_fp_df)
            
            _, scaled_val_df = scale_fingerprints(val_fp_df, 'fp_scaler.pkl', fit=False, columns_to_scale=columns_to_scale)


            print(scaled_val_df)


            # reset index and add id and value columns
            scaled_val_df = scaled_val_df.reset_index(drop = True)
            scaled_val_df.insert(0, 'id', scaled_val_df.index)
            scaled_val_df['value'] = val_df['value']

            print(scaled_val_df)
            


        if test_df is not None:
            # --- Process the test set ---
            print("Processing test set...")
            test_df, test_fps = process_dataset(test_df, params)
            test_fp_df = pd.DataFrame(test_fps)
            test_fp_df = add_additional_features(test_df, test_fp_df)

            
            # Reindex test features to match the training feature columns.
            test_fp_df = test_fp_df.reindex(columns=training_feature_columns, fill_value=0)
            test_fp_df = test_fp_df.fillna(0)
            print(test_fp_df)

            _, scaled_test_df = scale_fingerprints(test_fp_df, 'fp_scaler.pkl', fit=False, columns_to_scale=columns_to_scale)
     

            # reset index and add id and value columns
            scaled_test_df = scaled_test_df.reset_index(drop = True)
            scaled_test_df.insert(0, 'id', scaled_test_df.index)
            scaled_test_df['value'] = test_df['value']

            print(scaled_test_df)


        return final_train_df, scaled_val_df, scaled_test_df
    
    # Save fingerprint cache to reuse in future runs
    save_fingerprint_cache()

    return main(train_df, val_df, test_df,fp_scaler_path, prop_scaler_path)