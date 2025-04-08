import os
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Function for formatting ticks as 10^x
def log_format(x, pos):
    return f"$10^{{{x:.2f}}}$"


# Set input directory containing subdirectories
input_dir = "/data/wschertzer/aem_aging/create_datasets/shared_datasets/st/OH/3_18_25_noscale"
output_base_dir = os.getcwd()  # Current working directory

# Iterate over numeric subdirectories
for subdir in sorted(os.listdir(input_dir), key=int):
    if subdir.isdigit():  # Ensure the directory name is a number
        subdir_path = os.path.join(input_dir, subdir)
        i = subdir  # Use the directory name as the index

        train_path = os.path.join(subdir_path, f"fp_train_{i}.csv")
        val_path = os.path.join(subdir_path, f"fp_val_{i}.csv")
        test_path = os.path.join(subdir_path, f"fp_test_{i}.csv")
        test_time_path = os.path.join(subdir_path, f"test_{i}.csv")  # Test set for extracting time(h) and value


        if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):


            # Read train and val datasets
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)

            # Combine train and val
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            combined_df = combined_df.reset_index(drop=True)
            combined_df['id'] = combined_df.index
            combined_df = combined_df.drop("Unnamed: 0", axis=1)

            # Create new subdirectory in current working directory
            new_subdir_path = os.path.join(output_base_dir, subdir)
            os.makedirs(new_subdir_path, exist_ok=True)

            model_path = os.path.join(new_subdir_path, "model.pkl")

            if not os.path.exists(model_path):

                # Save combined dataset
                combined_csv_path = os.path.join(new_subdir_path, "combined.csv")
                combined_df.to_csv(combined_csv_path, index=False)

                # Create input_ml.txt file
                input_ml_path = os.path.join(new_subdir_path, "input_ml.txt")
                input_ml_content = f"""\
                file_fp_prop        = /data/wschertzer/aem_aging/modeling/workflows/3_18_25_GPR/{subdir}/combined.csv
                col_id              = id
                col_X               = afp_C4_C4_C4 mfp_MQNs24 mfp_MQNs25 bfp_23 mfp_MQNs31 afp_O1 afp_C4 afp_N4 mfp_Chi1n afp_C3_S4_C3 mfp_MQNs29 efp_fam_single afp_O2 afp_H1_C4_N4 afp_C3_C3_C3 bfp_355 mfp_MQNs26 bfp_114 bfp_145 bfp_526 afp_C3_C3_H1 bfp_239 efp_norm_mol_wt bfp_294 efp_main_chain_ring bfp_295 bfp_349 efp_ring_dist_shortest afp_O1_S4_O1 bfp_464 mfp_MQNs13 bfp_314 mfp_Chi0n mfp_Chi0v bfp_289 afp_S4 mfp_Chi2n afp_C3_C3_S4 afp_H1 bfp_197 efp_main_chain_rel mfp_MQNs36 afp_C4_N4_C4 afp_C3_C3_C4 mfp_MQNs27 mfp_Chi1v mfp_MQNs17 bfp_334 afp_C4_C4_H1 afp_H1_C4_H1 bfp_267 mfp_MQNs20 bfp_482 mfp_MQNs19 afp_C3_S4_O1 bfp_343 mfp_NumAromaticRings bfp_222 bfp_359 bfp_354 efp_side_chain_abs mfp_Chi2v afp_C3_C4_H1 afp_C3 mfp_MQNs14 mfp_MQNs16 mfp_MQNs21 efp_ring efp_numatoms_none_H afp_C3_O2_C3 bfp_304 mfp_tpsa afp_C3_C3_O2 afp_C3_C4_N4 afp_C3_N2_N2 bfp_283 bfp_350 afp_H1_C3_N3 bfp_358 bfp_282 afp_C3_C4_C3 afp_C3_C3_N3 afp_H1_C4_S2 afp_C4_C4_N3 efp_side_chain_large_abs afp_C3_C4_S2 mfp_MQNs32 afp_S2 afp_C3_C3_N2 afp_C4_C3_N2 afp_C3_C3_S2 bfp_315 mfp_MQNs35 mfp_NumAliphaticRings efp_3v_side afp_N3 afp_C3_N3_N2 mfp_MQNs30 bfp_347 afp_N2_N2_N3 afp_N2 bfp_305 afp_C3_S2_C4 afp_C4_N3_N2 afp_H1_C4_N3 afp_C3_N3_C4 afp_C4_C3_H1 efp_3v_main bfp_340 bfp_360 efp_fam_ketone afp_C3_N3_C3 bfp_101 afp_N3_C3_N3 bfp_257 mfp_MQNs28 bfp_85 afp_C3_C4_N3 afp_C3_C3_O1 bfp_209 bfp_458 afp_C4_C4_N4 efp_multi_ring_dimer bfp_290 afp_H1_C3_H1 bfp_316 afp_C3_C4_C4 mfp_MQNs42 bfp_456 bfp_455 bfp_261 mfp_MQNs41 bfp_244 bfp_342 bfp_454 bfp_481 bfp_483 afp_C4_C4_O2 afp_C4_C3_O2 bfp_426 afp_C3_O2_C4 afp_C4_C3_O1 efp_4v_main bfp_428 efp_fam_acrylate afp_O1_C3_O2 bfp_218 bfp_430 bfp_424 bfp_249 bfp_361 afp_H1_C4_O2 afp_C4_O2_C4 afp_C3_N4_C4 efp_4v_side afp_C3_C3_N4 bfp_248 afp_C3_C4_O2 bfp_344 bfp_277 afp_C2_C3_C3 bfp_328 afp_C3_C2_N1 afp_C4_N3_C4 afp_F1 bfp_356 afp_C4_C4_F1 bfp_384 bfp_299 mfp_MQNs15 afp_C2 afp_F1_C4_F1 afp_N1 bfp_379 bfp_58 bfp_112 bfp_3 bfp_113 afp_C2_C4_H1 afp_C2_C4_C4 afp_C4_C2_N1 afp_C4_N3_H1 bfp_312 bfp_313 bfp_461 bfp_508 bfp_525 bfp_497 bfp_329 afp_C3_C3_F1 bfp_181 afp_O2_C4_O2 afp_H1_N3_H1 bfp_311 afp_C4_O2_H1 bfp_266 bfp_353 bfp_288 afp_N3_C4_N3 bfp_337 afp_Br1 afp_Br1_C3_C3 bfp_82 afp_C3_O2_O2 bfp_473 bfp_146 afp_C3_C4_F1 bfp_469 bfp_335 afp_Br1_C4_C3 bfp_448 bfp_486 bfp_471 afp_C4_C3_N3 afp_Br1_C4_H1 bfp_131 bfp_133 bfp_271 bfp_89 bfp_386 bfp_208 bfp_338 afp_H1_C3_S4 bfp_241 bfp_480 bfp_243 efp_fam_carbonateester afp_O1_C3_O1 mfp_MQNs37 bfp_100 bfp_475 bfp_11 bfp_260 bfp_387 bfp_460 bfp_235 afp_N2_C3_N2 afp_C3_N2_C3 bfp_238 bfp_453 afp_C4_N4_H1 afp_C4_O2_Si4 mfp_MQNs33 afp_H1_C4_Si4 afp_C4_Si4_C4 bfp_348 afp_C4_C4_Si4 afp_C4_Si4_O2 afp_Si4 afp_C3_O2_H1 bfp_494 bfp_242 RH(%) Temp(C) theor_IEC additive_*CC* additive_1-vinyl-propyltriethoxysilane-imidazolium additive_Im-phenyl-Silsesquioxane additive_N/A additive_PAGE additive_PE additive_PEG additive_Polyvinyl Imidazolium additive_Polyvinylbenzyl chloride additive_QA-Silsesquioxane additive_ZrO2 additive_diethanediol additive_glutaraldehyde additive_halloysite_nanotube additive_hexanedithiol additive_octanedithiol additive_phenyl-Silsesquioxane solvent_KOH solvent_N/A solvent_NaOH solvent_conc(M) stab_temp prop_OHCond(mS/cm) time(h)
                col_Y               = value
                file_model          = model.pkl
                file_parity         = parity.jpg
                file_trainset       = trainset.csv
                file_testset        = testset.csv
                file_parameters     = parameters.csv
                ML_algo             = GPR
                ML_kernel           = 2
                ML_fp_scale         = 1
                ML_n_fold_CV        = 5
                ML_trainset_shuffle = 0
                ML_train_size       = 1
                """
                with open(input_ml_path, "w") as f:
                    f.write(input_ml_content)

                # Run ML training
                subprocess.run(["ml", "input_ml.txt"], cwd=new_subdir_path)

                # Create input_predict.txt file
                input_predict_path = os.path.join(new_subdir_path, "input_predict.txt")
                input_predict_content = f"""\
                file_model = model.pkl
                file_fingerprint = {subdir_path}/fp_test_{i}.csv
                file_output = predicted_Y.csv
                """
                with open(input_predict_path, "w") as f:
                    f.write(input_predict_content)

                # Run prediction
                subprocess.run(["predict", "input_predict.txt"], cwd=new_subdir_path)

            # Load predicted values
            predicted_path = os.path.join(new_subdir_path, "predicted_Y.csv")
            if os.path.exists(predicted_path):
                predicted_df = pd.read_csv(predicted_path)

                # Merge predicted values with test dataset
                test_time_df = pd.read_csv(test_time_path)[['time(h)', 'value']].reset_index(drop=True)

                print(test_time_df)

                merged_df = pd.concat([test_time_df, predicted_df], axis=1)

                merged_df.rename(columns={'value': 'true_value', merged_df.columns[2]: 'predicted_value'}, inplace=True)

                # Save merged dataset
                merged_csv_path = os.path.join(new_subdir_path, "merged_predictions.csv")
                merged_df.to_csv(merged_csv_path, index=False)

                # === Create Figure & Axis ===
                fig, ax1 = plt.subplots(figsize=(8, 6))

                # Scatter plot for true values
                ax1.scatter(
                    merged_df["time(h)"], 
                    merged_df["true_value"], 
                    label="True OHCond(mS/cm)", 
                    color='red', 
                    alpha=0.7, 
                    edgecolors='black', 
                    s=100
                )

                # Scatter plot for predicted values
                ax1.scatter(
                    merged_df["time(h)"], 
                    merged_df["predicted_value"], 
                    label="Predicted OHCond(mS/cm)", 
                    color='blue', 
                    alpha=0.7, 
                    edgecolors='black', 
                    s=100
                )

                # Set title with bold text and increased font size
                ax1.set_title("GPR Time vs True & Predicted OH⁻ Conductivity", fontsize=18, fontweight='bold')

                # Set labels with bold text and larger font size
                ax1.set_xlabel("Time (h)", fontsize=16, fontweight='bold')
                ax1.set_ylabel("Predicted OH⁻ Conductivity (mS/cm)", fontsize=16, fontweight='bold')

                # Set tick label font size
                ax1.tick_params(axis='both', labelsize=14)

                # Apply logarithmic formatting for x and y ticks
                ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax1.xaxis.set_major_formatter(ticker.FuncFormatter(log_format))
                ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax1.yaxis.set_major_formatter(ticker.FuncFormatter(log_format))

                # Add legend
                ax1.legend(fontsize=14, frameon=True)

                # Save and show plot
                plot_path = os.path.join(new_subdir_path, "prediction_plot.png")
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                print(f"Plot saved as {plot_path}")
                plt.show()

print("Processing completed!")
