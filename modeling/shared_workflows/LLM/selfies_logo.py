import os
import json
import shutil
import subprocess

# Define the parent directory containing dataset subdirectories (source datasets)
dataset_parent_dir = "/home/wschertzer3/aem_aging/create_datasets/shared_datasets/st/OH/3_18_25_noscale"  # Modify this path

# Define the current working directory where models will be trained
output_root_dir = os.getcwd()

# Define paths to the scripts
train_script = "llama_aem.py"
test_script = "predict.py"

# Define a base settings template
base_settings = {
    "new_model": "",
    "test_preds": False,
    "train_preds": False,
    "val_preds": False,
    "plot_props": ["OHCond(mS/cm)"],
    "train_dataset_file": "",
    "test_dataset_file": "",
    "val_dataset_file": "",
    "scaler_file": "/home/wschertzer3/aem_aging/create_datasets/prepare_datasets/property_scalers.pkl",
    "unscale": False,
    "adapter_path": "",
    "regression_head_config_path": "",
    "regression_head_path": ""
}


def find_model_files(model_root):
    """Recursively searches for adapter_config.json and regression_head.pth in model_root."""
    adapter_path, regression_head_path, regression_head_config_path = None, None, None
    
    for root, _, files in os.walk(model_root):
        for file in files:
            if file == "adapter_config.json":
                adapter_path = os.path.abspath(root)
            elif file == "regression_head.pth":
                regression_head_path = os.path.join(root, file)
            elif file == "regression_head_config.json":
                regression_head_config_path = os.path.join(root, file)
        
        # If all files are found, stop searching further
        if adapter_path and regression_head_path and regression_head_config_path:
            break

    return adapter_path, regression_head_path, regression_head_config_path



# Iterate over dataset subdirectories
for subdir in sorted(os.listdir(dataset_parent_dir), key=int):
    dataset_subdir_path = os.path.join(dataset_parent_dir, subdir)

    if not os.path.isdir(dataset_subdir_path):
        continue  # Skip files, process directories only

    print(f"Processing dataset subdirectory: {subdir}")

    # Create a corresponding subdirectory in the current working directory
    model_subdir = os.path.join(output_root_dir, subdir)
    os.makedirs(model_subdir, exist_ok=True)

    # Copy dataset files into the new model directory
    for filename in [f"train_{subdir}.csv", f"val_{subdir}.csv", f"test_{subdir}.csv"]:
        src_file = os.path.join(dataset_subdir_path, filename)
        dest_file = os.path.join(model_subdir, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)

    # Copy scripts to the new model directory
    shutil.copy(train_script, model_subdir)
    shutil.copy(test_script, model_subdir)

    # Define adapter and regression head paths
    adapter_path, regression_head_path, regression_head_config_path = find_model_files(model_subdir)

    # Modify settings.json for this model
    settings = base_settings.copy()
    settings["new_model"] = f"aem_aging_{subdir}"
    settings["train_dataset_file"] = os.path.join(model_subdir, f"train_{subdir}.csv")
    settings["test_dataset_file"] = os.path.join(model_subdir, f"test_{subdir}.csv")
    settings["val_dataset_file"] = os.path.join(model_subdir, f"val_{subdir}.csv")
    settings["adapter_path"] = adapter_path
    settings["regression_head_path"] = regression_head_path
    settings["regression_head_config_path"] = regression_head_config_path

    # Save modified settings.json in the new model directory
    settings_path = os.path.join(model_subdir, "settings.json")
    with open(settings_path, "w") as json_file:
        json.dump(settings, json_file, indent=4)

    # Run training inside the new model directory
    print(f"Training model for {subdir} in {model_subdir}...")
    subprocess.run(["python", train_script], cwd=model_subdir)

    # redefine adapter and regression head path after creation
    adapter_path, regression_head_path, regression_head_config_path = find_model_files(model_subdir)


    # Ensure they are properly set in settings.json
    settings["adapter_path"] = adapter_path
    settings["regression_head_path"] = regression_head_path
    settings["regression_head_config_path"] = regression_head_config_path

    # Save updated settings.json
    settings_path = os.path.join(model_subdir, "settings.json")
    with open(settings_path, "w") as json_file:
        json.dump(settings, json_file, indent=4)

    # Modify predict.py for testing
    predict_file = os.path.join(model_subdir, "predict.py")
    with open(predict_file, "r") as file:
        predict_content = file.read()

    # Ensure adapter_path and test dataset file are correctly set
    predict_content = predict_content.replace('adapter_path = ".*"', f'adapter_path = "{adapter_path}"')
    predict_content = predict_content.replace('regression_head_path = ".*"', f'regression_head_path = "{regression_head_path}"')
    predict_content = predict_content.replace('regression_head_config_path = ".*"', f'regression_head_config_path = "{regression_head_config_path}"')
    predict_content = predict_content.replace('test_dataset_file = ".*"', f'test_dataset_file = "{settings["test_dataset_file"]}"')

    # Save modified predict.py in the new model directory
    with open(predict_file, "w") as file:
        file.write(predict_content)

    # Run testing inside the new model directory
    print(f"Testing model for {subdir} in {model_subdir}...")
    subprocess.run(["python", "predict.py"], cwd=model_subdir)

print("All models trained and tested successfully.")
