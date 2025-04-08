import os
import json
import shutil
import subprocess

# Define the parent directory containing dataset subdirectories (source datasets)
dataset_parent_dir = "/home/wschertzer3/aem_aging/create_datasets/shared_datasets/st/OH/3_18_25_noscale"  # Modify this path

# Define the current working directory where models will be trained
output_root_dir = os.getcwd()

# Define paths to the scripts
train_script = "train_penn.py"
test_script = "predict.py"

# Define a base settings template
base_settings = {
    "new_model": "",
    "test_preds": False,
    "train_preds": False,
    "val_preds": False,
    "plot_props": [
        "OHCond(mS/cm)"
    ],
    "train_dataset_file": "",
    "test_dataset_file": "",
    "val_dataset_file": "",
    "model_file": "",
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 1000,
    "l1": 512,
    "l2": 512,
    "d1": 0.25,
    "d2": 0.25,
    "weight_decay": 0.001,
    "activation": "leaky_relu"
}

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
    for filename in [f"fp_train_{subdir}.csv", f"train_{subdir}.csv", f"fp_val_{subdir}.csv", f"val_{subdir}.csv", f"fp_test_{subdir}.csv", f"test_{subdir}.csv"]:
        src_file = os.path.join(dataset_subdir_path, filename)
        dest_file = os.path.join(model_subdir, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)

    # Copy scripts to the new model directory
    shutil.copy(train_script, model_subdir)
    shutil.copy(test_script, model_subdir)


    # Modify settings.json for this model
    settings = base_settings.copy()
    settings["new_model"] = f"penn_{subdir}"
    settings["train_dataset_file"] = os.path.join(model_subdir, f"fp_train_{subdir}.csv")
    settings["test_dataset_file"] = os.path.join(model_subdir, f"fp_test_{subdir}.csv")
    settings["val_dataset_file"] = os.path.join(model_subdir, f"fp_val_{subdir}.csv")
    settings["model_file"] = os.path.join(model_subdir, "oh_penn_model.pth")

    # Save modified settings.json in the new model directory
    settings_path = os.path.join(model_subdir, "settings.json")
    with open(settings_path, "w") as json_file:
        json.dump(settings, json_file, indent=4)

    if not os.path.exists(f"{model_subdir}/oh_penn_model.pth"):
        # Run training inside the new model directory
        print(f"Training model for {subdir} in {model_subdir}...")
        subprocess.run(["python", train_script], cwd=model_subdir)

        # Run testing inside the new model directory
        print(f"Testing model for {subdir} in {model_subdir}...")
        subprocess.run(["python", "predict.py"], cwd=model_subdir)
    
    else:
        # model exists
        print("Model Exists!")
        # Run testing inside the new model directory

        print(f"Testing model for {subdir} in {model_subdir}...")
        subprocess.run(["python", "predict.py"], cwd=model_subdir)


print("All models trained and tested successfully.")