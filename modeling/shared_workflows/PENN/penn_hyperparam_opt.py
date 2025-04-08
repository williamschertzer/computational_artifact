import os
import json
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import subprocess

# Define paths to your existing datasets
settings_path = "settings.json"

with open(settings_path, 'r') as file:
    base_settings = json.load(file)

# Load all data into a single DataFrame
train_df = pd.read_csv(base_settings["train_dataset_file"])
val_df = pd.read_csv(base_settings["val_dataset_file"])
test_df = pd.read_csv(base_settings["test_dataset_file"])

combined_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

# Perform an 80/10/10 split
train_data, temp_data = train_test_split(combined_df, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save these datasets for use in training
train_data.to_csv("train_optuna.csv", index=False)
val_data.to_csv("val_optuna.csv", index=False)
test_data.to_csv("test_optuna.csv", index=False)

# Define Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 500, 1000)
    l1 = trial.suggest_int('l1', 16, 512, step=16)
    l2 = trial.suggest_int('l2', 16, 512, step=16)
    d1 = trial.suggest_float('d1', 1e-3, 0.5)
    d2 = trial.suggest_float('d2', 1e-3, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    activation = trial.suggest_categorical('activation', ["relu", "leaky_relu", "elu", "tanh"])

    # Create subdirectory for this trial
    trial_dir = f"trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    # Copy train_penn.py to the trial directory
    shutil.copy("train_penn.py", trial_dir)

    # Update settings with suggested hyperparameters
    trial_settings = base_settings.copy()
    trial_settings.update({
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "l1": l1,
        "l2": l2,
        "d1": d1,
        "d2": d2,
        "weight_decay": weight_decay,
        "activation": activation,
        "train_dataset_file": os.path.abspath("train_optuna.csv"),
        "val_dataset_file": os.path.abspath("val_optuna.csv"),
        "test_dataset_file": os.path.abspath("test_optuna.csv"),
        "new_model": f"optuna_model_trial_{trial.number}",
        "model_file": os.path.join(trial_dir, f"optuna_model_trial_{trial.number}.pth")
    })

    # Write updated settings to file
    trial_settings_path = os.path.join(trial_dir, f"settings.json")
    with open(trial_settings_path, "w") as file:
        json.dump(trial_settings, file, indent=4)

    # Run training
    result = subprocess.run(["python", "train_penn.py", "--settings", trial_settings_path], cwd=trial_dir, capture_output=True, text=True)

    print(result)

    if result.returncode != 0:
        print(f"Training failed for trial {trial.number} with return code {result.returncode}.")
        print(result.stderr)
        raise Exception("Training script failed!")

    # Extract the validation loss from the standard output
    for line in result.stdout.splitlines():
        if "Validation Loss:" in line:
            val_loss = float(line.split(":")[-1].strip())
            print(f"Trial {trial.number} - Validation Loss: {val_loss}")
            return val_loss

    # If we reach here, it means the validation loss was not found
    raise RuntimeError("Validation loss not found in training script output.")


# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)

# Save best hyperparameters
with open("best_hyperparameters.json", "w") as file:
    json.dump(study.best_params, file, indent=4)

print("Hyperparameter optimization completed.")
