import torch
from torch import nn
import pandas as pd
import numpy as np
import json
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from train_penn import OH_PENN, MLP_PENN, prepare_data
import os
import glob



def log_format(x, pos):
    return f"$10^{{{x:.2f}}}$"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load settings.json
# Path to the JSON file
json_file_path = "settings.json"

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)


# create config
config = {
    "l1": data["l1"],
    "l2": data["l2"],
    "d1": data["d1"],
    "d2": data["d2"],
    "activation": data['activation']
}

# Load test dataset
df_test = pd.read_csv(data['test_dataset_file']).drop("Unnamed: 0", axis=1).drop('id', axis = 1)

# replace tim(h) with inscaled version

# find data dir
test_dir = os.path.dirname(data['test_dataset_file'])
test_i_file = glob.glob(os.path.join(test_dir, 'test_*.csv'))[0]
# Load matched train_i/val_i datasets
test_i_df = pd.read_csv(test_i_file)
# Replace 'time(h)' column
df_test['time(h)'] = test_i_df['time(h)']


# Extract features (modify feature column names based on your dataset)
feature_columns = [col for col in df_test.columns if col not in ["time(h)", "value"]]

# Convert to PyTorch tensors
fp_test_tensor, t_test_tensor, sigma = prepare_data(df_test, feature_columns)

# Load the trained model
model = OH_PENN(n_fp=fp_test_tensor.shape[1], device=device, config=config).to(device)
model.load_state_dict(torch.load("oh_penn_model.pth", map_location=device))


model.eval()  # Set model to evaluation mode

# Lists to store predictions and constants
predictions = []
constants_list = []

# Iterate through each row of the test set
for index, row in df_test.iterrows():
    # Convert row to PyTorch tensors
    fp_sample = torch.tensor(row[feature_columns].values, dtype=torch.float32).to(device).unsqueeze(0)  # (1, n_fp)
    t_sample = torch.tensor(row["time(h)"], dtype=torch.float32).to(device).view(1, 1)  # (1, 1)

    # Run inference to get sigma and constants
    with torch.no_grad():
        sigma_pred = model(fp_sample, t_sample, train=False).item()  # conductivity prediction
        constants = model(fp_sample, t_sample, get_constants=True)   # retrieve constants

    # Store predictions and constants
    predictions.append(sigma_pred)
    constants_list.append({
        "A": constants["A"].item(),
        "B": constants["B"].item(),
        "t0": constants["t0"].item(),
        "alpha": constants["alpha"].item()
    })

# Add predictions and constants to DataFrame
merged_df = pd.DataFrame(constants_list)
merged_df['predicted_value'] = predictions
merged_df['time(h)'] = df_test['time(h)']
merged_df['value'] = df_test['value']



# Save predictions to CSV
output_path = "merged_predictions.csv"
merged_df.to_csv(output_path, index=False)

fig, ax1 = plt.subplots(figsize=(8, 6))

# Scatter plot for true values
ax1.scatter(
    merged_df["time(h)"], 
    merged_df["value"], 
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
ax1.set_title("PeNN Time vs True & Predicted OH⁻ Conductivity", fontsize=18, fontweight='bold')

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
plot_path = os.path.join("prediction_plot.png")
plt.savefig(plot_path, bbox_inches='tight', dpi=300)
print(f"Plot saved as {plot_path}")
print(f"Predictions saved to: {output_path}")