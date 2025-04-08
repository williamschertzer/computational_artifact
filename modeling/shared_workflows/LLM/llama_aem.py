import math
import re
import os
import torch
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from datasets import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
import pickle
from transformers.modeling_outputs import ModelOutput
from torch import nn
import itertools
import optuna
import selfies as sf
from functools import partial
from sklearn.model_selection import train_test_split



class functionholder:

    def load_lines(jsonl_file : str) -> list[dict]:
        """ Load a JSONL file into list of dictionaries. """
        with open(jsonl_file) as fp:
            jsonlines = list(fp)
        return [ json.loads(json_str) for json_str in jsonlines ]

    def gen(lines):
        yield from lines

    def predict_prop(model, tokenizer, prompt, time, unscale=False):
        """
        Use the model + regression head to get numeric predictions (A, B, t0, alpha) 
        for each prompt/time pair. 
        """
        # Make sure we return numeric results in a list
        results = []
            
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # # Print tokenized input
        # print("Input tokens:", inputs["input_ids"].shape)  # Should match expected length
        # print("Input token IDs:", inputs["input_ids"])
        # print("Decoded prompt:", tokenizer.decode(inputs["input_ids"][0]))  # Check if the full prompt is retained
        
        # Forward pass with no grad
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        

        # Extract the last hidden state and the attention mask
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        attention_mask = inputs["attention_mask"]  # (batch, seq_len)
        
        # Pass through custom regression head
        regression_out = model.regression_head(last_hidden, attention_mask)  # (batch, 4)
        
        # For a single example, regression_out[0] is [A, B, t0, alpha]
        A, B, t0, alpha = regression_out[0]
        
        # Convert each to a Python float
        sigmoid = nn.Sigmoid()
        A   = sigmoid(A) * 2.5
        B   = sigmoid(B) * 2.5
        t0  = sigmoid(t0) * 3
        alpha = sigmoid(alpha) * 10
        
        # compute the conductivity from A,B,t0, alpha at the given time
        
        time_input = time

        # negative softmax form
        exponent = -(time_input - t0) * alpha       
        exp_term = math.exp(exponent)
        pred_conductivity = A - ((A - B) / (1 + exp_term))

        # store parameters and final cond:
        results.append({"A": A, "B": B, "t0": t0, "alpha": alpha, "time":time_input,
                        "pred_conductivity": pred_conductivity})
        
        print(results)
        
        return results

    def calc_r2(df, prop, colx, coly):
        # Filter the DataFrame by the specified property
        df = df[df['prop'] == prop]

        # Only proceed if we have data
        if len(df['value']) != 0:
            # Calculate R^2 and 
            r_squared = r2_score(df[colx], df[coly])
        else:
            r_squared = -10
        return r_squared

    def plot_scatter_with_regression(df, prop, colx, coly, data_dir, plots_dir, filename, linestyle='--'):
        # Set style
        sns.set_style("whitegrid")

        df.to_csv(f"{data_dir}/{filename}.csv")

        # Filter the DataFrame by the specified property
        df = df[df['prop'] == prop]

        # Only proceed if we have data
        if len(df['value']) != 0:
            # Create a JointGrid with regression
            g = sns.jointplot(
                x=colx, y=coly, data=df, kind="reg", height=7,
                joint_kws={
                    'scatter_kws': {'color': 'blue', 'alpha': 0.5, 's': 100},
                    'line_kws': {'color': 'red', 'linewidth': 2, 'linestyle': linestyle},
                    'ci': None
                },
                marginal_kws=dict(bins=50)
            )

            # Grab the main scatter axis
            ax = g.ax_joint

            # Make the plot square by setting the aspect ratio to 'equal'
            # We will also unify the x and y limits so that they match.
            min_val = min(df[colx].min(), df[coly].min())
            max_val = max(df[colx].max(), df[coly].max())

            # Expand the range slightly for padding
            range_padding = 0.05 * (max_val - min_val)
            ax.set_xlim(min_val - range_padding, max_val + range_padding)
            ax.set_ylim(min_val - range_padding, max_val + range_padding)
            ax.set_aspect('equal', 'box')

            # Plot parity line (y = x)
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                linestyle='--',
                color='black',
                label='y = x'
            )

            # Calculate Pearson correlation coefficient
            r_value = df[colx].corr(df[coly])
            # Calculate R^2 and RMSE
            r_squared = r2_score(df[colx], df[coly])
            rmse = np.sqrt(mean_squared_error(df[colx], df[coly]))
            print(f"r = {np.round(r_value, 3)}")
            print(f"R^2 = {np.round(r_squared, 3)}")
            print(f"RMSE = {np.round(rmse, 3)}")

            # Set labels
            x_axis_label = f"{colx}: all props"
            ax.set_xlabel(x_axis_label, fontsize=20)
            ax.set_ylabel(coly, fontsize=20)

            # Set labels to "Experimental {prop}" and "Predicted {prop}"
            ax.set_xlabel(f"Experimental {prop}", fontsize=20)
            ax.set_ylabel(f"Predicted {prop}", fontsize=20)

            # Adjust tick parameters
            ax.tick_params(axis='both', direction='in', width=1.5, labelsize=14)

            # Annotate statistics on the plot
            ax.annotate(f'r = {r_value:.2f}', xy=(0.1, 0.9), xycoords='axes fraction', ha='left', fontsize=16)
            ax.annotate(f'R^2 = {r_squared:.2f}', xy=(0.1, 0.8), xycoords='axes fraction', ha='left', fontsize=16)
            ax.annotate(f'RMSE = {rmse:.2f}', xy=(0.1, 0.7), xycoords='axes fraction', ha='left', fontsize=16)

            # Save the plot
            out_filename = f"{plots_dir}/{filename}.png"
            plt.savefig(out_filename, bbox_inches='tight')

            # Show plot
            plt.show()

    def preprocess_polymer_smiles(smiles):
        """
        Replace wildcard attachment points [*] with Hydrogen (H) 
        to ensure valid SELFIES conversion.
        """
        if '[*]' not in smiles:
            smiles = smiles.replace('*', '[*]')

        return smiles.replace("[*]", "[H]")

    def instruction_format_pred(row, dataset_field):
        property_name_str = str(row['prop'])
        match = re.match(r"([a-zA-Z\s]+)(\([^\)]+\))", property_name_str)
        if match:
            prop_clean = match.group(1).strip()  # e.g., "Swelling"
            unit = match.group(2).strip()        # e.g., "(%)"
        else:
            prop_clean = property_name_str
            unit = ""

        system_prompt = (
            "<SYSTEM>: You are an AI assistant specializing in material property modeling.\n"
        )
        
        question = f"What are the A, B, t0, and alpha for the {prop_clean} of the following AEM: "

        if row['theor_IEC'] is not None:
            question += f"IEC: {row['theor_IEC']}, "
        if row['smiles1'] is not None:
            question += f"comonomer 1: {sf.encoder(functionholder.preprocess_polymer_smiles(row['smiles1']))}, "
        else:
            print("------SMILES1 is None:------", row)
            quit()
        if row['smiles2'] is not None:
            question += f"comonomer 2: {sf.encoder(functionholder.preprocess_polymer_smiles(row['smiles2']))}, "
        if row['smiles3'] is not None:
            question += f"comonomer 3: {sf.encoder(functionholder.preprocess_polymer_smiles(row['smiles3']))}, "
        if row['c1'] is not None:
            question += f"c1: {row['c1']}, "
        else:
            print("------Composition is None:------", row)
            quit()
        if row['c2'] is not None:
            question += f"c2: {row['c2']}, " 
        if row['c3'] is not None:
            question += f"c3: {row['c3']}, "
        if row['additive_name_1'] is not None:
            question += f"Additive 1: {row['additive_name_1']}, " 
        if row['additive_name_2'] is not None:
            question += f"Additive 2: {row['additive_name_2']}, " 
        if row['additivec1'] is not None:
            question += f"Additive c1: {row['additivec1']}, " 
        if row['additivec2'] is not None:
            question += f"Additive c2: {row['additivec2']}, " 
        if row['additivec3'] is not None:
            question += f"Additive c3: {row['additivec3']}, " 
        if row['solvent'] is not None:
            question += f"Solvent: {row['solvent']}, "
        if row['solvent_conc(M)'] is not None:
            question += f"Solvent conc: {row['solvent_conc(M)']} M, "
        if row['stab_temp'] is not None:
            question += f"Stability test temperature: {row['stab_temp']}, "
        if row['RH(%)'] is not None:
            question += f"Relative Humidity: {row['RH(%)']} %, "
        if row['Temp(C)'] is not None:
            question += f"Measurement temperature: {row['Temp(C)']} C\n"
        
        # Assistant response placeholder (JSON output expected)
        assistant_response = "<ASSISTANT>: "

        # Combine everything into a single formatted row
        row[dataset_field] = f"{system_prompt}{question}{assistant_response}"

        return row

    def instruction_format_train(row, dataset_field):
        property_name_str = str(row['prop'])
        match = re.match(r"([a-zA-Z\s]+)(\([^\)]+\))", property_name_str)
        if match:
            prop_clean = match.group(1).strip()  # e.g., "Swelling"
            unit = match.group(2).strip()        # e.g., "(%)"
        else:
            prop_clean = property_name_str
            unit = ""


        # System role instruction
        system_prompt = (
            "<SYSTEM>: You are an AI assistant specializing in material property modeling.\n"
        )

        # row[dataset_field] = f"<s>
        
        question = f"What are the A, B, t0, and alpha for the {prop_clean} of the following AEM: "

        if row['theor_IEC'] is not None:
            question += f"IEC: {row['theor_IEC']}, "
        if row['smiles1'] is not None:
            question += f"comonomer 1: {sf.encoder(functionholder.preprocess_polymer_smiles(row['smiles1']))}, "
        else:
            print("------SMILES1 is None:------", row)
            quit()
        if row['smiles2'] is not None:
            question += f"comonomer 2: {sf.encoder(functionholder.preprocess_polymer_smiles(row['smiles2']))}, "
        if row['smiles3'] is not None:
            question += f"comonomer 3: {sf.encoder(functionholder.preprocess_polymer_smiles(row['smiles3']))}, "
        if row['c1'] is not None:
            question += f"c1: {row['c1']}, "
        else:
            print("------Composition is None:------", row)
            quit()
        if row['c2'] is not None:
            question += f"c2: {row['c2']}, " 
        if row['c3'] is not None:
            question += f"c3: {row['c3']}, "
        if row['additive_name_1'] is not None:
            question += f"Additive 1: {row['additive_name_1']}, " 
        if row['additive_name_2'] is not None:
            question += f"Additive 2: {row['additive_name_2']}, " 
        if row['additivec1'] is not None:
            question += f"Additive c1: {row['additivec1']}, " 
        if row['additivec2'] is not None:
            question += f"Additive c2: {row['additivec2']}, " 
        if row['additivec3'] is not None:
            question += f"Additive c3: {row['additivec3']}, " 
        if row['solvent'] is not None:
            question += f"Solvent: {row['solvent']}, "
        if row['solvent_conc(M)'] is not None:
            question += f"Solvent conc: {row['solvent_conc(M)']} M, "
        if row['stab_temp'] is not None:
            question += f"Stability test temperature: {row['stab_temp']}, "
        if row['RH(%)'] is not None:
            question += f"Relative Humidity: {row['RH(%)']} %, "
        if row['Temp(C)'] is not None:
            question += f"Measurement temperature: {row['Temp(C)']} C\n"
        
        # JSON-based assistant response
        json_output = {"predicted_value": float(row["value"])}
        assistant_response = f"<ASSISTANT>: {json.dumps(json_output)}"

        # Combine everything into a single formatted row
        row[dataset_field] = f"{system_prompt}{question}{assistant_response}"

        return row

    def add_regression_targets(row):
        # Convert the measurement time and target conductivity to float
        row["time"] = float(row["time(h)"]) if row["time(h)"] is not None else 0.0
        row["target_conductivity"] = float(row["value"])
        return row

    def get_preds(row, dataset_field, model, tokenizer, unscale = False):
        # Use measurement time from dataset
        time = row['time(h)'] if row['time(h)'] is not None else 0  # Default to 0 hours if missing

        response = functionholder.predict_prop(model, tokenizer, row['prompt_pred'], time, unscale=unscale)

        row[dataset_field] = response  # Store computed OH conductivity

        return row

    def custom_data_collator(features, tokenizer):

        keys_to_keep = ["input_ids", "attention_mask", "time", "target_conductivity"]

        # Filter out extra keys from each feature.
        filtered_features = [{k: f[k] for k in keys_to_keep if k in f} for f in features]
    
        # Collate the standard fields using the default collator.
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        batch = data_collator(filtered_features)
        
        # Manually add extra fields if they are not present.
        if "time" not in batch:
            try:
                batch["time"] = torch.tensor([f["time"] for f in features], dtype=torch.float)
            except KeyError:
                print("Warning: 'time' not found in the features.")
        if "target_conductivity" not in batch:
            try:
                batch["target_conductivity"] = torch.tensor(
                    [f["target_conductivity"] for f in features], dtype=torch.float
                )
            except KeyError:
                print("Warning: 'target_conductivity' not found in the features.")
        
        return batch

    def extract_params(pred_prop):
        pred = pred_prop[0]
        print('---pred---', pred)
        
        return pred

    def extract_pred_conductivity(x):
        return x['pred_conductivity']

    def log_format(x, pos):
        return f"$10^{{{x:.2f}}}$"

class MLPRegressionHead(nn.Module):
    def __init__(self, num_layers, input_sizes, output_sizes, dropout_rates, activation=nn.ReLU):
        """
        Args:
            input_sizes (list of int): Input size for each layer.
            output_sizes (list of int): Output size for each layer.
            dropout_rates (list of float): Dropout rate for each layer.
            activation (nn.Module): Activation function to use (default: ReLU).
        """
        super().__init__()

        # Validate inputs
        assert len(output_sizes) == num_layers, "Length of output_sizes must match number of layers"
        assert len(dropout_rates) == num_layers, "Length of dropout_rates must match number of layers"

        layers = []

        # Build the MLP layers dynamically
        for i in range(num_layers):
            layers.append(nn.Linear(input_sizes[i], output_sizes[i]))  # Linear layer
            layers.append(activation())  # Activation function
            layers.append(nn.Dropout(dropout_rates[i]))  # Dropout layer

        # Output layer: The last `output_size` should be 4 (for A, B, t0, alpha)
        layers.append(nn.Linear(output_sizes[-1], 4))

        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        sum_hidden = torch.sum(hidden_states * mask, dim=1)  # sum over seq_len
        lengths = torch.sum(mask, dim=1)  # (batch, 1)
        pooled = sum_hidden / lengths  # (batch, hidden_size)

        raw_output = self.mlp(pooled)

        return raw_output

class CustomSFTTrainer(SFTTrainer):

    def __init__(self, *args, num_layers, input_sizes, output_sizes, dropout_rates, activation, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes
        self.dropout_rates = dropout_rates
        self.activation = activation

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, precomputed_outputs=None):
        # ----- get outputs -----
        labels = inputs.get("input_ids")
        outputs = model(**inputs)
        
        # ----- Regression Loss -----
        # Get last layer hidden states from the transformer (shape: batch x seq_len x hidden_size)
        hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs.get("attention_mask")
        # Pass through the regression head to obtain 4 numbers per instance
        regression_out = model.regression_head(hidden_states, attention_mask)  # (batch, 4)

        sigmoid = nn.Sigmoid()
        
        A = sigmoid(regression_out[:, 0]) * 2.5
        B = sigmoid(regression_out[:, 1]) * 2.5
        t0 = sigmoid(regression_out[:, 2]) * 3
        alpha = sigmoid(regression_out[:, 3]) * 10
        
        # Get measurement time; assume 'time' is provided as a tensor (shape: batch,)
        time = inputs.get("time")
        
        # Compute predicted conductivity using the differentiable formula
        exponent = -(time - t0) * alpha       
        exp_term = torch.exp(exponent)
        pred_conductivity = A - ((A - B) / (1 + exp_term))
        target_conductivity = inputs.get("target_conductivity")
        
        # Compute MSE loss for the regression task
        mse_loss = nn.MSELoss()(pred_conductivity, target_conductivity)
        total_loss = mse_loss

        
        # # Backward pass to compute gradients
        # mse_loss.backward()

        # # Print gradients
        # print("\n=== Checking LLM Gradients ===")
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f"{name}: Gradient Norm = {torch.norm(param.grad).item()}")


        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # get outputs and hidden states
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs.get("attention_mask")
        # Pass through the regression head to obtain 4 numbers per instance
        regression_out = model.regression_head(hidden_states, attention_mask)  # (batch, 4)
        
        print('----test----', regression_out[:, 0])
        # quit()

        sigmoid = nn.Sigmoid()

        A = sigmoid(regression_out[:, 0]) * 2.5
        B = sigmoid(regression_out[:, 1]) * 2.5
        t0 = sigmoid(regression_out[:, 2]) * 3
        alpha = sigmoid(regression_out[:, 3]) * 10

        print('a, b, t0, alpha: ', A, B, t0, alpha)
        
        # Get measurement time; assume 'time' is provided as a tensor (shape: batch,)
        time = inputs.get("time")
        
        # Compute predicted conductivity using the differentiable formula
        exponent = -(time - t0) * alpha       
        exp_term = torch.exp(exponent)
        pred_conductivity = A - ((A - B) / (1 + exp_term))
        
        # Get the target conductivity; assume provided as a tensor (shape: batch,)
        target_conductivity = inputs.get("target_conductivity")
        
        # Compute MSE loss for the regression task
        mse_loss = nn.MSELoss()(pred_conductivity, target_conductivity)
        eval_loss = mse_loss.cpu().detach()
        return (eval_loss, None, None)
    
    def save_model(self, output_dir: str, _internal_call: bool = False):
        """
        Save the model and regression head to the checkpoint directory.
        """
        super().save_model(output_dir, _internal_call)  # Save LoRA adapter & base model

        # Save the regression head
        reg_head_path = os.path.join(output_dir, "regression_head.pth")
        torch.save(self.model.regression_head.state_dict(), reg_head_path)

        # Save regression head configuration
        reg_head_config = {
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.num_layers,
            "input_sizes": self.input_sizes,
            "output_size": self.output_sizes,
            "dropout_rates": self.dropout_rates,
            "activation": self.activation
        }
        reg_head_config_path = os.path.join(output_dir, "regression_head_config.json")
        with open(reg_head_config_path, "w") as f:
            json.dump(reg_head_config, f)

        print(f"Regression head saved to {reg_head_path}")
        print(f"Regression head config saved to {reg_head_config_path}")

def train(learning_rate, train_epoch, lora_r, lora_alpha, lora_dropout, batch_size, num_layers, input_sizes, output_sizes, dropout_rates, activation, grad_accum, checkpoint):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    ## Check the environment
    assert torch.cuda.is_available(), "Failed to detect GPUs, make sure you set up cuda correctly!"
    print("Number of GPUs available: ", torch.cuda.device_count())

    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
    os.environ["HF_HOME"] = "/data/William"
    print("Huggingface Home =", os.environ['HF_HOME'])

    # The model to load from a directory or the model name from the Hugging Face hub.
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    # Activate 4-bit precision for model loading
    use_4bit = False

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = True

    # Load the entire model on the GPU 0
    # device_map = {"": 0}
    device_map = "auto"

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load model
    model = AutoModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        token='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'
    )

    # Ensure hidden states are returned (this may be set globally or per forward pass)
    model.config.output_hidden_states = True

    # Attach the regression head and ensure gradients are being calculated
    activation_fn = getattr(nn, activation)  # Convert string to activation function
    model.regression_head = MLPRegressionHead(num_layers, input_sizes, output_sizes, dropout_rates, activation_fn)
    model.regression_head.to(model.device)
    print("Regression head architecture: ")
    print(model.regression_head)
    for param in model.regression_head.parameters():
        param.requires_grad = True  # This ensures that gradients will be computed

    # Print to verify
    for name, param in model.regression_head.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = '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')
    tokenizer.pad_token = tokenizer.eos_token

    # settings
    # Path to the JSON file
    json_file_path = "settings.json"

    # Read the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Assign the variables
    new_model = (
    f"lr_{learning_rate}/"
    f"epochs_{train_epoch}/"
    f"r_{lora_r}/"
    f"a_{lora_alpha}/"
    f"loradrop_{lora_dropout}/"
    f"batch_{batch_size}/"
    f"layers_{num_layers}/"
    f"dim_{input_sizes}/"
    f"regdrop_{dropout_rates}/"
    f"act_{activation}"
    )
    test_preds = data["test_preds"]
    train_preds = data["train_preds"]
    val_preds = data["val_preds"]
    plot_props = data["plot_props"]
    train_dataset_file = data["train_dataset_file"]
    test_dataset_file = data["test_dataset_file"]
    val_dataset_file = data["val_dataset_file"]
    parent_directory = os.getcwd()
    scaler_file = data["scaler_file"]
    unscale = data["unscale"]
    with open(scaler_file, 'rb') as f:
        scalers = pickle.load(f)
    pattern = r'ASST:\s*(-?[0-9]+\.[0-9]+)'


    train_data = pd.read_csv(train_dataset_file)
    test_data = pd.read_csv(test_dataset_file)
    val_data = pd.read_csv(val_dataset_file)


    # Print the sizes of each split
    # print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")


    # Output directory where the model predictions and checkpoints will be stored
    os.makedirs(os.path.join(parent_directory, "results"), exist_ok = True)
    output_dir = os.path.join(parent_directory, f"results/{new_model}/checkpoints")
    plots_dir = os.path.join(parent_directory, f"results/{new_model}/plots")
    data_dir = os.path.join(parent_directory, f"results/{new_model}/data")
    os.makedirs(plots_dir, exist_ok = True)
    os.makedirs(data_dir, exist_ok = True)

    train_data.to_json(f'{data_dir}/train_data.jsonl', orient='records', lines=True)
    val_data.to_json(f'{data_dir}/val_data.jsonl', orient='records', lines=True)
    test_data.to_json(f'{data_dir}/test_data.jsonl', orient='records', lines=True)

    # generate train dataset
    print('loading in training dataset')
    lines = functionholder.load_lines(f"{data_dir}/train_data.jsonl")
    train_dataset = Dataset.from_generator(functionholder.gen, gen_kwargs={'lines': lines})

    # generate validation dataset
    print('loading in validation dataset')
    lines = functionholder.load_lines(f"{data_dir}/val_data.jsonl")
    val_dataset = Dataset.from_generator(functionholder.gen, gen_kwargs={'lines': lines})

    # generate test dataset
    print('loading in test dataset')
    lines = functionholder.load_lines(f"{data_dir}/test_data.jsonl")
    test_dataset = Dataset.from_generator(functionholder.gen, gen_kwargs={'lines': lines})


    # map train dataset
    dataset_field = 'prompt'
    print('mapping train dataset')
    train_dataset = train_dataset.map(lambda row: functionholder.instruction_format_train(row, dataset_field))
    train_dataset = train_dataset.map(functionholder.add_regression_targets)
    print('mapping validation dataset')
    val_dataset = val_dataset.map(lambda row: functionholder.instruction_format_train(row, dataset_field))
    val_dataset = val_dataset.map(functionholder.add_regression_targets)

    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = lora_r

    # Alpha parameter for LoRA scaling
    lora_alpha = lora_alpha

    # Dropout probability for LoRA layers
    lora_dropout = lora_dropout

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Number of training epochs
    num_train_epochs = train_epoch

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False

    # Batch size per GPU for training
    per_device_train_batch_size = batch_size

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = batch_size

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = grad_accum

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = learning_rate

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule (constant a bit better than cosine)
    lr_scheduler_type = "constant"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # calc number of data points in train set
    num_data_points = len(train_data)

    # calc steps based on batch size and number of data points
    steps = math.ceil(num_data_points / per_device_train_batch_size)
    print('num data points, batch size, steps', num_data_points, per_device_eval_batch_size, steps)

    # Save checkpoint every X updates steps
    save_steps = steps

    # eval steps
    eval_steps = steps

    # Log every X updates steps
    logging_steps = math.floor(steps / 4)

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False
    dataset_text_field='prompt'

    tokenizer.padding_side = "right" # Fix overflow issue with fp16 training

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)


    # Set supervised fine-tuning parameters
    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        data_collator=partial(functionholder.custom_data_collator, tokenizer=tokenizer),
        num_layers=num_layers,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        dropout_rates=dropout_rates,
        activation=activation,
        # callbacks=[early_stopping],
        args = SFTConfig(
            dataset_text_field=dataset_field,
            max_seq_length=max_seq_length,
            packing=packing,
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            eval_strategy="steps",
            eval_steps=eval_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_strategy='steps',
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            gradient_checkpointing=gradient_checkpointing,
            # target_modules = test
            report_to="none"))

    # train or load the model
    if not os.path.exists(f"{parent_directory}/{new_model}/"):

        # Train model

        # Check if checkpoint directory exists and has files
        checkpoint_dir = os.path.join(output_dir,f'checkpoint-{checkpoint}')

        if os.path.isdir(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
            print(f"Resuming from checkpoint: {checkpoint_dir}")
            # load base model checkpoint
            model = AutoModel.from_pretrained(checkpoint_dir, device_map="auto")
            # load regression head checkpoint
            reg_head_config_path = os.path.join(checkpoint_dir, "regression_head_config.json")
            if os.path.exists(reg_head_config_path):
                with open(reg_head_config_path, "r") as f:
                    reg_head_config = json.load(f)
                
                # Dynamically create the regression head
                activation_fn = getattr(nn, reg_head_config["activation"])  # Convert string to activation function
                
                model.regression_head = MLPRegressionHead(
                    num_layers=reg_head_config["num_layers"],
                    input_sizes=reg_head_config["input_sizes"],
                    output_sizes=reg_head_config["output_size"],
                    dropout_rates=reg_head_config["dropout_rates"],
                    activation=activation_fn
                ).to(model.device)


                # Load saved regression head weights
                reg_head_path = os.path.join(checkpoint_dir, "regression_head.pth")
                if os.path.exists(reg_head_path):
                    model.regression_head.load_state_dict(torch.load(reg_head_path))
                    model.regression_head.to(model.device)
                    print("Regression head loaded successfully.")
                    trainer.train()
                else:
                    print("Warning: Regression head weights not found!")
                    quit()

            else:
                print("Warning: Regression head config not found!")
                quit()
        
        else:
            print("No checkpoint found. Training from scratch.")
            for param in model.regression_head.parameters():
                param.requires_grad = True
            # for name, param in model.regression_head.named_parameters():
            #     print(f"{name}: requires_grad = {param.requires_grad}")
            # quit()
            trainer.train()

        # Save trained model
        trainer.model.save_pretrained(new_model)
        # Save the regression head weights
        reg_head_path = os.path.join(new_model, "regression_head.pth")
        torch.save(model.regression_head.state_dict(), reg_head_path)

        # Save the regression head configuration
        reg_head_config = {
            "hidden_size": model.config.hidden_size,
            "num_layers": num_layers,
            "input_sizes": input_sizes,
            "output_size": output_sizes,
            "dropout_rates": dropout_rates,
            "activation": activation
        }

        reg_head_config_path = os.path.join(new_model, "regression_head_config.json")
        with open(reg_head_config_path, "w") as f:
            json.dump(reg_head_config, f)


        model.load_adapter(f"./{new_model}/")
        model.regression_head.load_state_dict(torch.load(f"{new_model}/regression_head.pth"))
        model.regression_head.to(model.device)
    else:
        print(f'model exists, loading in: {new_model}')
        model.load_adapter(f"./{new_model}/")
        # Load regression head config
        reg_head_config_path = os.path.join(new_model, "regression_head_config.json")
        if os.path.exists(reg_head_config_path):
            with open(reg_head_config_path, "r") as f:
                reg_head_config = json.load(f)

            print(reg_head_config)

            # Dynamically initialize the regression head
            activation_fn = getattr(nn, reg_head_config["activation"])  # Convert string to activation function
            model.regression_head = MLPRegressionHead(
                    num_layers=reg_head_config["num_layers"],
                    input_sizes=reg_head_config["input_sizes"],
                    output_sizes=reg_head_config["output_size"],
                    dropout_rates=reg_head_config["dropout_rates"],
                    activation=activation_fn
                ).to(model.device)

            # Load regression head weights
            reg_head_path = os.path.join(new_model, "regression_head.pth")
            if os.path.exists(reg_head_path):
                model.regression_head.load_state_dict(torch.load(reg_head_path, map_location=model.device))
                model.regression_head.to(model.device)
                print("Regression head loaded successfully.")
            else:
                print("Warning: Final regression head weights not found!")
                quit()

        else:
            print("Warning: Final regression head config not found!")
            quit()


    # evaluate the model
    # Compute validation loss after training
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]

    print(f"Validation Loss: {eval_loss}")
    

    # Ensure model is in evaluation mode
    model.eval()

    if test_preds:
        # Make prediction for the Test set
        # test set prompt for prediction
        dataset_field = "prompt_pred"
        test_dataset = test_dataset.map(lambda row: functionholder.instruction_format_pred(row, dataset_field))
        print('making predictions on test data')
        dataset_field = "pred_prop"
        test_dataset = test_dataset.map(lambda row: functionholder.get_preds(row, dataset_field, model, tokenizer))
        test_dataset_df = test_dataset.to_pandas()

        print("Test Set: ")
        print("Size of the dataset: ", test_dataset_df.shape[0])
        
        # Extract the float number and create a new column
        # Convert 'pred_prop_float' to numeric, forcing non-numeric values to NaN
        test_dataset_df['pred_prop_float'] = test_dataset_df['pred_prop'].apply(lambda row_list: row_list[0].get("pred_conductivity", float('nan')) if len(row_list) > 0 else float('nan'))

        test_dataset_df.to_csv('test_dataset_with_pred.csv')
        print('this is pred prop float before dropping', test_dataset_df['pred_prop_float'])

        # Drop rows where 'pred_prop_float' is NaN (which includes non-numeric values)
        test_dataset_df = test_dataset_df.dropna(subset=['pred_prop_float'])
        
        print('this is pred prop float after dropping', test_dataset_df['pred_prop_float'])

        # Parity plot for the Test set
        test_dataset_df['value'] = test_dataset_df['value'].astype('float64')
        for prop in plot_props:
            print(f"plotting {prop}, test set predictions")
            functionholder.plot_scatter_with_regression(test_dataset_df, prop, "value", "pred_prop_float", data_dir, plots_dir, "test", linestyle='--')

    if val_preds:
        # Val set prompt for prediction
        dataset_field = "prompt_pred"
        val_dataset = val_dataset.map(lambda row: functionholder.instruction_format_pred(row, dataset_field))

        print('making predictions on val data')
        dataset_field = "pred_prop"
        val_dataset = val_dataset.map(lambda row: functionholder.get_preds(row, dataset_field, model, tokenizer))
        val_dataset_df = val_dataset.to_pandas()

        print("val Set: ")
        print("Size of the dataset: ", val_dataset_df.shape[0])

        
        # Extract the float number and create a new column
        # Convert 'pred_prop_float' to numeric, forcing non-numeric values to NaN
        val_dataset_df['pred_prop_float'] = val_dataset_df['pred_prop'].apply(lambda row_list: row_list[0].get("pred_conductivity", float('nan')) if len(row_list) > 0 else float('nan'))


        val_dataset_df.to_csv('val_dataset_with_pred.csv')
        print('this is pred prop float before dropping', val_dataset_df['pred_prop_float'])

        # Drop rows where 'pred_prop_float' is NaN (which includes non-numeric values)
        val_dataset_df = val_dataset_df.dropna(subset=['pred_prop_float'])
        
        print('this is pred prop float after dropping', val_dataset_df['pred_prop_float'])

        # Parity plot for the Val set
        val_dataset_df['value'] = val_dataset_df['value'].astype('float64')
        for prop in plot_props:
            functionholder.plot_scatter_with_regression(val_dataset_df, prop, "value", "pred_prop_float", data_dir, plots_dir, "val", linestyle='--')
        
        # GET TEST SET OH COND R2 FOR HYPERPARAM OPTMIZATION
        prop = 'OHCond(mS/cm)'
        val_r2 = functionholder.calc_r2(val_dataset_df, prop,"value", "pred_prop_float")

    if train_preds:
        # Train set prompt for prediction
        dataset_field = "prompt_pred"
        train_dataset = train_dataset.map(lambda row: functionholder.instruction_format_pred(row, dataset_field))

        print('making predictions on train data')
        dataset_field = "pred_prop"
        train_dataset = train_dataset.map(lambda row: functionholder.get_preds(row, dataset_field, model, tokenizer))
        train_dataset_df = train_dataset.to_pandas()

        print("train Set: ")
        print("Size of the dataset: ", train_dataset_df.shape[0])
        
        # Extract the float number and create a new column
        # Convert 'pred_prop_float' to numeric, forcing non-numeric values to NaN
        train_dataset_df['pred_prop_float'] = train_dataset_df['pred_prop'].apply(lambda row_list: row_list[0].get("pred_conductivity", float('nan')) if len(row_list) > 0 else float('nan'))


        train_dataset_df.to_csv('train_dataset_with_pred.csv')
        print('this is pred prop float before dropping', train_dataset_df['pred_prop_float'])

        # Drop rows where 'pred_prop_float' is NaN (which includes non-numeric values)
        train_dataset_df = train_dataset_df.dropna(subset=['pred_prop_float'])
        
        print('this is pred prop float after dropping', train_dataset_df['pred_prop_float'])

        # Parity plot for the Train set
        train_dataset_df['value'] = train_dataset_df['value'].astype('float64')
        for prop in plot_props:
            functionholder.plot_scatter_with_regression(train_dataset_df, prop, "value", "pred_prop_float", data_dir, plots_dir, "train", linestyle='--')
    
    return eval_loss

def objective(trial):
    """
    Define the objective function for Bayesian Optimization.
    """
    # Define the search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 5e-3)
    lora_r = trial.suggest_categorical("lora_r", [8, 16])
    lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32])
    lora_dropout = trial.suggest_uniform("lora_dropout", 0.0, 0.1)
    batch_size = trial.suggest_categorical("batch_size", [16,16])
    train_epoch = trial.suggest_int("train_epoch", 10, 10)
    num_layers = trial.suggest_int("num_layer", 2, 6)
    # Suggest layer sizes and dropout rates dynamically
    allowed_dims = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]
    hidden_dim = trial.suggest_categorical("hidden_dim", allowed_dims)
    input_sizes = [2048] + [hidden_dim] * (num_layers - 1)
    output_sizes = [hidden_dim] * (num_layers)
    # # Suggest sizes for each layer
    # for i in range(num_layers):

    #     out_size = trial.suggest_categorical(f"layer_{i}_output_size", allowed_dims)
    #     output_sizes.append(out_size)

    #     # Next layer's input size = this layer's output size
    #     if i < num_layers - 1:
    #         input_sizes.append(out_size)




    # dropout_rates = [trial.suggest_uniform(f"layer_{i}_dropout", 0.05, 0.2) for i in range(num_layers)]
    dropout_value = trial.suggest_uniform("dropout_value", 0.05, 0.2)
    dropout_rates = [dropout_value]*num_layers
    activation = trial.suggest_categorical("activation", ['ReLU', 'LeakyReLU'])
    
    # Call the train function

    print('---inputs---', input_sizes)
    print('---outputs---', output_sizes)

    result = train(
        learning_rate, train_epoch, lora_r, lora_alpha, lora_dropout,
        batch_size, num_layers, input_sizes, output_sizes, dropout_rates, activation, grad_accum = 4, checkpoint = '0'
    )
    
    # Return the validation loss as the objective to minimize
    return result


# Define paths to your existing datasets
temp_settings_path = "settings_temp.json"

with open(temp_settings_path, 'r') as file:
    base_settings = json.load(file)

# Load all data into a single DataFrame
train_df = pd.read_csv(base_settings["train_dataset_file"])
val_df = pd.read_csv(base_settings["val_dataset_file"])
test_df = pd.read_csv(base_settings["test_dataset_file"])

combined_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

# Perform an 80/10/10 split
train_data, temp_data = train_test_split(combined_df, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_data = train_data.drop("Unnamed: 0", axis = 1)
val_data = val_data.drop("Unnamed: 0", axis = 1)
test_data = test_data.drop("Unnamed: 0", axis = 1)

# Save these datasets for use in training
train_data.to_csv("train_optuna.csv", index=False)
val_data.to_csv("val_optuna.csv", index=False)
test_data.to_csv("test_optuna.csv", index=False)


# Create an Optuna study with Bayesian Optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)  # Run n trials

# Get the best hyperparameters
best_params = study.best_params
print("Best hyperparameters found:", best_params)








# learning_rate = 0.0002
# train_epoch = 10
# lora_r = 8
# lora_alpha = 16
# lora_dropout = 0.06
# batch_size = 16
# num_layers = 2
# input_sizes = [2048, 1024]
# output_sizes = [1024, 256]
# dropout_rates = [0.06, 0.2]
# activation = 'ReLU'
# grad_accum = 1
# checkpoint = '2744'


# if __name__ == "__main__":

#     result = train(
#             learning_rate, train_epoch, lora_r, lora_alpha, lora_dropout,
#             batch_size, num_layers, input_sizes, output_sizes, dropout_rates, activation, grad_accum, checkpoint
#         )