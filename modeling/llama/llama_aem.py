import re
import os
import torch
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
import pickle


# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator

def load_lines(jsonl_file : str) -> list[dict]:
    """ Load a JSONL file into list of dictionaries. """
    with open(jsonl_file) as fp:
        jsonlines = list(fp)
    return [ json.loads(json_str) for json_str in jsonlines ]

def gen():
    yield from lines

def predict_prop(prompt, unscale=False):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    result = model.generate(inputs,
        early_stopping=True, num_beams=5,
        max_new_tokens = 10,
        temperature=0.1,
        top_p=0.8,
        do_sample=True)
    response = tokenizer.decode(result[0])

    # Parse the property and value from the prompt or response
    # Assuming the response contains the property and predicted value in a known format
    # For example: "OHCond(mS/cm): 0.85" or similar
    for prop in scalers.keys():
        if prop in prompt and unscale:
            # Extract the scaled value (assuming it follows the property in the response)
            try:
                pattern = r'ASST:\s*([0-9]+\.[0-9]+)'
                # Extract the float number and create a new column
                scaled_value = repsonse.str.extract(pattern).astype(float)

                # Unscale the value using the corresponding scaler
                unscaled_value = scalers[prop].inverse_transform([[scaled_value]])[0][0]

                # Update the response to include the unscaled value
                response = response.replace(f"{scaled_value}", f"{unscaled_value:.3f}")
            except (IndexError, ValueError):
                print(f"Error processing property {prop} in response.")
    
    return response
    
def plot_scatter_with_regression(df, prop, colx, coly, filename, linestyle='--'):
    # Set style
    #sns.set_style("whitegrid")

    df.to_csv(f"/data/wschertzer/aem_aging/modeling/llama/results/{new_model}/data/{filename}.csv")

    df = df[df['prop'] == prop]
    if len(df['value']) != 0:
        # Create joint plot with scatter plot, histograms on both axes, and a regression line
        sns.jointplot(x=colx, y=coly, data=df, kind="reg", height=7,
                                joint_kws={'scatter_kws': {'color': 'blue', 'alpha': 0.5, 's': 100},
                                            'line_kws': {'color': 'red', 'linewidth': 2, 'linestyle':linestyle}, 'ci': None},
                                            marginal_kws=dict(bins=50))

        # Set ticks inwards and proper axis thickness
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in', width=1.5)

        # Calculate Pearson correlation coefficient
        r_value = df[colx].corr(df[coly])
        # Calculate R^2 and RMSE
        r_squared = r2_score(df[colx], df[coly])
        rmse = np.sqrt(mean_squared_error(df[colx], df[coly]))
        print(f"r = {np.round(r_value, 3)}")
        print(f"R^2 = {np.round(r_squared, 3)}")
        print(f"RMSE = {np.round(rmse, 3)}")
        # Set labels and title
        x_axis_label = f"{colx}: all props"
        plt.xlabel(x_axis_label, fontsize=20)
        plt.ylabel(coly, fontsize=20)

        # Set fontsize for x-axis and y-axis numbers
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)

        # Increase x and y axis range
        plt.xlim(df[colx].min() - (df[colx].max()-df[colx].min())*0.05, df[colx].max() + (df[colx].max()-df[colx].min())*0.05)
        plt.ylim(df[coly].min() - (df[coly].max()-df[coly].min())*0.05, df[coly].max() + (df[coly].max()-df[coly].min())*0.05)

        # Add text annotation with correlation coefficient
        plt.annotate(f'r = {r_value:.2f}', xy=(0.1, 0.9), xycoords='axes fraction', ha='left', fontsize=16)
        plt.annotate(f'R^2 = {r_squared:.2f}', xy=(0.1, 0.8), xycoords='axes fraction', ha='left', fontsize=16)
        plt.annotate(f'RMSE = {rmse:.2f}', xy=(0.1, 0.7), xycoords='axes fraction', ha='left', fontsize=16)

        # Save the plot as a PNG file
        os.makedirs(f"/data/wschertzer/aem_aging/modeling/llama/{new_model}/plots", exist_ok = True)
        filename = f"/data/wschertzer/aem_aging/modeling/llama/results/{new_model}/plots/{filename}_{prop.split('(')[0]}.png"
        plt.savefig(filename, bbox_inches='tight')

        # Show plot
        plt.show()

def instruction_format_pred(row):
    property_name_str = str(row['prop'])
    match = re.match(r"([a-zA-Z\s]+)(\([^\)]+\))", property_name_str)
    if match:
        prop_clean = match.group(1).strip()  # e.g., "Swelling"
        unit = match.group(2).strip()        # e.g., "(%)"
    else:
        prop_clean = property_name_str
        unit = ""


    prompt = f"USER: What is the {prop_clean} {unit} for the Anion Exchange Membrane following characteristics: "
    if row['theor_IEC'] is not None:
        prompt += f"(0) Theoretical IEC: {row['theor_IEC']}, "
    if row['smiles1'] is not None:
        prompt += f"(1) comonomers {row['smiles1']}, {row['smiles2']}, {row['smiles3']}, "
    else:
        print("------SMILES1 is None------")
        quit()
    if row['c1'] is not None:
        prompt += f"(2) Relative compositions of smiles1, smiles2 and smiles3: {row['c1']}, {row['c2']}, {row['c3']},"
    else:
        prompt += "(2) Relative compositions of smiles1, smiles2 and smiles3: Unknown, "
    if row['additive_name_1'] is not None:
        prompt += f"(3) Additives names: {row['additive_name_1']}, {row['additive_name_2']},"
    else:
        prompt += "(3) Additives names: None, "
    if row['additive_smiles1'] is not None:
        prompt += f"(4) Additives smiles: {row['additive_smiles1']}, {row['additive_smiles2']}, {row['additive_smiles3']},"
    else:
        prompt += "(4) Additives smiles: None, "
    if row['additive_name_1'] is not None:
        prompt += f"(5) Additives relative compositions: {row['additivec1']}, {row['additivec2']}, {row['additivec3']},"
    else:
        prompt += "(5) Additives relative compositions: None, "
    if row['solvent'] is not None:
        prompt += f"(6) Solvent: {row['solvent']},"
    else:
        prompt += f"(6) Solvent: None,"
    if row['solvent_conc(M)'] is not None:
        prompt += f"(7) Solvent concentration: {row['solvent_conc(M)']} M,"
    else:
        prompt += f"(7) Solvent concentration: None,"
    if row['stab_temp'] is not None:
        prompt += f"(8) Stability test temperature: {row['stab_temp']},"
    else:
        prompt += f"(8) Stability test temperature: None,"
    if row['RH(%)'] is not None:
        prompt += f"(9) Relative Humidity: {row['stab_temp']} %,"
    else:
        prompt += f"(9) Relative Humidity: None,"
    if row['stab_temp'] is not None:
        prompt += f"(10) Time of Measurement: {row['time (h)']} hours,"
    else:
        prompt += f"(10) Time of Measurement: 0 hours,"
    if row['Temp(C)'] is not None:
        prompt += f"(11) Measurement temperature: {row['Temp(C)']} C,"
    else:
        prompt += f"(11) Measurement temperature: None"

    row[dataset_field] = (
        f"[INST] <<SYS>>\n"
        f"You are a helpful chemistry assistant specialized in polymer property prediction.\n"
        f"<</SYS>>\n"
        f"{prompt}"
        f"[/INST]\n"
    )

    return row

def instruction_format_train(row):
    property_name_str = str(row['prop'])
    match = re.match(r"([a-zA-Z\s]+)(\([^\)]+\))", property_name_str)
    if match:
        prop_clean = match.group(1).strip()  # e.g., "Swelling"
        unit = match.group(2).strip()        # e.g., "(%)"
    else:
        prop_clean = property_name_str
        unit = ""

    prompt = f"USER: What is the {prop_clean} {unit} for the Anion Exchange Membrane following characteristics: "
    if row['theor_IEC'] is not None:
        prompt += f"(0) Theoretical IEC: {row['theor_IEC']}, "
    if row['smiles1'] is not None:
        prompt += f"(1) comonomers {row['smiles1']}, {row['smiles2']}, {row['smiles3']}, "
    else:
        print("------SMILES1 is None------")
        quit()
    if row['c1'] is not None:
        prompt += f"(2) Relative compositions of smiles1, smiles2 and smiles3: {row['c1']}, {row['c2']}, {row['c3']},"
    else:
        prompt += "(2) Relative compositions of smiles1, smiles2 and smiles3: Unknown, "
    if row['additive_name_1'] is not None:
        prompt += f"(3) Additives names: {row['additive_name_1']}, {row['additive_name_2']},"
    else:
        prompt += "(3) Additives names: None, "
    if row['additive_smiles1'] is not None:
        prompt += f"(4) Additives smiles: {row['additive_smiles1']}, {row['additive_smiles2']}, {row['additive_smiles3']},"
    else:
        prompt += "(4) Additives smiles: None, "
    if row['additive_name_1'] is not None:
        prompt += f"(5) Additives relative compositions: {row['additivec1']}, {row['additivec2']}, {row['additivec3']},"
    else:
        prompt += "(5) Additives relative compositions: None, "
    if row['solvent'] is not None:
        prompt += f"(6) Solvent: {row['solvent']},"
    else:
        prompt += f"(6) Solvent: None,"
    if row['solvent_conc(M)'] is not None:
        prompt += f"(7) Solvent concentration: {row['solvent_conc(M)']} M,"
    else:
        prompt += f"(7) Solvent concentration: None,"
    if row['stab_temp'] is not None:
        prompt += f"(8) Stability test temperature: {row['stab_temp']},"
    else:
        prompt += f"(8) Stability test temperature: None,"
    if row['RH(%)'] is not None:
        prompt += f"(9) Relative Humidity: {row['stab_temp']} %,"
    else:
        prompt += f"(9) Relative Humidity: None,"
    if row['stab_temp'] is not None:
        prompt += f"(10) Time of Measurement: {row['time (h)']} hours,"
    else:
        prompt += f"(10) Time of Measurement: 0 hours,"
    if row['Temp(C)'] is not None:
        prompt += f"(11) Measurement temperature: {row['Temp(C)']} C,"
    else:
        prompt += f"(11) Measurement temperature: None"
    
    row[dataset_field] = (
        f"[INST] <<SYS>>\n"
        f"You are a helpful assistant specialized in polymer property prediction.\n"
        f"<</SYS>>\n"
        f"{prompt}"
        "[/INST]\n"  # end user input
        f"ASST: {row['value']} {unit}"  # end of assistant answer
    )
    
    return row

def get_preds(row):
    row[dataset_field] = f"{predict_prop(row['prompt_pred'], unscale=unscale)}"
    return row


## Check the environment
assert torch.cuda.is_available(), "Failed to detect GPUs, make sure you set up cuda correctly!"
print("Number of GPUs available: ", torch.cuda.device_count())

major, _ = torch.cuda.get_device_capability()
if major >= 8:
    print("=" * 80)
    print("Your GPU supports bfloat16: accelerate training with bf16=True")
    print("=" * 80)
os.environ["HF_HOME"] = "/data/wschertzer"
print("Huggingface Home =", os.environ['HF_HOME'])

# pip install -U accelerate peft bitsandbytes transformers trl

# The model to load from a directory or the model name from the Hugging Face hub.
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Activate 4-bit precision for model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = True

# Load the entire model on the GPU 0
device_map = {"": 0}

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    token='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'
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# settings
test_preds = True
train_preds = True
val_preds = True
plot_props = ['OHCond(mS/cm)', 'WU(wt%)', 'Swelling(%)']
df = pd.read_csv('/data/wschertzer/aem_aging/modeling/data/scaled_values.csv')
new_model = "aem_aging_1_21_25_llama3"
home_dir = os.getcwd()
scaler_file = '/data/wschertzer/aem_aging/modeling/data/property_scalers.pkl'
unscale = False
with open(scaler_file, 'rb') as f:
    scalers = pickle.load(f)

# Split into train+validation and test
train_val_data, test_data = train_test_split(df, test_size=0.1, random_state=1)

# Split train+validation into train and validation
train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=1)  # Adjust the test_size as needed

# Print the sizes of each split
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")


# Output directory where the model predictions and checkpoints will be stored
os.makedirs("./results", exist_ok = True)
output_dir = f"./results/{new_model}/checkpoints"
plots_dir = f"./results/{new_model}/plots"
data_dir = f"./results/{new_model}/data"
os.makedirs(plots_dir, exist_ok = True)
os.makedirs(data_dir, exist_ok = True)

train_data.to_json(f'{data_dir}/train_data.jsonl', orient='records', lines=True)
val_data.to_json(f'{data_dir}/val_data.jsonl', orient='records', lines=True)
test_data.to_json(f'{data_dir}/test_data.jsonl', orient='records', lines=True)

# generate train dataset
print('loading in training dataset')
lines = load_lines(f"{data_dir}/train_data.jsonl")
train_dataset = Dataset.from_generator(gen)

# generate validation dataset
print('loading in validation dataset')
lines = load_lines(f"{data_dir}/val_data.jsonl")
val_dataset = Dataset.from_generator(gen)

# generate test dataset
print('loading in test dataset')
lines = load_lines(f"{data_dir}/test_data.jsonl")
test_dataset = Dataset.from_generator(gen)

dataset_field = 'prompt'
print('mapping train dataset')
train_dataset = train_dataset.map(instruction_format_train)
print('mapping validation dataset')
val_dataset = val_dataset.map(instruction_format_train)

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 16

# Alpha parameter for LoRA scaling
lora_alpha = 8

# Dropout probability for LoRA layers
lora_dropout = 0.05

################################################################################
# TrainingArguments parameters
################################################################################

# Number of training epochs
num_train_epochs = 10

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 8

# Batch size per GPU for evaluation
per_device_eval_batch_size = 8

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 8

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "linear"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 20

# Log every X updates steps
logging_steps = 20

# eval steps
eval_steps = 20

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 2048

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

# # Set training parameters
# training_arguments = TrainingArguments(
#     output_dir=output_dir,
#     num_train_epochs=num_train_epochs,
#     per_device_train_batch_size=per_device_train_batch_size,
#     evaluation_strategy="steps",
#     eval_steps=100,
#     gradient_accumulation_steps=gradient_accumulation_steps,
#     optim=optim,
#     save_steps=save_steps,
#     logging_steps=logging_steps,
#     learning_rate=learning_rate,
#     weight_decay=weight_decay,
#     fp16=fp16,
#     bf16=bf16,
#     max_grad_norm=max_grad_norm,
#     max_steps=max_steps,
#     warmup_ratio=warmup_ratio,
#     group_by_length=group_by_length,
#     lr_scheduler_type=lr_scheduler_type,
#     report_to="none"
# )

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
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
        report_to="none"
))

# train or load the model
if not os.path.exists(f"./{new_model}/"):

    # Train model

    # Check if checkpoint directory exists and has files
    checkpoint_dir = os.path.join(output_dir,'checkpoint-675')

    if os.path.isdir(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        print(f"Resuming from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        print("No checkpoint found. Training from scratch.")
        trainer.train()

    # Save trained model
    trainer.model.save_pretrained(new_model)

    model.load_adapter(f"./{new_model}/")
else:
    print(f'model exists, loading in: {new_model}')
    model.load_adapter(f"./{new_model}/")

if test_preds:
    # Make prediction for the Test set
    # test set prompt for prediction
    dataset_field = "prompt_pred"
    test_dataset = test_dataset.map(instruction_format_pred)
    print('making predictions on test data')
    dataset_field = "pred_prop"
    test_dataset = test_dataset.map(get_preds)
    test_dataset_df = test_dataset.to_pandas()

    print("Test Set: ")
    print("Size of the dataset: ", test_dataset_df.shape[0])

    # Define a regular expression pattern to capture the float number between "ASST:" and "%"
    pattern = r'ASST:\s*([0-9]+\.[0-9]+)'
    # Extract the float number and create a new column
    test_dataset_df['pred_prop_float'] = test_dataset_df['pred_prop'].str.extract(pattern).astype(float)
    

    # Drop rows where 'pred_PCE_float' column has NaN values
    print(test_dataset_df, test_dataset_df.iloc[0]['pred_prop'],'this is value: ', test_dataset_df.iloc[0]['value'] )

    test_dataset_df = test_dataset_df.dropna(subset=['pred_prop_float'])

    # Parity plot for the Test set
    test_dataset_df['value'] = test_dataset_df['value'].astype('float64')
    for prop in plot_props:
        print(f"plotting {prop}, test set predictions")
        plot_scatter_with_regression(test_dataset_df, prop, "value", "pred_prop_float", "test", linestyle='--')

if train_preds:
    # Train set prompt for prediction
    dataset_field = "prompt_pred"
    train_dataset = train_dataset.map(instruction_format_pred)

    # Make prediction for the Train set
    print('making predictions on train data')
    dataset_field = "pred_prop"
    train_dataset = train_dataset.map(get_preds)
    train_dataset_df = train_dataset.to_pandas()

    # Define a regular expression pattern to capture the float number between "ASST:" and "%"
    pattern = r'ASST:\s*([0-9]+\.[0-9]+)'
    # Extract the float number and create a new column
    train_dataset_df['pred_prop_float'] = train_dataset_df['pred_prop'].str.extract(pattern).astype(float)
    # train_dataset_df.to_csv('train_dataset_with_pred.csv')

    # Drop rows where 'pred_PCE_float' column has NaN values
    train_dataset_df = train_dataset_df.dropna(subset=['pred_prop_float'])

    print("Train Set: ")
    print("Size of the dataset: ", train_dataset_df.shape[0])

    # Parity plot for the Train set
    train_dataset_df['value'] = train_dataset_df['value'].astype('float64')
    for plot in plot_props:
        plot_scatter_with_regression(train_dataset_df, prop, "value", "pred_prop_float", "train", linestyle='--')


if val_preds:
    # Val set prompt for prediction
    dataset_field = "prompt_pred"
    val_dataset = val_dataset.map(instruction_format_pred)

    # Make prediction for the Val set
    print('making predictions on validation data')
    dataset_field = "pred_prop"
    val_dataset = val_dataset.map(get_preds)
    val_dataset_df = train_dataset.to_pandas()

    # Define a regular expression pattern to capture the float number between "ASST:" and "%"
    pattern = r'ASST:\s*([0-9]+\.[0-9]+)'
    # Extract the float number and create a new column
    val_dataset_df['pred_prop_float'] = val_dataset_df['pred_prop'].str.extract(pattern).astype(float)
    # val_dataset_df.to_csv('val_dataset_with_pred.csv')

    # Drop rows where 'pred_PCE_float' column has NaN values
    val_dataset_df = val_dataset_df.dropna(subset=['pred_prop_float'])

    print("Val Set: ")
    print("Size of the dataset: ", val_dataset_df.shape[0])

    # Parity plot for the Val set
    val_dataset_df['value'] = val_dataset_df['value'].astype('float64')
    for prop in plot_props:
        plot_scatter_with_regression(val_dataset_df, prop, "value", "pred_prop_float", "train", linestyle='--')

