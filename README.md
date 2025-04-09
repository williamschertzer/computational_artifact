# Computational Artifact: William Schertzer CSE-MSE PhD Qualifying Exam

This repository contains the computational artifact submitted by **William Schertzer** for the PhD CSE-MSE Qualifying Examination. It includes the full workflow for processing, analyzing, and modeling anion exchange membrane (AEM) data extracted from the literature, with the aim of accelerating materials discovery using machine learning.

## Repository Structure

### `raw_data/`
Contains the original AEM datasets, directly extracted and curated from peer-reviewed literature.

### `create_datasets/`
This directory includes scripts for:
- **Cleaning raw data**
- **Generating molecular fingerprints**
- **Performing train-validation-test splits** (including leave-one-group-out cross-validation)

To run the full dataset preparation pipeline:
- Use `process_dataset_all.py` to create a dataset using the entire corpus.
- Use `process_dataset_logo.py` to generate time-dependent leave-one-group-out (LOGO) splits.

Settings for both scripts are specified in `process_dataset.json` (located in the main directory).

Final processed datasets are stored in `create_datasets/shared_datasets/`.

### `analyze_data/`
Contains tools for analyzing the dataset and trained model results:
- `dataset_stats_dir/analyze_data.ipynb`: Analyzes chemical diversity, property distributions, and other key dataset statistics.
- `loss_plots/` and `time_plots/`: Visualize model training loss and time-dependent degradation behaviors, respectively.

### `modeling/`
Contains modeling workflows and environment setup tools:
- `llama_dir/`: Environment setup for training large language models and PyTorch-based workflows. The `make.sh` script builds the required Conda environment.
- `shared_workflows/`: Includes templates and shared code for training and evaluating:
  - Gaussian Process Regression (GPR)
  - Physics-Enforced Neural Networks (PENN)
  - Large Language Models (LLMs)
- `workflows/`: User-specific or experimental workflows not included in version control.

### Root-Level Files
- `fingerprint_cache.pkl`: Cached molecular fingerprints to avoid redundant computation across SMILES strings.
- `FP_scaler.pkl`: Scaler object used to normalize fingerprint vectors.
- `ranked_filtered_candidates.csv`: A ranked list of fluorine-free AEM candidates predicted to meet all target property requirements based on ML models.
- `OH_dataset.csv`: Example non-fingerprinted dataset for hydroxide conductivity (OH conductivity) after preprocessing, ready for machine learning.
