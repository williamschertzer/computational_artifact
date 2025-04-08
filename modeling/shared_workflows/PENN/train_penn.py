import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import json
import os
import glob

activation_functions = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "tanh": nn.Tanh()
}

# Load settings.json
with open('settings.json') as f:
    settings = json.load(f)

config = {
    "l1": settings["l1"],
    "l2": settings["l2"],
    "d1": settings["d1"],
    "d2": settings["d2"],
    "activation": settings['activation']
}


class EarlyStopping:
    def __init__(self, patience=500, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0

    def reset(self):
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float('inf')

class OH_PENN(nn.Module):
    def __init__(self, n_fp, device, config):
        """
        Physics-Enforced Neural Network (PENN) for modeling OH conductivity degradation.

        :param n_fp: Number of input polymer fingerprint features
        :param device: CPU or CUDA device
        :param config: Dictionary specifying hidden layer sizes and dropout
        """
        super(OH_PENN, self).__init__()
        self.device = device
        self.mlp = MLP_PENN(n_fp, config, latent_param_size=4)  # Predicts A, B, t0, and alpha
        self.sigmoid = nn.Sigmoid()
        self.exp = torch.exp
        self.to(self.device)

    def forward(self, fp, t, train=False, get_constants=False):
        params = self.mlp(fp)  # Predict 4 physical parameters
        self.A = (self.sigmoid(params[:, 0]) * 2.5).unsqueeze(1)
        self.B = (self.sigmoid(params[:, 1]) * 2.5).unsqueeze(1)
        self.t0 = (self.sigmoid(params[:, 2]) * 3).unsqueeze(1)
        self.alpha = (self.sigmoid(params[:, 3]) * 10).unsqueeze(1)

        # Compute OH conductivity using the given degradation equation
        exponent = -(t - self.t0) * self.alpha       
        exp_term = torch.exp(exponent)
        sigma_t = self.A - ((self.A - self.B) / (1 + exp_term))

        if train:
            return sigma_t, self.A, self.B, self.t0, self.alpha
        elif get_constants:
            return {
                "A": self.A,
                "B": self.B,
                "t0": self.t0,
                "alpha": self.alpha
            }
        else:
            return sigma_t

class MLP_PENN(nn.Module):
    def __init__(self, n_fp, config, latent_param_size=4):
        """
        Multilayer perceptron (MLP) for predicting physics-based degradation parameters.

        :param n_fp: Number of input polymer fingerprint features
        :param config: Dictionary specifying hidden layer sizes and dropout
        :param latent_param_size: Number of output parameters (A, B, t0, alpha)
        """
        super(MLP_PENN, self).__init__()
        l1, l2, d1, d2 = config["l1"], config["l2"], config["d1"], config["d2"]
        self.n_fp = n_fp
        self.layer_1 = nn.Linear(n_fp, l1)
        # self.bn1 = nn.BatchNorm1d(l1)  # Batch normalization after the first layer
        self.d1 = nn.Dropout(p=d1)
        activation_functions = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh()
        }
        self.activation = activation_functions[config.get("activation", "relu")]

        if l2 > 0:
            self.layer_2 = nn.Linear(l1, l2)
            # self.bn2 = nn.BatchNorm1d(l2)  # Batch normalization after the second layer
            self.d2 = nn.Dropout(p=d2)
            self.out = nn.Linear(l2, latent_param_size)
            self.layers = nn.ModuleList([
                self.layer_1, self.d1, self.activation,
                self.layer_2, self.d2, self.activation,
                self.out
            ])
        else:
            self.out = nn.Linear(l1, latent_param_size)
            self.layers = nn.ModuleList([self.layer_1, self.d1, self.activation, self.out])

    def forward(self, fp):
        """
        Forward pass of the MLP.

        :param fp: Polymer fingerprint vector (batch_size, n_fp)
        :return: Predicted physical parameters (batch_size, latent_param_size)
        """
        x = fp.view(-1, self.n_fp)
        for l in self.layers:
            x = l(x)
        return x

def oh_conductivity_loss(pred_sigma_t, true_sigma_t, A, B, t0, alpha, time):
    """
    Custom loss function for training the physics-enforced neural network.

    :param pred_sigma_t: Predicted conductivity values
    :param true_sigma_t: True experimental conductivity values
    :param A, B, t0, alpha: Predicted physics-based parameters
    :param time: Time values corresponding to predictions
    """
    mse_loss = nn.MSELoss()(pred_sigma_t, true_sigma_t)


    # Physics-based constraints to keep parameters in reasonable ranges
    physics_loss = (
        torch.mean(torch.relu(A - 2.5)) +
        torch.mean(torch.relu(B - 2.5)) +
        torch.mean(torch.relu(t0 - 3)) +
        torch.mean(torch.relu(alpha - 10))
    )

    return mse_loss + 0.0 * physics_loss # Combine MSE loss with physics-informed penalty

def prepare_data(df, feature_columns):
    fp_data = df[feature_columns].values
    t_data = df["time(h)"].values
    sigma_data = df["value"].values

    return (
        torch.tensor(fp_data, dtype=torch.float32),
        torch.tensor(t_data, dtype=torch.float32).unsqueeze(1),
        torch.tensor(sigma_data, dtype=torch.float32).unsqueeze(1),
    )

def train(settings):
    # Load datasets
    train_df = pd.read_csv(settings['train_dataset_file']).drop("Unnamed: 0", axis=1).drop('id', axis = 1)
    val_df = pd.read_csv(settings['val_dataset_file']).drop("Unnamed: 0", axis=1).drop('id', axis = 1)

    feature_columns = [col for col in train_df.columns if col not in ["time(h)", "value"]]

    train_dir = os.path.dirname(settings['train_dataset_file'])
    val_dir = os.path.dirname(settings['val_dataset_file'])

    train_i_file = glob.glob(os.path.join(train_dir, 'train_*.csv'))[0]
    val_i_file = glob.glob(os.path.join(val_dir, 'val_*.csv'))[0]

    train_i_df = pd.read_csv(train_i_file)
    val_i_df = pd.read_csv(val_i_file)

    train_df['time(h)'] = train_i_df['time(h)']
    val_df['time(h)'] = val_i_df['time(h)']

    train_fp, train_t, train_sigma = prepare_data(train_df, feature_columns)
    val_fp, val_t, val_sigma = prepare_data(val_df, feature_columns)

    batch_size = settings['batch_size']
    train_loader = DataLoader(TensorDataset(train_fp, train_t, train_sigma), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_fp, val_t, val_sigma), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OH_PENN(n_fp=train_fp.shape[1], device=device, config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    early_stopper = EarlyStopping(patience=200, delta=1e-4)

    epochs = settings['epochs']
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for fp_batch, t_batch, true_sigma_batch in train_loader:
            fp_batch, t_batch, true_sigma_batch = fp_batch.to(device), t_batch.to(device), true_sigma_batch.to(device)
            optimizer.zero_grad()
            sigma_pred, A, B, t0, alpha = model(fp_batch, t_batch, train=True)
            loss = oh_conductivity_loss(sigma_pred, true_sigma_batch, A, B, t0, alpha, t_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for fp_batch, t_batch, true_sigma_batch in val_loader:
                fp_batch, t_batch, true_sigma_batch = fp_batch.to(device), t_batch.to(device), true_sigma_batch.to(device)
                sigma_pred, A, B, t0, alpha = model(fp_batch, t_batch, train=True)
                loss = oh_conductivity_loss(sigma_pred, true_sigma_batch, A, B, t0, alpha, t_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), "oh_penn_model.pth")
    print("Training complete! Model saved.")

    return avg_val_loss

if __name__ == "__main__":
    with open('settings.json', 'r') as f:
        settings = json.load(f)

    result = train(settings)
    print(f"Validation Loss: {result:.4f}")
