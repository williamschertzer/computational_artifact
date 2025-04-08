import torch
import torch.nn as nn
from enum import Enum
import torch.optim as optim

class OH_Constants(Enum):
    A = "A"       # Asymptotic conductivity (long-term)
    B = "B"       # Magnitude of conductivity loss
    t0 = "t0"     # Inflection point
    tau = "tau"   # Characteristic time decay

class LossTypes(Enum):
    tot_train = "epoch_train_loss"
    avg_train = "avg_train_loss"
    tot_val = "epoch_validation_loss"
    avg_val = "avg_validation_loss"
    phys = "physics_informed_loss"

class EarlyStopping:
    def __init__(self, patience=30, delta=0):
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
    def __init__(self, n_fp, device, config={"l1": 120, "l2": 120, "d1": 0.2, "d2": 0.2}):
        """
        Physics-Enforced Neural Network (PENN) for modeling OH conductivity degradation.

        :param n_fp: Number of input polymer fingerprint features
        :param device: CPU or CUDA device
        :param config: Dictionary specifying hidden layer sizes and dropout
        """
        super(OH_PENN, self).__init__()
        self.device = device
        self.mlp = MLP_PENN(n_fp, config, latent_param_size=4)  # Predicts A, B, t0, and tau
        self.sigmoid = nn.Sigmoid()
        self.exp = torch.exp
        self.to(self.device)

    def forward(self, fp, t, train=False, get_constants=False):
        params = self.mlp(fp)  # Predict 4 physical parameters
        self.A = self.sigmoid(params[:, 0])
        self.B = self.sigmoid(params[:, 1])
        self.t0 = self.sigmoid(params[:, 2])
        self.tau = torch.clamp(self.sigmoid(params[:, 3]), min=1e-6)  # Clamp tau to avoid division by zero

        # Compute OH conductivity using the given degradation equation
        clamped_exponent = torch.clamp(-(t - self.t0) / self.tau, min=-50, max=50)
        exp_term = torch.exp(clamped_exponent)
        sigma_t = self.A - (self.B / (1 + exp_term))

        if train:
            return sigma_t, self.A, self.B, self.t0, self.tau
        elif get_constants:
            return {
                "A": self.A,
                "B": self.B,
                "t0": self.t0,
                "tau": self.tau
            }
        else:
            return sigma_t

class MLP_PENN(nn.Module):
    def __init__(self, n_fp, config, latent_param_size=4):
        """
        Multilayer perceptron (MLP) for predicting physics-based degradation parameters.

        :param n_fp: Number of input polymer fingerprint features
        :param config: Dictionary specifying hidden layer sizes and dropout
        :param latent_param_size: Number of output parameters (A, B, t0, tau)
        """
        super(MLP_PENN, self).__init__()
        l1, l2, d1, d2 = config["l1"], config["l2"], config["d1"], config["d2"]
        self.n_fp = n_fp
        self.layer_1 = nn.Linear(n_fp, l1)
        self.relu = nn.ReLU()
        self.d1 = nn.Dropout(p=d1)

        if l2 > 0:
            self.layer_2 = nn.Linear(l1, l2)
            self.d2 = nn.Dropout(p=d2)
            self.out = nn.Linear(l2, latent_param_size)
            self.layers = nn.ModuleList([self.layer_1, self.d1, self.relu, self.layer_2, self.d2, self.relu, self.out])
        else:
            self.out = nn.Linear(l1, latent_param_size)
            self.layers = nn.ModuleList([self.layer_1, self.d1, self.relu, self.out])

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

def oh_conductivity_loss(pred_sigma_t, true_sigma_t, A, B, t0, tau, time):
    """
    Custom loss function for training the physics-enforced neural network.

    :param pred_sigma_t: Predicted conductivity values
    :param true_sigma_t: True experimental conductivity values
    :param A, B, t0, tau: Predicted physics-based parameters
    :param time: Time values corresponding to predictions
    """
    mse_loss = nn.MSELoss()(pred_sigma_t, true_sigma_t)


    # Physics-based constraints to keep parameters in reasonable ranges
    physics_loss = (
        torch.mean(torch.relu(A - 2.5)) +  # A should not exceed 2.5
        torch.mean(torch.relu(B - 2.5)) +  # B should not exceed 2.5
        torch.mean(torch.relu(t0 - 3)) +  # t0 should be within a reasonable timeframe
        torch.mean(torch.relu(tau - 10))  # tau should be within a realistic range
    )

    return mse_loss + 0.1 * physics_loss  # Combine MSE loss with physics-informed penalty

