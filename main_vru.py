# -*- coding: utf-8 -*-
"""
VRU Trajectory Prediction Training Module

This script trains various models (FNN, LSTM, etc.) to predict future 
Vulnerable Road User (VRU) paths based on 2D historical observation data.

Created on Tue Jul 15 10:20:42 2025
@author: Altinses
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from utils.generate_folder import GenerateFolder
from utils.data_storage import DataStorage
from visualize.config_plots import configure_plt
from visualize.plots import plot_losses
from dataset.vru_dataset import VRUDataset
from model.transformer_GRU_model import (
    GRUModel, LSTMModel, FNNModel, RNNModel, A_LSTMModel, VA_LSTMModel
)

# --- Configuration & Hyperparameters ---
NUM_TRIALS = 10
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_OBSERVATIONS = 10
M_PREDICTIONS = 90
FEATURE_DIMS = 2  # 2D Data (X, Y)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, dataloader, criterion):
    """Evaluates the model on the test set."""
    model.eval()
    buffer = 0.0
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            
            # Forward pass with permutation to (Batch, Features, Seq)
            output = model(data.permute(0, 2, 1)).permute(0, 2, 1)
            
            # If VA model, we use the mu values for evaluation
            if isinstance(model, VA_LSTMModel):
                prediction = model.var_model.mu_values.unflatten(
                    1, (FEATURE_DIMS, M_PREDICTIONS)
                ).permute(0, 2, 1)
            else:
                prediction = output
                
            buffer += criterion(prediction, label).item()
            
    return buffer / len(dataloader)


def main():
    configure_plt()

    # --- Load Data ---
    dataset_path = os.path.join(
        os.path.dirname(os.getcwd()), 'datasets', 'trajectory', 
        'dataset', 'VRU_dataset', 'pedestrians', 'moving'
    )
    files = os.listdir(dataset_path)
    
    # Split indices
    indices = torch.randperm(len(files))
    train_indices = indices[:220]
    test_indices = indices[220:]

    train_files = [os.path.join(dataset_path, files[idx]) for idx in train_indices]
    test_files = [os.path.join(dataset_path, files[idx]) for idx in test_indices]

    train_dataset = VRUDataset(train_files, n_obs=N_OBSERVATIONS, m_pred=M_PREDICTIONS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = VRUDataset(test_files, n_obs=N_OBSERVATIONS, m_pred=M_PREDICTIONS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Training Trials ---
    for trial in range(NUM_TRIALS):
        print(f"--- Starting Trial {trial + 1}/{NUM_TRIALS} ---")
        
        # 1. Setup Folders
        folder = GenerateFolder(GenerateAll=False)
        train_folder = folder.GenerateTrainFolder(generate=True, name='_FNN_VRU')
        paths = folder.GenerateDataFolder(generate=True, location=train_folder.trainfolder)
        data_f, img_f, net_f, table_f = paths

        # 2. Initialize Model
        # model = GRUModel(input_dim = 10, hidden_dim = 90, output_dim = 90, num_layers = 2).to(DEVICE)
        # model = LSTMModel(input_dim = 10, hidden_dim = 90, output_dim = 90, num_layers = 2).to(DEVICE)
        # model = FNNModel(input_dim = 20, hidden_dim = 128, output_dim = 180, num_layers = 5).to(DEVICE)
        # model = RNNModel(input_dim = 10, hidden_dim = 90, output_dim = 90, num_layers = 2).to(DEVICE)
        # model = A_LSTMModel(input_dim = 10, hidden_dim = 90, output_dim = 90, num_layers = 2).to(DEVICE)
        model = VA_LSTMModel(input_dim = 10, hidden_dim = 90, output_dim = 90, num_layers = 2).to(DEVICE)
        
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scaler = GradScaler()
        
        storage = DataStorage(['Epochs', 'Batch', 'loss', 'testloss'], show=2, line=100, header=500, precision=5)
        
        # 3. Initial Test Loss
        test_loss = evaluate_model(model, test_loader, criterion)
        batch_count = 0

        # 4. Training Loop
        for epoch in range(EPOCHS):
            model.train()
            for data, label in train_loader:
                optimizer.zero_grad()
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    data, label = data.to(DEVICE), label.to(DEVICE)
                    
                    # Permute for Recurrent layers (Batch, Channels, Seq)
                    output = model(data.permute(0, 2, 1)).permute(0, 2, 1)
                    
                    if isinstance(model, VA_LSTMModel):
                        # Special 2D unflattening for VRU data
                        mu = model.var_model.mu_values.unflatten(1, (FEATURE_DIMS, M_PREDICTIONS)).permute(0, 2, 1)
                        std = model.var_model.std.unflatten(1, (FEATURE_DIMS, M_PREDICTIONS)).permute(0, 2, 1)
                        log_var = model.var_model.log_var_values.unflatten(1, (FEATURE_DIMS, M_PREDICTIONS)).permute(0, 2, 1)
                        
                        loss = 0.5 * ((label - mu) ** 2 / std ** 2 + log_var + 1.837).mean()
                    else:
                        loss = criterion(output, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                batch_count += 1
                storage.Store([epoch, batch_count, loss.item(), test_loss])
                
                # Validation every 100 batches
                if batch_count % 100 == 0:
                    test_loss = evaluate_model(model, test_loader, criterion)
                    model.train()

            scheduler.step()

        # 5. Save Model and Stats
        torch.save(model.state_dict(), os.path.join(net_f, 'model_weights.pth'))
        torch.save(model, os.path.join(net_f, 'model.pt'))
        torch.save(storage, os.path.join(data_f, 'train_storage.pt'))
        
        # 6. Final Plot
        fig_loss = plot_losses(storage)
        fig_loss.savefig(os.path.join(train_folder.trainfolder, 'loss_plot.png'), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()