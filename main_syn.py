# -*- coding: utf-8 -*-
"""
Path Prediction Training Module

This script trains a Variational LSTM (VA_LSTMModel) to predict future paths 
based on historical observation data. It utilizes PyTorch's Automatic Mixed 
Precision (AMP) for faster training and logs performance metrics.

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
from dataset.synthetic_dataset import PathPredictionDataset
from model.transformer_GRU_model import GRUModel, LSTMModel, FNNModel, RNNModel, A_LSTMModel, VA_LSTMModel

# --- Configuration & Hyperparameters ---
NUM_TRIALS = 10
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_OBSERVATIONS = 10
M_PREDICTIONS = 90
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, dataloader, criterion):
    """
    Evaluates the model on the provided dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the test/validation set.
        criterion (nn.Module): The loss function.

    Returns:
        torch.Tensor: The average loss over the evaluation dataset.
    """
    buffer = 0.0
    
    with torch.no_grad():
        for index, (data, label) in enumerate(dataloader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            
            # Forward pass
            if isinstance(model, VA_LSTMModel):
                _ = model(data.permute(0, 2, 1)).permute(0, 2, 1)
                prediction = model.var_model.mu_values.unflatten(1, (3, M_PREDICTIONS)).permute(0, 2, 1)
            else:
                prediction = model(data.permute(0, 2, 1)).permute(0, 2, 1)
        
            buffer += criterion(prediction, label)
            
    # Calculate average loss
    test_loss = buffer / len(dataloader)
    return test_loss


def main():
    """Main training pipeline for the path prediction model."""
    
    configure_plt()

    # --- Load Data ---
    paths1 = torch.load('./resources/path1.pt')
    train_dataset = PathPredictionDataset(paths1, n_obs=N_OBSERVATIONS, m_pred=M_PREDICTIONS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    paths2 = torch.load('./resources/path2.pt')
    test_dataset = PathPredictionDataset(paths2, n_obs=N_OBSERVATIONS, m_pred=M_PREDICTIONS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Trials Loop ---
    for trial in range(NUM_TRIALS):
        print(f"--- Starting Trial {trial + 1}/{NUM_TRIALS} ---")
        
        # 1. Setup Folders
        folder = GenerateFolder(GenerateAll=False)
        train_folder = folder.GenerateTrainFolder(generate=True, name='_LSTM_VA')
        data_folder, img_folder, net_folder, table_folder = folder.GenerateDataFolder(
            generate=True, location=train_folder.trainfolder
        )

        # 2. Initialize Model
        
        # model = GRUModel(input_dim = 10, hidden_dim = 90, output_dim = 90, num_layers = 2).to(DEVICE)
        # model = LSTMModel(input_dim = 10, hidden_dim = 90, output_dim = 90, num_layers = 2).to(DEVICE)
        # model = FNNModel(input_dim = 30, hidden_dim = 256, output_dim = 270, num_layers = 5).to(DEVICE)
        # model = RNNModel(input_dim = 10, hidden_dim = 90, output_dim = 90, num_layers = 2).to(DEVICE)
        # model = A_LSTMModel(input_dim = 10, hidden_dim = 90, output_dim = 90, num_layers = 2).to(DEVICE)
        model = VA_LSTMModel(input_dim=10, hidden_dim=90, output_dim=90, num_layers=2).to(DEVICE)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scaler = GradScaler()
        
        storage = DataStorage(
            ['Epochs', 'Batch', 'loss', 'testloss'], 
            show=2, 
            line=100, 
            header=500, 
            precision=5
        )
        
        # 3. Initial Testing
        test_loss = evaluate_model(model, test_loader, criterion)
        
        # 4. Training Loop
        batch_idx = 0
        
        for epoch in range(EPOCHS):
            for _, (data, label) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Mixed precision context
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    data, label = data.to(DEVICE), label.to(DEVICE)
                    
                    if isinstance(model, VA_LSTMModel):
                        mu = model.var_model.mu_values.unflatten(1, (3, 90)).permute(0, 2, 1)
                        log_var = model.var_model.log_var_values.unflatten(1, (3, 90)).permute(0, 2, 1)
                        std = model.var_model.std.unflatten(1, (3, 90)).permute(0, 2, 1)
                        
                        # Custom Negative Log-Likelihood Loss calculation
                        loss = 0.5 * ((label - mu) ** 2 / std ** 2 + log_var + 1.837).mean()
                    else:
                        prediction = model(data.permute(0, 2, 1)).permute(0, 2, 1)
                        loss = criterion(prediction, label) 

                # Backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                batch_idx += 1
                
                # Log metrics
                storage.Store([epoch, batch_idx, loss.item(), test_loss.item()])
                
                # Periodic Evaluation
                if batch_idx % 100 == 0:
                    test_loss = evaluate_model(model, test_loader, criterion)
                    
            # Update learning rate schedule at the end of each epoch
            scheduler.step()
            
        # 5. Save Data and Models
        torch.save(model.state_dict(), os.path.join(folder.netfolder, 'model_weights.pth'))
        torch.save(model, os.path.join(folder.netfolder, 'model.pt'))
        torch.save(storage, os.path.join(folder.datafolder, 'train_storage.pt'))
        
        # 6. Plotting
        fig = plot_losses(storage)
        fig.savefig(
            os.path.join(train_folder.trainfolder, 'loss_plot.png'), 
            dpi=300, 
            bbox_inches='tight'
        )


if __name__ == "__main__":
    main()