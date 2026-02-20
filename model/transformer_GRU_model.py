# -*- coding: utf-8 -*-
"""
Deep Learning Sequence Architectures Module

This module defines various neural network architectures for processing 
and predicting sequential data, including Transformers, GRUs, LSTMs, 
and Variational Autoencoder (VAE) augmented models.

Created on %(date)s
@author: Diyar Altinses, M.Sc.
"""

import torch
import torch.nn as nn

from .base_variational import Variotional


class TransformerEncoder(nn.Module):
    """
    Encodes input sequences using a Transformer encoder architecture.
    """

    def __init__(
        self, 
        input_dim: int = 192, 
        d_model: int = 512, 
        nhead: int = 8, 
        num_encoder_layers: int = 3, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.0
    ):
        """
        Initializes the Transformer Encoder model.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of the input features. Defaults to 192.
        d_model : int, optional
            Dimension of the internal model embeddings. Defaults to 512.
        nhead : int, optional
            Number of attention heads. Defaults to 8.
        num_encoder_layers : int, optional
            Number of stacked encoder layers. Defaults to 3.
        dim_feedforward : int, optional
            Dimension of the internal feedforward network. Defaults to 2048.
        dropout : float, optional
            Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, norm=None
        )
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input sequence.

        Parameters
        ----------
        src : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Encoded sequence of shape (batch_size, seq_len, d_model).
        """
        src = self.linear(src)
        output = self.transformer_encoder(src)
        return output


class GRUModel(nn.Module):
    """
    Encodes input sequences using a Gated Recurrent Unit (GRU) model.
    """

    def __init__(
        self, 
        input_dim: int = 512, 
        hidden_dim: int = 128, 
        output_dim: int = 96, 
        num_layers: int = 5
    ):
        """
        Initializes the GRU Model.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of the input features. Defaults to 512.
        hidden_dim : int, optional
            Dimension of the hidden state. Defaults to 128.
        output_dim : int, optional
            Dimension of the output. Defaults to 96.
        num_layers : int, optional
            Number of GRU layers. Defaults to 5.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence through the GRU and linear decoding layer.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted output of shape (batch_size, output_dim).
        """
        # _ represents the final hidden state; we extract all outputs instead
        hidden, _ = self.gru(x)
        
        # Note: Original code passes the entire sequence through the linear layer,
        # contrary to the comment saying "Nur den letzten hidden state nutzen"
        output = self.fc(hidden)  
        return output


class FNNModel(nn.Module):
    """
    Encodes input sequences using a Feed Forward Neural Network (FNN).
    Note: Bypasses sequence structure by flattening the input.
    """

    def __init__(
        self, 
        input_dim: int = 30, 
        hidden_dim: int = 256, 
        output_dim: int = 270, 
        num_layers: int = 5
    ):
        """
        Initializes the FNN Model.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of the flattened input features. Defaults to 30.
        hidden_dim : int, optional
            Dimension of the hidden state. Defaults to 256.
        output_dim : int, optional
            Dimension of the output. Defaults to 270.
        num_layers : int, optional
            Unused in the current configuration. Defaults to 5.
        """
        super().__init__()
        feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.fnn = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            feedforward, 
            # nn.Unflatten(dim=1, unflattened_size=(3, 90)) # 3d dataset
            nn.Unflatten(dim=1, unflattened_size=(2, 90)) # 2d dataset
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the flattened sequence through the FNN.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            Predicted output of shape (batch_size, 2, 90).
        """
        output = self.fnn(x)
        return output
    

# NOTE: This class definition overwrites a previous LSTMModel definition in memory
class LSTMModel(nn.Module):
    """
    Encodes input sequences using a Long Short-Term Memory (LSTM) model.
    """

    def __init__(
        self, 
        input_dim: int = 512, 
        hidden_dim: int = 128, 
        output_dim: int = 96, 
        num_layers: int = 5
    ):
        """
        Initializes the LSTM Model.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of the input features. Defaults to 512.
        hidden_dim : int, optional
            Dimension of the hidden state. Defaults to 128.
        output_dim : int, optional
            Dimension of the output. Defaults to 96.
        num_layers : int, optional
            Number of LSTM layers. Defaults to 5.
        """
        super().__init__()
        self.gru = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence through the LSTM and linear decoding layer.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted output of shape (batch_size, output_dim).
        """
        hidden, _ = self.gru(x)
        output = self.fc(hidden)  # Note: Uses all hidden states, not just the last
        return output
    
    
class RNNModel(nn.Module):
    """
    Encodes input sequences using a standard Recurrent Neural Network (RNN).
    """

    def __init__(
        self, 
        input_dim: int = 512, 
        hidden_dim: int = 128, 
        output_dim: int = 96, 
        num_layers: int = 5
    ):
        """
        Initializes the RNN Model.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of the input features. Defaults to 512.
        hidden_dim : int, optional
            Dimension of the hidden state. Defaults to 128.
        output_dim : int, optional
            Dimension of the output. Defaults to 96.
        num_layers : int, optional
            Number of RNN layers. Defaults to 5.
        """
        super().__init__()
        self.gru = nn.RNN(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence through the RNN and linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted output of shape (batch_size, output_dim).
        """
        hidden, _ = self.gru(x)
        output = self.fc(hidden)  
        return output
    

class A_LSTMModel(nn.Module):
    """
    Alternative configuration for the LSTM Model (2 layers).
    """

    def __init__(
        self, 
        input_dim: int = 512, 
        hidden_dim: int = 128, 
        output_dim: int = 96, 
        num_layers: int = 2
    ):
        """
        Initializes the A_LSTM Model.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of the input features. Defaults to 512.
        hidden_dim : int, optional
            Dimension of the hidden state. Defaults to 128.
        output_dim : int, optional
            Dimension of the output. Defaults to 96.
        num_layers : int, optional
            Number of LSTM layers. Defaults to 2.
        """
        super().__init__()
        self.gru = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence through the LSTM and linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted output of shape (batch_size, output_dim).
        """
        hidden, _ = self.gru(x)
        output = self.fc(hidden)  
        return output


class VA_LSTMModel(nn.Module):
    """
    Variational-Augmented LSTM Model.
    Processes sequences through an LSTM, decodes them via linear layers, 
    and maps the output to a probabilistic latent space via a VAE component.
    """

    def __init__(
        self, 
        input_dim: int = 512, 
        hidden_dim: int = 128, 
        output_dim: int = 96, 
        num_layers: int = 5
    ):
        """
        Initializes the Variational LSTM Model.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of the input features. Defaults to 512.
        hidden_dim : int, optional
            Dimension of the hidden state. Defaults to 128.
        output_dim : int, optional
            Dimension of the intermediate output. Defaults to 96.
        num_layers : int, optional
            Number of LSTM layers. Defaults to 5.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(output_dim, output_dim)
        
        # Variational component to project into latent distribution
        # self.var_model = Variotional(latent_dims=270, variotional_dims=270)  # 3d dataset
        self.var_model = Variotional(latent_dims=180, variotional_dims=180) # 2d dataset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence and passes it through the VAE block.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted output tensor unflattened to shape (batch_size, 3, 90).
        """
        hidden, _ = self.lstm(x)
        output = self.fc(hidden)  
        output = self.fc_out(self.relu(output))
        
        # Flatten input for VAE (redundant if output is already 2D) and unflatten target
        out = self.var_model(output.flatten(start_dim=1)).unflatten(1, (2, 90)) # 2d dataset
        # out = self.var_model(output.flatten(start_dim=1)).unflatten(1, (3, 90)) # 3d dataset
        return out
    
    
if __name__ == '__main__':
    # --- Example Usage and Standalone Testing ---
    
    # Hyperparameters
    batch_size = 32
    input_size = 288
    d_model = 512
    nhead = 8
    num_encoder_layers = 2
    dim_feedforward = 2048
    hidden_dim = 128
    output_size = 96
    num_layers = 2

    # Instantiate models
    transformer_encoder = TransformerEncoder(
        input_size, d_model, nhead, num_encoder_layers, dim_feedforward
    )
    gru_model = GRUModel(d_model, hidden_dim, output_size, num_layers)
    
    # Chain models together sequentially
    model = nn.Sequential(transformer_encoder, gru_model)
    
    # Create dummy input sequence
    input_data = torch.randn(batch_size, 16, input_size)

    # Execute forward pass
    output = model(input_data)

    # Print resulting tensor shape (Expected: [32, 16, 96])
    print(f"Chained model output shape: {output.shape}")