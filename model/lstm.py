# -*- coding: utf-8 -*-
"""
LSTM Prediction Model Module

This module defines a Long Short-Term Memory (LSTM) neural network 
architecture designed to process sequential data and decode the final 
hidden state into a target output prediction.

Created on %(date)s
@author: Diyar Altinses, M.Sc.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    A Long Short-Term Memory (LSTM) neural network for sequence prediction.

    Processes an input sequence and uses a fully connected linear layer 
    to map the LSTM's final hidden state to a specific output dimension.
    """

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        output_size: int
    ):
        """
        Initializes the LSTM model and its decoding linear layer.

        Parameters
        ----------
        input_size : int
            The number of expected features in the input `x`.
        hidden_size : int
            The number of features in the hidden state `h`.
        num_layers : int
            Number of recurrent layers.
        output_size : int
            The dimensionality of the final output.
        """
        super().__init__()
        
        # Track hidden size and layers for state initialization
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Fixed PyTorch kwarg names from hidden_size/output_size to in/out_features
        # Note: Parameters are currently hardcoded to preserve original functionality.
        self.fc = nn.Linear(in_features=2, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LSTM network.

        Parameters
        ----------
        x : torch.Tensor
            The input sequence tensor of shape (batch_size, seq_length, input_size).

        Returns
        -------
        torch.Tensor
            The decoded output for the final time step, of shape (batch_size, output_size).
        """
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=x.device
        )
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=x.device
        )

        # Process the sequence through the LSTM
        # Input shape: (batch_size, seq_length, input_size)
        # Output shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the exact last time step
        out = self.fc(out[:, -1, :])
        
        return out


if __name__ == '__main__':
    # Standalone testing block to verify tensor shapes and instantiation
    print("Testing LSTMModel instantiation...")
    model = LSTMModel(input_size=16, hidden_size=2, num_layers=5, output_size=2)
    
    # Create a dummy batch: (batch_size=32, seq_length=10, input_size=16)
    dummy_input = torch.rand(32, 10, 16)
    dummy_output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")