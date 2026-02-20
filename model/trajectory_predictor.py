# -*- coding: utf-8 -*-
"""
LSTM Sequence-to-Sequence Model Module

This module implements an Encoder-Decoder LSTM architecture for 
autoregressive trajectory prediction.

Created on %(date)s
@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    Encoder portion of the Seq2Seq model.
    Processes the input sequence and outputs the final hidden and cell states.
    """

    def __init__(
        self, 
        input_features: int, 
        hidden_size: int, 
        num_layers: int, 
        dropout: float
    ):
        """
        Initializes the LSTM Encoder.

        Parameters
        ----------
        input_features : int
            Number of features per timestamp (e.g., 3 for x, y, z).
        hidden_size : int
            Dimensionality of the LSTM hidden states.
        num_layers : int
            Number of stacked LSTM layers.
        dropout : float
            Dropout probability applied between LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_features, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the LSTM Encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor of shape (batch_size, input_seq_len, input_features).

        Returns
        -------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            - out: LSTM output for each timestamp of shape 
              (batch_size, input_seq_len, hidden_size).
            - (hidden, cell): The final hidden and cell states of the encoder, 
              both of shape (num_layers, batch_size, hidden_size).
        """
        # h0 and c0 are automatically initialized to zeros if not provided
        out, (hidden, cell) = self.lstm(x)
        return out, (hidden, cell)


class LSTMDecoder(nn.Module):
    """
    Decoder portion of the Seq2Seq model.
    Generates the output sequence autoregressively step-by-step.
    """

    def __init__(
        self, 
        output_features: int, 
        hidden_size: int, 
        num_layers: int, 
        output_seq_len: int
    ):
        """
        Initializes the LSTM Decoder.

        Parameters
        ----------
        output_features : int
            Number of output features per timestamp (e.g., 3 for x, y, z).
        hidden_size : int
            Dimensionality of the LSTM hidden states.
        num_layers : int
            Number of stacked LSTM layers.
        output_seq_len : int
            The total length of the sequence to generate.
        """
        super().__init__()
        self.output_seq_len = output_seq_len
        self.output_features = output_features
        self.num_layers = num_layers

        # LSTM cell unrolled during decoding
        self.lstm = nn.LSTM(
            output_features, 
            hidden_size, 
            num_layers, 
            batch_first=True
        )

        # Linear projection layer from hidden state to final feature dimensions
        self.fc_out = nn.Linear(hidden_size, output_features)

    def forward(
        self, 
        encoder_hidden: torch.Tensor, 
        encoder_cell: torch.Tensor,
        initial_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the LSTM Decoder.

        Generates the output sequence autoregressively (the prediction of the 
        previous step becomes the input for the next step).

        Parameters
        ----------
        encoder_hidden : torch.Tensor
            Final hidden state from the encoder. Shape: (num_layers, batch_size, hidden_size).
        encoder_cell : torch.Tensor
            Final cell state from the encoder. Shape: (num_layers, batch_size, hidden_size).
        initial_input : torch.Tensor
            The starting point for decoding (e.g., the last known point of the input). 
            Shape: (batch_size, 1, output_features).

        Returns
        -------
        torch.Tensor
            The predicted output sequence of shape (batch_size, output_seq_len, output_features).
        """
        # Initialize decoder states with the final encoder states
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        outputs = []
        decoder_input = initial_input

        for _ in range(self.output_seq_len):
            # Step the LSTM forward one unit of time
            lstm_out, (decoder_hidden, decoder_cell) = self.lstm(
                decoder_input, (decoder_hidden, decoder_cell)
            )

            # Project the LSTM output back to the spatial feature dimension
            prediction = self.fc_out(lstm_out)
            outputs.append(prediction)

            # Autoregressive update: feed the current prediction into the next step
            decoder_input = prediction

            # Concatenate all predicted steps along the sequence dimension (dim=1)
        final_outputs = torch.cat(outputs, dim=1)
        return final_outputs


class LSTMSeq2Seq(nn.Module):
    """
    End-to-End LSTM Sequence-to-Sequence model for trajectory prediction.
    """

    def __init__(
        self,
        input_features: int = 3,         
        input_seq_len: int = 20,         
        output_seq_len: int = 80,        
        hidden_size: int = 256,          
        num_layers: int = 2,             
        dropout: float = 0.1             
    ):
        """
        Initializes the LSTM Seq2Seq model.

        Parameters
        ----------
        input_features : int, optional
            Number of features per timestamp (e.g., 3 for x, y, z). Defaults to 3.
        input_seq_len : int, optional
            Length of the input observation sequence. Defaults to 20.
        output_seq_len : int, optional
            Length of the output prediction sequence. Defaults to 80.
        hidden_size : int, optional
            Dimensionality of the LSTM hidden states. Defaults to 256.
        num_layers : int, optional
            Number of LSTM layers for both encoder and decoder. Defaults to 2.
        dropout : float, optional
            Dropout rate for LSTM layers. Defaults to 0.1.
        """
        super().__init__()
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.input_features = input_features

        self.encoder = LSTMEncoder(input_features, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(input_features, hidden_size, num_layers, output_seq_len)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complete LSTM Seq2Seq model.

        Parameters
        ----------
        src : torch.Tensor
            Input sequence tensor of shape (batch_size, input_seq_len, input_features).

        Returns
        -------
        torch.Tensor
            Predicted output sequence tensor of shape 
            (batch_size, output_seq_len, input_features).
        """
        # Encoder pass
        _, (encoder_hidden, encoder_cell) = self.encoder(src)

        # Decoder initial input: slice the very last timestamp of the source sequence
        initial_decoder_input = src[:, -1:, :] 

        # Decoder pass
        predicted_trajectory = self.decoder(
            encoder_hidden, 
            encoder_cell, 
            initial_decoder_input
        )

        return predicted_trajectory


if __name__ == "__main__":
    # --- Example Usage and Standalone Testing ---
    
    # Model Parameters
    INPUT_FEATURES = 3
    INPUT_SEQ_LEN = 20
    OUTPUT_SEQ_LEN = 80
    HIDDEN_SIZE = 256  
    NUM_LAYERS = 2     
    DROPOUT = 0.1
    BATCH_SIZE = 32

    # Verify CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate model
    model = LSTMSeq2Seq(
        input_features=INPUT_FEATURES,
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=OUTPUT_SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    print("\nModel Architecture:")
    print(model)

    # Create dummy input
    dummy_input = torch.randn(BATCH_SIZE, INPUT_SEQ_LEN, INPUT_FEATURES).to(device)
    print(f"\nDummy Input Shape: {dummy_input.shape}")

    # Execute Inference Forward Pass
    model.eval()  # Disables dropout for inference
    with torch.no_grad():  
        predicted_trajectory = model(dummy_input)
    print(f"Predicted Trajectory Shape: {predicted_trajectory.shape}")

    # Validate output shape
    expected_output_shape = (BATCH_SIZE, OUTPUT_SEQ_LEN, INPUT_FEATURES)
    assert predicted_trajectory.shape == expected_output_shape, \
        f"Error: Expected shape {expected_output_shape}, Got {predicted_trajectory.shape}"
    print("\nOutput shape is correct!")

    # Check parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # --- Basic Training Loop Example ---
    print("\n--- Running basic training loop example ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Dummy target data (Ground Truth)
    dummy_target = torch.randn(BATCH_SIZE, OUTPUT_SEQ_LEN, INPUT_FEATURES).to(device)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()          # Re-enable dropout for training
        optimizer.zero_grad()  # Reset gradients

        output = model(dummy_input)             # Forward pass
        loss = criterion(output, dummy_target)  # Compute loss

        loss.backward()   # Backpropagation
        optimizer.step()  # Update weights

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("\nBasic training loop complete.")
    print("Remember to use standard normalization, dataloaders, and validation evaluation!")