# -*- coding: utf-8 -*-
"""
Path Prediction Dataset Module

This module provides a custom PyTorch Dataset for handling 3D drone 
trajectories, slicing them into observation (input) and prediction 
(target) sequences for sequence-to-sequence learning.

Created on Tue Jul 15 10:15:07 2025
@author: Diyar Altinses, M.Sc.
"""

import torch
import numpy as np
from typing import Tuple, Union


class PathPredictionDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for drone path prediction.

    Given the first `n_obs` steps of a path, the model must predict 
    the subsequent `m_pred` steps.
    """

    def __init__(
        self, 
        paths: Union[np.ndarray, torch.Tensor], 
        n_obs: int = 10, 
        m_pred: int = 10, 
        relative: bool = False
    ):
        """
        Initializes the dataset by validating dimensions and converting 
        data to PyTorch float tensors.

        Parameters
        ----------
        paths : Union[np.ndarray, torch.Tensor]
            Array of shape (N, T, 3) representing the trajectories.
        n_obs : int, optional
            Number of observed steps provided to the model. Defaults to 10.
        m_pred : int, optional
            Number of future steps to predict. Defaults to 10.
        relative : bool, optional
            If True, input and target coordinates are expressed relative 
            to the first observed point. Defaults to False.
        """
        assert len(paths.shape) == 3 and paths.shape[2] == 3, "Expected shape (N, T, 3)"
        
        # Safely convert to PyTorch tensor regardless of input type
        if isinstance(paths, np.ndarray):
            self.paths = torch.from_numpy(paths).float()
        else:
            self.paths = paths.clone().detach().float()
            
        # Optional normalization block (currently disabled)
        # self.mean = self.paths.mean(dim=(0, 1), keepdim=True)
        # self.std = self.paths.std(dim=(0, 1), keepdim=True)
        # self.paths = (self.paths - self.mean) / self.std
        
        self.n_obs = n_obs
        self.m_pred = m_pred
        self.relative = relative
        self.T = paths.shape[1]

        assert n_obs + m_pred <= self.T, "n_obs + m_pred must be <= total path length T"

    def __len__(self) -> int:
        """
        Returns the total number of trajectories in the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the input and target sequences for a given index.

        Parameters
        ----------
        idx : int
            Index of the trajectory to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - input_seq: The observed trajectory sequence of shape (n_obs, 3).
            - target_seq: The target trajectory sequence of shape (m_pred, 3).
        """
        path = self.paths[idx]

        input_seq = path[:self.n_obs]
        target_seq = path[self.n_obs : self.n_obs + self.m_pred]

        if self.relative:
            origin = input_seq[0]
            input_seq = input_seq - origin
            target_seq = target_seq - origin

        return input_seq, target_seq


if __name__ == '__main__':
    # Try block added to gracefully handle missing files during standalone testing
    try:
        paths1 = torch.load('../resources/path1.pt')
        dataset = PathPredictionDataset(paths1, n_obs=10, m_pred=20)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Sample batch
        for x, y in loader:
            print(f"Input shape: {x.shape}")   # Expected: (batch_size, n_obs, 3)
            print(f"Target shape: {y.shape}")  # Expected: (batch_size, m_pred, 3)
            break
            
    except FileNotFoundError:
        print("Data file '../resources/path1.pt' not found. Please generate the dataset first.")