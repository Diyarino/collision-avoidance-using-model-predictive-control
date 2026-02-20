# -*- coding: utf-8 -*-
"""
VRU Dataset and Segmentation Module

This module provides tools to load Vulnerable Road User (VRU) trajectory 
data from CSV files, segment the time-series data using a sliding window 
approach, and serve it via a PyTorch Dataset for sequence-to-sequence modeling.

Created on Tue Jul 15 10:15:07 2025
@author: Diyar Altinses, M.Sc.
"""

from typing import Union, List, Tuple, Sequence

import torch
import pandas as pd
import numpy as np


def sliding_window_segmentation(
    data_list: Sequence[Union[np.ndarray, List]], 
    window_size: int = 100, 
    step_size: int = 50
) -> List[torch.Tensor]:
    """
    Segments variable-length time-series arrays into fixed-length windows 
    using a sliding stride.

    Parameters
    ----------
    data_list : Sequence[Union[np.ndarray, List]]
        List of arrays, each shaped (N_i, 2) where N_i is a variable length.
    window_size : int, optional
        Length of the output windows (e.g., 100 timesteps). Defaults to 100.
    step_size : int, optional
        Stride between window starting points (e.g., 50 for 50% overlap). 
        Defaults to 50.

    Returns
    -------
    List[torch.Tensor]
        List of PyTorch tensors, each representing a segment of shape 
        (window_size, 2).
    """
    windows: List[torch.Tensor] = []
    
    for arr in data_list:
        arr = np.asarray(arr)  # Ensure input is a NumPy array
        n_samples = arr.shape[0]
        
        for start in range(0, n_samples - window_size + 1, step_size):
            end = start + window_size
            window = arr[start:end]
            # Convert to PyTorch float tensor and append
            windows.append(torch.from_numpy(window).float())
            
    return windows


class VRUDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for Vulnerable Road User (VRU) path prediction.

    Loads trajectory data from a list of CSV files, extracts the relevant 
    coordinate columns, and segments them using a sliding window. Given the 
    first `n_obs` steps, the model must predict the subsequent `m_pred` steps.
    """

    def __init__(self, paths: Sequence[str], n_obs: int = 10, m_pred: int = 10):
        """
        Initializes the VRU Dataset by loading, parsing, and segmenting CSV data.

        Parameters
        ----------
        paths : Sequence[str]
            A list of file paths (strings) pointing to the CSV data files.
        n_obs : int, optional
            Number of observed steps provided to the model. Defaults to 10.
        m_pred : int, optional
            Number of future steps to predict. Defaults to 10.
        """
        # Load data from CSVs, extracting all rows and columns from index 2 onward
        data = [pd.read_csv(file).values[:, 2:] for file in paths]
        
        # Segment the data using a sliding window
        windowed_data = sliding_window_segmentation(
            data, 
            window_size=100, 
            step_size=10
        )
        self.trajectories = torch.stack(windowed_data)
        
        self.n_obs = n_obs
        self.m_pred = m_pred
        
    def __len__(self) -> int:
        """
        Returns the total number of segmented windows in the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the observation and prediction sequences for a given index.

        Parameters
        ----------
        idx : int
            Index of the segmented window to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - input_seq: The observed sequence of shape (n_obs, 2).
            - target_seq: The target prediction sequence of shape (m_pred, 2).
        """
        path = self.trajectories[idx]

        input_seq = path[:self.n_obs]
        target_seq = path[self.n_obs : self.n_obs + self.m_pred]

        return input_seq, target_seq


if __name__ == '__main__':
    # Placeholder for script testing
    pass