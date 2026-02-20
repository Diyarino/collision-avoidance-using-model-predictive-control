# -*- coding: utf-8 -*-
"""
Loss Visualization Module

This module provides utility functions for plotting training and testing
loss curves from a custom DataStorage object, applying a moving average 
to smooth the training data for better readability.

Created on %(date)s
@author: Diyar Altinses, M.Sc.
"""

from typing import Any

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_losses(storage: Any) -> plt.Figure:
    """
    Generates a plot of the training and testing losses over batches.

    Parameters
    ----------
    storage : DataStorage
        The DataStorage instance containing the logged training metrics. 
        It must contain 'loss', 'testloss', and 'Batch' arrays within its 
        `StoredValues` dictionary.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure containing the loss curves, 
        which can then be saved or displayed.
    """
    fig = plt.figure(figsize=(6, 3))
    
    # Plot raw training loss with high transparency
    plt.plot(
        storage.StoredValues['loss'], 
        alpha=0.3, 
        color='royalblue'
    )
    
    # Plot smoothed training loss (moving average over a window of 100)
    smoothed_loss = np.convolve(
        storage.StoredValues['loss'], 
        np.ones(100) / 100.0, 
        mode='valid'
    )
    plt.plot(
        smoothed_loss, 
        alpha=1.0, 
        color='royalblue', 
        label='Training'
    )
    
    # Plot testing loss
    # Sliced from index 100 onward to align with the smoothed training loss
    plt.plot(
        storage.StoredValues['Batch'][100:], 
        storage.StoredValues['testloss'][100:], 
        alpha=1.0, 
        color='red', 
        label='Testing'
    )
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.ylim(0.0, 1.5)
    plt.tight_layout()
    
    return fig


if __name__ == '__main__':
    # Placeholder for standalone testing
    test_tensor = torch.rand(1)