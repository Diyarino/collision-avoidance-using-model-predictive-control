# -*- coding: utf-8 -*-
"""
Dynamic Sequence Generator Module

This module provides a utility function to dynamically construct PyTorch 
nn.Sequential models based on a string representation of layer dimensions 
and configuration parameters.

Created on Thu Aug 11 11:34:57 2022
@author: Diyar Altinses, M.Sc.
"""

from typing import Dict, Optional, Any
import torch

from .base_skip import NoChange


def generate_sequence(
    net_setup: str, 
    layer: Optional[str] = 'Linear', 
    layer_parameters: Optional[Dict[str, Any]] = None,
    activation: Optional[str] = 'ReLU', 
    activation_parameters: Optional[Dict[str, Any]] = None,
    batch_norm: Optional[str] = None, 
    batch_norm_parameters: Optional[Dict[str, Any]] = None,
    dropout: Optional[str] = None, 
    dropout_parameters: Optional[Dict[str, Any]] = None,
    pooling: bool = True,
    sequence: Optional[torch.nn.Module] = None
) -> torch.nn.Sequential:
    """
    Generates a PyTorch Sequential model dynamically based on a setup string.

    Parameters
    ----------
    net_setup : str
        A string representing the dimensions of the layers, separated by 
        hyphens (e.g., "32-64-128" for layers transitioning from 32 to 64 to 128).
    layer : str, optional
        The name of the PyTorch layer class to use (e.g., 'Linear', 'Conv1d'). 
        Defaults to 'Linear'.
    layer_parameters : dict, optional
        Additional kwargs passed to the layer instantiation. Defaults to None.
    activation : str, optional
        The name of the PyTorch activation class (e.g., 'ReLU'). Defaults to 'ReLU'.
    activation_parameters : dict, optional
        Additional kwargs passed to the activation instantiation. Defaults to None.
    batch_norm : str, optional
        The name of the PyTorch batch normalization class. Defaults to None.
    batch_norm_parameters : dict, optional
        Additional kwargs passed to the batch norm instantiation. Defaults to None.
    dropout : str, optional
        If provided, uses `torch.nn.Dropout`. Defaults to None.
    dropout_parameters : dict, optional
        Additional kwargs passed to the dropout instantiation. Defaults to None.
    pooling : bool, optional
        If True, applies `torch.nn.MaxPool2d` before the activation layer. 
        Defaults to True.
    sequence : torch.nn.Module, optional
        An existing Sequential module to append layers to. If None, a new 
        Sequential block is created. Defaults to None.

    Returns
    -------
    torch.nn.Sequential
        The newly generated or updated sequential block.
    """
    # Safely handle mutable default arguments
    layer_parameters = layer_parameters or {}
    activation_parameters = activation_parameters or {}
    batch_norm_parameters = batch_norm_parameters or {}
    dropout_parameters = dropout_parameters or {}

    network_setup = net_setup.split('-')
    
    if sequence is None:
        sequence = torch.nn.Sequential()
        
    # Dynamically fetch the PyTorch classes by name, or use NoChange pass-through
    layer_class = getattr(torch.nn, layer) if layer else NoChange
    activation_class = getattr(torch.nn, activation) if activation else NoChange
    batch_norm_class = getattr(torch.nn, batch_norm) if batch_norm else NoChange
    dropout_class = torch.nn.Dropout if dropout else NoChange
    pool_class = torch.nn.MaxPool2d if pooling else NoChange
    
    for idx in range(len(network_setup) - 1):
        buffer_sequence = torch.nn.Sequential()
        
        in_features = int(network_setup[idx])
        out_features = int(network_setup[idx + 1])
        
        # Prepare Batch Norm kwargs, enforcing the expected number of features
        batch_norm_dict = {"num_features": out_features, **batch_norm_parameters}
        
        # 1. Main Calculation Layer (e.g., Linear, Conv2d)
        buffer_sequence.add_module(
            f"{len(buffer_sequence)}_Calc", 
            layer_class(in_features, out_features, **layer_parameters)
        )
        
        # 2. Append Activation, Pooling, Norm, and Dropout (except for the final layer)
        if idx < (len(network_setup) - 2):
            buffer_sequence.add_module(
                f"{len(buffer_sequence)}_Pool", 
                pool_class(**{'kernel_size': 2})
            )
            buffer_sequence.add_module(
                f"{len(buffer_sequence)}_Act", 
                activation_class(**activation_parameters)
            )
            buffer_sequence.add_module(
                f"{len(buffer_sequence)}_BatchNorm", 
                batch_norm_class(**batch_norm_dict)
            )
            buffer_sequence.add_module(
                f"{len(buffer_sequence)}_Drop", 
                dropout_class(**dropout_parameters)
            )
        
        # Add the nested block to the main sequence
        sequence.add_module(f"{len(sequence)}_block", buffer_sequence)      
            
    return sequence