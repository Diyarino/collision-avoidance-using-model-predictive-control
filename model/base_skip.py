# -*- coding: utf-8 -*-
"""
Model Utility Layers Module

This module provides custom utility PyTorch layers designed for 
debugging and pass-through operations within neural network architectures 
(especially useful in nn.Sequential blocks).

Created on Thu Mar  3 12:53:33 2022
@author: Diyar Altinses, M.Sc.
"""

from typing import Any
import torch


class NoChange(torch.nn.Module):
    """
    A pass-through layer that performs no operations on the input.

    This acts as an identity function. It is particularly useful for 
    dynamically bypassing layers or acting as a placeholder in sequential 
    models.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the pass-through layer.

        Parameters
        ----------
        **kwargs : dict, optional
            Arbitrary keyword arguments (ignored).
        """
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Passes the input tensor through without modifications.

        Parameters
        ----------
        inp : torch.Tensor 
            The input tensor to bypass.

        Returns
        -------
        torch.Tensor
            The exact, unmodified input tensor.
        """
        return inp


class Prints(torch.nn.Module):
    """
    A debugging layer that prints the shape of the input tensor.

    This is highly useful for diagnosing tensor shape mismatches 
    when placed between layers in a torch.nn.Sequential block.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the print debugging layer.

        Parameters
        ----------
        **kwargs : dict, optional
            Arbitrary keyword arguments (ignored).
        """
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Prints the shape of the input tensor and returns it unmodified.

        Parameters
        ----------
        inp : torch.Tensor 
            The input tensor to inspect.

        Returns
        -------
        torch.Tensor
            The unmodified input tensor.
        """
        print(inp.shape)
        return inp