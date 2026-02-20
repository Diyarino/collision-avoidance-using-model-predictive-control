# -*- coding: utf-8 -*-
"""
Variational Autoencoder Component Module

This module provides the latent space mapping for a Variational Autoencoder (VAE).
It includes the fully connected layers for mean and log-variance estimation, 
the reparameterization trick, and the Kullback-Leibler (KL) divergence calculation.

Created on Tue May  3 11:03:54 2022
@author: Diyar Altinses, M.Sc.
"""

import torch
import torch.nn as nn


class Variotional(nn.Module):
    """
    Variational latent space module for an autoencoder.

    Maps deterministic input features into a probabilistic latent space 
    parameterized by a mean and variance, enabling the network to learn 
    continuous, generative representations.
    """

    def __init__(self, latent_dims: int, variotional_dims: int = 10):
        """
        Initializes the mean and variance projection networks.

        Parameters
        ----------
        latent_dims : int
            The dimensionality of the input features to the latent block.
        variotional_dims : int, optional
            The dimensionality of the output probabilistic latent space. 
            Defaults to 10.
        """
        super().__init__()
        
        # Network to estimate the mean (mu) of the latent distribution
        self.mu = nn.Sequential(
            nn.Linear(latent_dims, variotional_dims),
            nn.ReLU(),
            nn.Linear(variotional_dims, variotional_dims)
        )
        
        # Network to estimate the log-variance of the latent distribution
        self.var = nn.Sequential(
            nn.Linear(latent_dims, variotional_dims),
            nn.ReLU(),
            nn.Linear(variotional_dims, variotional_dims)
        )
        
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        var: torch.Tensor, 
        eps: float = 1e-5
    ) -> torch.Tensor:
        """
        Applies the reparameterization trick to allow backpropagation 
        through the stochastic sampling process.

        Parameters
        ----------
        mu : torch.Tensor
            The mean of the latent distribution.
        var : torch.Tensor
            The standard deviation (or scale) of the latent distribution.
        eps : float, optional
            A small epsilon value for numerical stability. Defaults to 1e-5.

        Returns
        -------
        torch.Tensor
            A sampled tensor from the parameterized Normal distribution.
        """
        # PyTorch Normal expects (loc, scale) -> (mean, standard deviation)
        distribution = torch.distributions.Normal(mu, var + eps)
        return distribution.rsample()
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL divergence between the learned latent distribution 
        N(mu, sigma^2) and a standard normal prior N(0, 1).
        
        Parameters
        ----------
        mu : torch.Tensor
            Tensor of shape [batch_size, latent_dim] representing the mean.
        logvar : torch.Tensor
            Tensor of shape [batch_size, latent_dim] representing the log-variance.
            
        Returns
        -------
        torch.Tensor
            Scalar KL loss (mean over the batch).
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass projecting inputs to a sampled latent representation.

        Parameters
        ----------
        inp : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            The stochastically sampled latent vector.
        """
        self.mu_values = self.mu(inp)
        self.log_var_values = self.var(inp)
        
        # Convert log-variance to standard deviation safely
        self.std = torch.exp(self.log_var_values / 2).abs()

        # Calculate KL divergence (Note: currently passes std instead of log_var_values)
        self.kl = self.kl_divergence(self.mu_values, self.std)
        
        return self.reparameterize(self.mu_values, self.std)


if __name__ == '__main__':
    # Standalone testing block
    print('Testing Variotional module...')
    test_input = torch.randn(32, 50)  # batch_size=32, latent_dims=50
    model = Variotional(latent_dims=50, variotional_dims=10)
    output = model(test_input)
    print(f"Output shape: {output.shape}")