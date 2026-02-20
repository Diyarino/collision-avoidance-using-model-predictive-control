# -*- coding: utf-8 -*-
"""
3D Path Visualization Module

This module provides tools for plotting and visualizing 3D drone 
trajectories and their collision points using Matplotlib.

Created on Tue Jul 15 10:00:25 2025
@author: Altinses
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_paths(path1: np.ndarray, path2: np.ndarray) -> None:
    """
    Plots two 3D drone paths and highlights their start and collision points.

    Parameters
    ----------
    path1 : np.ndarray
        Trajectory of the first drone, expected shape (T, 3).
    path2 : np.ndarray
        Trajectory of the second drone, expected shape (T, 3).

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot continuous trajectories
    ax.plot(*path1.T, label="Drone 1", c='blue')
    ax.plot(*path2.T, label="Drone 2", c='red')
    
    # Plot starting positions
    ax.scatter(*path1[0], c='blue', marker='o', label='Start 1')
    ax.scatter(*path2[0], c='red', marker='o', label='Start 2')
    
    # Plot collision point (assuming collision happens exactly halfway)
    collision_idx = len(path1) // 2
    ax.scatter(
        *path1[collision_idx], 
        c='purple', 
        marker='X', 
        s=80, 
        label='Collision'
    )
    
    # Set titles and labels
    ax.set_title("Realistic Drone Collision Paths")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # Example usage for testing
    # pos1 = np.array([0, 0, 0])
    # pos2 = np.array([10, -5, 5])
    # test_path1, test_path2 = compute_simple_collision_paths(pos1, pos2)
    # plot_3d_paths(test_path1, test_path2)
    pass