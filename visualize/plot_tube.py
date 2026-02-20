# -*- coding: utf-8 -*-
"""
Variance Tube Visualization Module

This module provides 3D plotting utilities to visualize trajectory 
uncertainty (variance) as a continuous, semi-transparent tube 
around a predicted 3D path.

Created on Wed Jul 16 21:58:32 2025
@author: Altinses
"""

from typing import Union, List, Any
import numpy as np


def plot_variance_tube(
    ax: Any, 
    path: Union[np.ndarray, List], 
    variance: Union[np.ndarray, List], 
    color: str = 'blue', 
    n_circle_points: int = 16, 
    alpha: float = 0.3
) -> None:
    """
    Plots a 3D tube around a path where the radius represents the variance.

    Constructs a local orthogonal coordinate frame at each point along the 
    path using the tangent vector, then extrudes a circle scaled by the 
    square root of the variance to create a surface mesh.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The Matplotlib 3D axes object to plot on.
    path : Union[np.ndarray, List]
        The 3D coordinates of the trajectory, expected shape (N, 3).
    variance : Union[np.ndarray, List]
        The variance (uncertainty) at each point, expected shape (N,) or (N, 1).
    color : str, optional
        Color of the tube surface. Defaults to 'blue'.
    n_circle_points : int, optional
        Resolution of the tube (number of vertices per cross-section). 
        Defaults to 16.
    alpha : float, optional
        Transparency of the tube surface (0.0 to 1.0). Defaults to 0.3.
    """
    path = np.asarray(path)
    variance = np.asarray(variance).flatten()
    n_points = len(path)

    # Prepare normalized circle template
    theta = np.linspace(0, 2 * np.pi, n_circle_points)
    
    # Template shape: (n_circle_points, 3)
    circle_template = np.stack(
        [np.cos(theta), np.sin(theta), np.zeros_like(theta)], 
        axis=1
    )

    # Allocate surface arrays (using lowercase to respect PEP 8)
    x_surface: List[np.ndarray] = []
    y_surface: List[np.ndarray] = []
    z_surface: List[np.ndarray] = []

    for i in range(n_points):
        # 1. Estimate tangent vector (direction of path) using finite differences
        if i == 0:
            tangent = path[i + 1] - path[i]
        elif i == n_points - 1:
            tangent = path[i] - path[i - 1]
        else:
            tangent = path[i + 1] - path[i - 1]

        # Normalize the tangent vector
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / tangent_norm

        # 2. Get orthogonal basis vectors
        # Avoid collinearity by picking a reference vector not aligned with the tangent
        if np.abs(tangent[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        ortho1 = np.cross(tangent, ref)
        ortho1 /= np.linalg.norm(ortho1)
        ortho2 = np.cross(tangent, ortho1)

        # 3. Calculate radius (Standard Deviation = sqrt(Variance))
        radius = np.sqrt(variance[i])

        # 4. Orient and scale the 3D circle at the current point
        circle = path[i] + radius * (
            circle_template[:, 0:1] * ortho1 + 
            circle_template[:, 1:2] * ortho2
        )
        
        x_surface.append(circle[:, 0])
        y_surface.append(circle[:, 1])
        z_surface.append(circle[:, 2])

    # Convert to arrays of shape (n_points, n_circle_points) for surface plotting
    x_surface = np.array(x_surface)
    y_surface = np.array(y_surface)
    z_surface = np.array(z_surface)

    # Plot the final extruded tube
    ax.plot_surface(
        x_surface, 
        y_surface, 
        z_surface, 
        color=color, 
        alpha=alpha, 
        linewidth=0, 
        antialiased=True
    )


if __name__ == '__main__':
    # Placeholder for script testing
    pass