# -*- coding: utf-8 -*-
"""
Drone Safety Zone Visualization Module

This module provides utilities to draw the safety zones of multiple drones 
as 3D wireframe spheres on a given Matplotlib axis.

Created on Tue Jul 15 17:00:55 2025
@author: Altinses
"""

from typing import List, Optional, Any

import numpy as np


def plot_safety_zones(
    ax: Any, 
    drones: List[Any], 
    colors: Optional[List[str]] = None, 
    alpha: float = 0.2, 
    resolution: Optional[int] = None
) -> None:
    """
    Draws the safety zones of multiple drones as 3D wireframe spheres.

    This function iterates through a list of drone objects, extracts their 
    current position and safety radius, and plots a wireframe sphere 
    representing their collision avoidance safety zone.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3D axes object on which to draw the safety zones.
    drones : List[Any]
        A list of Drone objects (must possess `state` and `safety_radius` attributes).
    colors : List[str], optional
        A list of Matplotlib color strings to use for each drone's safety zone.
        If None, a default sequence is cycled through. Defaults to None.
    alpha : float, optional
        The transparency level (0.0 to 1.0) for the wireframe spheres. Defaults to 0.2.
    resolution : int, optional
        Determines the resolution of the spherical mesh (number of azimuthal points). 
        If None, default values of 20 (azimuthal) and 10 (polar) are used. 
        Defaults to None.

    Returns
    -------
    None
        The function directly modifies the provided `ax` object.
    """
    # Default colors if none are provided
    default_colors = [
        "royalblue", "red", "green", "black", "mediumslateblue", 
        "orange", "darkviolet", "darkgoldenrod", "silver", "deepskyblue"
    ]
    
    # Color assignment and cycling logic
    if colors is None:
        colors_to_use = [
            default_colors[i % len(default_colors)] 
            for i in range(len(drones))
        ]
    else:
        if len(colors) < len(drones):
            print("Warning: Not enough colors provided for all drones. Cycling through.")
        colors_to_use = [
            colors[i % len(colors)] 
            for i in range(len(drones))
        ]

    # Set sphere resolution using complex numbers for np.mgrid steps
    if resolution is None:
        u_res = 20j  # 20 points
        v_res = 10j  # 10 points
    else:
        u_res = complex(0, resolution)
        v_res = complex(0, resolution // 2)  # Half resolution for v typically looks good

    # Generate mesh for a unit sphere once to save computation time
    u, v = np.mgrid[0:2*np.pi:u_res, 0:np.pi:v_res]
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    sin_v = np.sin(v)
    cos_v = np.cos(v)

    # Plot wireframe spheres for each drone
    for i, drone in enumerate(drones):
        center = drone.state[:3]
        radius = drone.safety_radius
        color = colors_to_use[i]

        # Calculate sphere coordinates by scaling and shifting the unit sphere mesh
        x = center[0] + radius * cos_u * sin_v
        y = center[1] + radius * sin_u * sin_v
        z = center[2] + radius * cos_v

        ax.plot_wireframe(x, y, z, color=color, alpha=alpha)


if __name__ == "__main__":
    # Placeholder for script testing
    pass