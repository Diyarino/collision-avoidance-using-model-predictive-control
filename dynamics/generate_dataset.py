# -*- coding: utf-8 -*-
"""
Collision Dataset Generation Module

This module generates large synthetic datasets of colliding 3D drone 
trajectories with randomized parameters for training machine learning models.

Created on Tue Jul 15 09:56:17 2025
@author: Diyar Altinses, M.Sc.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Union

# Assuming these are the local modules we just refactored
from collision_path import compute_realistic_collision_paths
from path_plot import plot_3d_paths


def generate_collision_dataset(
    n: int = 100,
    T: int = 100,
    space_scale: float = 10.0,
    jitter_range: Tuple[float, float] = (0.5, 2.0),
    curve_range: Tuple[float, float] = (0.2, 1.0),
    seed: Optional[int] = None,
    return_params: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
    """
    Generates a dataset of `n` realistic, colliding drone path pairs.

    Parameters
    ----------
    n : int, optional
        Number of path pairs to generate. Defaults to 100.
    T : int, optional
        Number of time steps per path. Defaults to 100.
    space_scale : float, optional
        Spatial scale (start points spawn within [-space_scale, space_scale]). 
        Defaults to 10.0.
    jitter_range : Tuple[float, float], optional
        Min and max bounds for lateral deviation jitter. Defaults to (0.5, 2.0).
    curve_range : Tuple[float, float], optional
        Min and max bounds for curvature strength. Defaults to (0.2, 1.0).
    seed : int, optional
        Random seed for reproducibility. Defaults to None.
    return_params : bool, optional
        Whether to return the dictionary of jitter and curve values used. 
        Defaults to False.

    Returns
    -------
    Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]
        - paths1: Paths of drone 1, shape (n, T, 3)
        - paths2: Paths of drone 2, shape (n, T, 3)
        - params (optional): Dictionary containing arrays of 'jitter' and 'curve_strength'
    """
    if seed is not None:
        np.random.seed(seed)

    paths1 = []
    paths2 = []
    jitters = []
    curves = []

    for _ in range(n):
        # Generate random start points within a cubic volume
        start1 = np.random.uniform(-space_scale, space_scale, 3)
        
        # Ensure drones don't spawn too close to each other
        offset = np.random.uniform(2.0, 5.0, 3)  
        start2 = start1 + offset * np.random.choice([-1, 1], 3)

        # Randomize path parameters for this specific pair
        jitter = np.random.uniform(*jitter_range)
        curve = np.random.uniform(*curve_range)
        collision_time = np.random.uniform(0.3, 0.7)
        
        # Extract a standard integer for waypoints
        n_waypoints = int(np.random.choice([5, 6, 7, 8, 9, 10]))
        
        # Generate the paths (Note: curve is currently not passed to preserve original functionality)
        p1, p2 = compute_realistic_collision_paths(
            start1, 
            start2, 
            T=T, 
            jitter=jitter, 
            collision_time=collision_time,
            n_waypoints=n_waypoints
        )
        
        paths1.append(p1)
        paths2.append(p2)
        jitters.append(jitter)
        curves.append(curve)

    # Stack lists into unified numpy arrays
    paths1_arr = np.stack(paths1)
    paths2_arr = np.stack(paths2)

    if return_params:
        params_dict = {
            "jitter": np.array(jitters), 
            "curve_strength": np.array(curves)
        }
        return paths1_arr, paths2_arr, params_dict
        
    return paths1_arr, paths2_arr


if __name__ == '__main__':
    # Test dataset generation
    print("Generating dataset...")
    test_paths1, test_paths2, test_params = generate_collision_dataset(
        n=10000,
        T=100,
        space_scale=20.0,
        jitter_range=(0.1, 1.5),
        curve_range=(0.1, 0.8),
        seed=None,
        return_params=True
    )
    
    # Save the generated dataset and parameters as PyTorch tensors
    print("Saving tensors...")
    # torch.save(test_params, 'parameters.pt')
    # torch.save(test_paths1, 'path1.pt')
    # torch.save(test_paths2, 'path2.pt')
    
    # Plot a random trajectory pair to verify
    idx = np.random.randint(0, len(test_paths1))
    plot_3d_paths(test_paths1[idx], test_paths2[idx])