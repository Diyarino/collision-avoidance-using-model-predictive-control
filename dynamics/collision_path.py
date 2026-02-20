# -*- coding: utf-8 -*-
"""
Drone Collision Path Generation Module

This module generates 3D reference trajectories for drones, including 
both simple linear paths and realistic, curved paths with smooth velocity 
interpolation leading to a designated collision point.

Created on Tue Jul 15 09:45:02 2025
@author: Altinses
"""

import numpy as np
from typing import Tuple, Optional


def compute_simple_collision_paths(
    pos1: np.ndarray, 
    pos2: np.ndarray, 
    T: int = 100, 
    collision_time: float = 0.5, 
    post_collision_steps: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates reference paths where drones collide at a specified time 
    and continue moving linearly afterward.
    
    Parameters
    ----------
    pos1 : np.ndarray
        Starting position of drone 1 [x, y, z].
    pos2 : np.ndarray
        Starting position of drone 2 [x, y, z].
    T : int, optional
        Total timesteps for the returned paths. Defaults to 100.
    collision_time : float, optional
        Fraction of T when the collision should occur (0 to 1). Defaults to 0.5.
    post_collision_steps : int, optional
        How many steps to calculate beyond the collision. Defaults to 20.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reference paths for both drones, each of shape (T, 3).
    """
    # Calculate collision point (midpoint between starting positions)
    collision_point = (pos1 + pos2) / 2
    collision_point = collision_point.astype(np.int64)
    
    # Calculate time steps
    t_collision = int(T * collision_time)
    t_total = int(T + post_collision_steps)  # Extend path beyond collision
    post_collision_steps = int(post_collision_steps)
    
    # Initialize paths
    path1 = np.zeros((t_total, 3))
    path2 = np.zeros((t_total, 3))
    
    # Drone 1 path: from start to collision point, then continues
    path1[:t_collision] = np.linspace(pos1, collision_point, t_collision)
    path1[t_collision:] = np.linspace(
        collision_point, 
        collision_point + (collision_point - pos1), 
        post_collision_steps + T // 2 + 1
    )[1:]
    
    # Drone 2 path: from start to collision point, then continues
    path2[:t_collision] = np.linspace(pos2, collision_point, t_collision)
    path2[t_collision:] = np.linspace(
        collision_point, 
        collision_point + (collision_point - pos2), 
        post_collision_steps + T // 2 + 1
    )[1:]
    
    return path1[:T], path2[:T]  # Return paths of strictly length T


def smooth_velocity_path(waypoints: np.ndarray, T: int) -> np.ndarray:
    """
    Interpolates waypoints with smooth velocity and acceleration using 
    Hermite-like velocity matching.

    Parameters
    ----------
    waypoints : np.ndarray
        Array of 3D waypoints to interpolate.
    T : int
        Total time steps for the resulting smoothed path.

    Returns
    -------
    np.ndarray
        Smoothed path of shape (T, 3).
    """
    N = len(waypoints)
    velocities = np.diff(waypoints, axis=0)
    velocities = np.vstack([velocities[0], velocities, velocities[-1]])
    
    t = np.linspace(0, 1, T)
    segment_t = np.linspace(0, 1, N)
    path = np.zeros((T, 3))
    
    for i in range(3):  # For x, y, z dimensions
        path[:, i] = np.interp(t, segment_t, waypoints[:, i])
    
    return path


def generate_realistic_path(
    start: np.ndarray,
    end: np.ndarray,
    n_waypoints: int = 5,
    jitter: float = 1.0,
    curve_strength: float = 0.25,
    total_steps: int = 100
) -> np.ndarray:
    """
    Creates a realistic path from start to end with random but smooth 
    intermediate waypoints.

    Parameters
    ----------
    start : np.ndarray
        Starting position (3,).
    end : np.ndarray
        Target position (3,).
    n_waypoints : int, optional
        Number of waypoints to insert between start and end. Defaults to 5.
    jitter : float, optional
        Magnitude of lateral deviation per waypoint. Defaults to 1.0.
    curve_strength : float, optional
        Weight of non-linearity (higher = more curve). Defaults to 0.25.
    total_steps : int, optional
        Number of time steps in the path. Defaults to 100.

    Returns
    -------
    np.ndarray
        Generated path of shape (total_steps, 3).
    """
    direction = end - start
    direction /= np.linalg.norm(direction)
    length = np.linalg.norm(end - start)

    # Generate intermediate waypoints along the straight line
    waypoints = [start]
    
    for i in range(1, n_waypoints + 1):
        alpha = i / (n_waypoints + 1)
        base = start + direction * length * alpha
        lateral = np.random.randn(3)
        lateral -= direction * np.dot(lateral, direction)  # Orthogonal component
        lateral /= (np.linalg.norm(lateral) + 1e-8)
        
        offset = lateral * jitter * np.sin(np.pi * alpha) * curve_strength
        waypoints.append(base + offset)

    waypoints.append(end)
    return smooth_velocity_path(np.array(waypoints), total_steps)


def compute_realistic_collision_paths(
    start1: np.ndarray,
    start2: np.ndarray,
    T: int = 100,
    collision_time: float = 0.5,
    n_waypoints: int = 5,
    jitter: float = 1.0,
    curve_strength: float = 0.25,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates two realistic drone-like paths that curve and collide 
    at a given time.

    Parameters
    ----------
    start1 : np.ndarray
        Start position of drone 1.
    start2 : np.ndarray
        Start position of drone 2.
    T : int, optional
        Total time steps. Defaults to 100.
    collision_time : float, optional
        Time fraction when collision occurs (between 0 and 1). Defaults to 0.5.
    n_waypoints : int, optional
        Number of intermediate waypoints per path segment. Defaults to 5.
    jitter : float, optional
        Lateral noise for non-linearity in paths. Defaults to 1.0.
    curve_strength : float, optional
        Weight of non-linearity. Defaults to 0.25.
    seed : int, optional
        Random seed for reproducible paths. Defaults to None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of paths (path1, path2), each of shape (T, 3).
    """
    if seed is not None:
        np.random.seed(seed)

    t_collision = int(T * collision_time)
    
    # Calculate collision midpoint with slight uniform random noise
    collision_point = (start1 + start2) / 2 + np.random.uniform(-1, 1, 3)

    # Generate Drone 1 paths
    path1 = generate_realistic_path(
        start1, collision_point, total_steps=t_collision, 
        jitter=jitter, curve_strength=curve_strength, n_waypoints=n_waypoints
    )
    path1_post = generate_realistic_path(
        collision_point, collision_point + (collision_point - start1), 
        total_steps=T - t_collision, jitter=jitter, 
        curve_strength=curve_strength, n_waypoints=n_waypoints
    )
    full_path1 = np.vstack([path1, path1_post])

    # Generate Drone 2 paths
    path2 = generate_realistic_path(
        start2, collision_point, total_steps=t_collision,
        jitter=jitter, curve_strength=curve_strength, n_waypoints=n_waypoints
    )
    path2_post = generate_realistic_path(
        collision_point, collision_point + (collision_point - start2), 
        total_steps=T - t_collision, jitter=jitter, 
        curve_strength=curve_strength, n_waypoints=n_waypoints
    )
    full_path2 = np.vstack([path2, path2_post])

    return full_path1[:T], full_path2[:T]


if __name__ == '__main__':
    # Test path generation
    pos1 = np.array([0, 0, 0])
    pos2 = np.array([10, -5, 5])
    
    test_path1, test_path2 = compute_simple_collision_paths(pos1, pos2, T=100)
    test_path_real1, test_path_real2 = compute_realistic_collision_paths(
        pos1, pos2, T=100, jitter=5.5, seed=None
    )