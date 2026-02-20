# -*- coding: utf-8 -*-
"""
Model Predictive Control (MPC) Module

This module implements an MPC algorithm with active collision avoidance 
for navigating drone agents around obstacles and other dynamic agents.

Created on Tue Jul 15 17:06:09 2025
@author: Altinses
"""

from typing import List, Tuple, Any, Dict

import numpy as np
from scipy.optimize import minimize


def mpc_control(
    drone: Any, 
    other_drones: List[Any], 
    t: int, 
    N: int = 10,
    Q: np.ndarray = np.eye(3), 
    R_u: np.ndarray = np.eye(3)
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Model Predictive Control (MPC) algorithm for a drone with active 
    collision avoidance against other drones.

    This controller calculates an optimal sequence of control inputs 
    (accelerations) over a prediction horizon `N` to minimize a cost function. 

    Parameters
    ----------
    drone : Any
        The controlled drone object (must have `state`, `ref_path`, `A`, `B`, 
        and `safety_radius`).
    other_drones : List[Any]
        A list of other drone objects to avoid.
    t : int
        The current time step or index in the simulation.
    N : int, optional
        The prediction horizon (future time steps). Defaults to 10.
    Q : np.ndarray, optional
        Weighting matrix for the state tracking error cost (3x3). 
        Defaults to `np.eye(3)`.
    R_u : np.ndarray, optional
        Weighting matrix for the control input cost (3x3). 
        Defaults to `np.eye(3)`.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        - action: The optimal control input (acceleration [ax, ay, az]) for 
          the current time step `t`. Shape is (3,).
        - cost_component: Dictionary containing the breakdown of the final cost.
    """
    
    def cost_(u_flat: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calculates the total cost for a given sequence of control inputs.

        Parameters
        ----------
        u_flat : np.ndarray
            Flattened 1D array of control inputs for `N` steps. Shape (N*3,).

        Returns
        -------
        Tuple[float, Dict[str, float]]
            The total scalar cost and a dictionary of individual cost components.
        """
        # Reshape and clip control inputs for the horizon
        u = np.clip(u_flat.reshape((N, 3)), -2.5, 2.5)
        
        tracking_cost: float = 0.0
        control_cost: float = 0.0
        collision_cost: float = 0.0
        distance_cost: float = 0.0
        target_miss_cost: float = 0.0
        total_cost: float = 0.0
        
        # Start prediction from current drone state
        x_pred = np.copy(drone.state) 

        # Determine the effective horizon length to stay within bounds
        effective_horizon_len = max(0, min(N, len(drone.ref_path) - t - 1))
        
        for k in range(effective_horizon_len):
            # Predict next state based on current state and control input
            x_pred = drone.A @ x_pred + drone.B @ u[k]
            p_pred = np.clip(x_pred[:3], -2.5, 2.5)

            # 1. Standard tracking cost: Penalize deviation from reference path
            ref_pos_k = drone.ref_path[t + k]
            tracking = float(1 * ((p_pred - ref_pos_k).T @ Q @ (p_pred - ref_pos_k)))
            tracking_cost += tracking
            
            # 2. Control effort cost: Penalize large accelerations
            # control = 1 * (u[k].T @ R_u @ u[k])
            # control_cost += control

            # 3. Distance: Penalize if drones are close
            p_others = [other.state[:3] for other in other_drones]
            dist_between_drones = [np.linalg.norm(p_pred - p_other) for p_other in p_others]
            sum_of_radii = [drone.safety_radius + other.safety_radius for other in other_drones]
            
            # Use sum generator to avoid ZeroDivisionError risks if distance is exactly 0
            distance_cost += 1 * sum(1 / max(1e-6, dist) for dist in dist_between_drones)
            
            # 4. Strong collision cost: Penalize overlap of safety zones
            overlap = [
                safety_dist - drone_dist 
                for safety_dist, drone_dist in zip(sum_of_radii, dist_between_drones)
            ]
            collision_cost += sum(
                1000 * (1 + safety_overlap)**4 
                for safety_overlap in overlap if safety_overlap > 0
            )
                    
        # 5. Target miss penalty: Penalize deviation from the final target position
        # final_target_position = drone.ref_path[-1]
        # target_miss_cost = t * 0.5 * np.linalg.norm(x_pred[:3] - final_target_position)
        
        cost_component = {
            'tracking_cost': tracking_cost, 
            'control_cost': control_cost,
            'collision_cost': collision_cost,
            'target_miss_cost': target_miss_cost
        }
        
        total_cost = (
            control_cost + target_miss_cost + 
            tracking_cost + distance_cost + collision_cost
        )

        return float(total_cost), cost_component


    def distance_constraint(u_flat: np.ndarray) -> np.ndarray:
        """
        Calculates distance constraint violations across the prediction horizon.
        Values >= 0 denote valid states.
        """
        u = np.clip(u_flat.reshape((N, 3)), -2.5, 2.5)
        x_pred = np.copy(drone.state)
        
        effective_horizon_len = max(0, min(N, len(drone.ref_path) - t - 1))
        violation_list = []
        
        for k in range(effective_horizon_len):
            x_pred = drone.A @ x_pred + drone.B @ u[k]
            p_pred = np.clip(x_pred[:3], -2.5, 2.5)
        
            p_others = [other.state[:3] for other in other_drones]
            dist_between_drones = [np.linalg.norm(p_pred - p_other) for p_other in p_others]
            
            # Note: The multiplier 1.1 provides a constraint buffer
            sum_of_radii = [drone.safety_radius + 1.1 * other.safety_radius for other in other_drones]
            
            # Constraint: distance - safety distance >= 0
            overlap = [
                drone_dist - safety_dist 
                for safety_dist, drone_dist in zip(sum_of_radii, dist_between_drones)
            ]
            violation_list.append(np.array(overlap))
        
        if not violation_list:
            return np.array([0.0])
            
        return np.array(violation_list).flatten()


    def cost_fn(u_flat: np.ndarray) -> float:
        """Wrapper to extract just the scalar cost for the SciPy optimizer."""
        return cost_(u_flat)[0]


    # Initial guess for control inputs based on average velocity to target
    displacement = drone.ref_path[-1] - drone.ref_path[0]
    average_velocity = displacement / N
    constrained_velocity = np.clip(average_velocity, -2.5, 2.5)
    u0 = np.tile(constrained_velocity, (N, 1))

    # Define bounds for control inputs (accelerations)
    bounds: List[Tuple[float, float]] = [(-2.5, 2.5) for _ in range(N * 3)]
    
    # Set up the constraints for the solver
    constraints = [{'type': 'ineq', 'fun': distance_constraint}]
    
    # Perform the optimization using SciPy's SLSQP algorithm
    res = minimize(
        cost_fn, 
        u0.flatten(), 
        bounds=bounds, 
        method='SLSQP', 
        constraints=constraints
    )

    # Extract the first optimal control input (acceleration) for the current step
    # We temporarily set N=1 scope for cost recalculation to match original logic
    _N_original = N
    action = res.x[:3].reshape(3,)
    _, cost_component = cost_(res.x)
    
    return action, cost_component


if __name__ == "__main__":
    pass