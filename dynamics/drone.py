# -*- coding: utf-8 -*-
"""
Drone Agent Dynamics Module

This module defines the Drone and Drone_restricted classes, modeling 
their kinematics using a linear discrete-time state-space representation.

Created on Tue Jul 15 09:42:16 2025
@author: Diyar Altinses, M.Sc.
"""

import numpy as np


class Drone:
    """
    A drone agent with kinematics, a reference path, and a safety zone.
    
    The state is represented as a 6-element vector containing 3D position 
    and 3D velocity: [x, y, z, vx, vy, vz].
    """

    def __init__(
        self, 
        drone_id: int, 
        initial_pos: np.ndarray, 
        initial_vel: np.ndarray, 
        dt: float = 0.1, 
        safety_radius: float = 1.0
    ):
        """
        Initializes the drone state and state-space matrices.

        Parameters
        ----------
        drone_id : int
            Drone identifier.
        initial_pos : np.ndarray
            Initial position [x, y, z] of shape (3,).
        initial_vel : np.ndarray
            Initial velocity [vx, vy, vz] of shape (3,).
        dt : float, optional
            Discrete time step for the simulation. Defaults to 0.1.
        safety_radius : float, optional
            Radius of the drone's collision safety zone. Defaults to 1.0.
        """
        self.id = drone_id
        self.state = np.hstack([initial_pos, initial_vel])
        self.dt = dt
        
        # State transition matrix (A)
        self.A = np.block([
            [np.eye(3), self.dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        
        # Input matrix (B)
        self.B = np.block([
            [0.5 * self.dt**2 * np.eye(3)],
            [self.dt * np.eye(3)]
        ]).reshape(6, 3)
        
        self.ref_path = None
        self.actual_path = []
        self.safety_radius = safety_radius
        self.var_safety_radius = False

    def update_state(self, u: np.ndarray) -> None:
        """
        Updates the drone's kinematic state based on the control input 
        and logs the new position.

        Parameters
        ----------
        u : np.ndarray
            Control input vector (e.g., accelerations) of shape (3,).
        """
        self.state = self.A @ self.state + self.B @ u
        self.actual_path.append(self.state[:3].copy())


class Drone_restricted:
    """
    A restricted drone agent with kinematics, bounded spatial coordinates, 
    and velocity limits.
    """

    def __init__(
        self, 
        drone_id: int, 
        initial_pos: np.ndarray, 
        initial_vel: np.ndarray, 
        dt: float = 0.1, 
        safety_radius: float = 1.0
    ):
        """
        Initializes the restricted drone state and state-space matrices.

        Parameters
        ----------
        drone_id : int
            Drone identifier.
        initial_pos : np.ndarray
            Initial position [x, y, z] of shape (3,).
        initial_vel : np.ndarray
            Initial velocity [vx, vy, vz] of shape (3,).
        dt : float, optional
            Discrete time step for the simulation. Defaults to 0.1.
        safety_radius : float, optional
            Radius of the drone's collision safety zone. Defaults to 1.0.
        """
        self.id = drone_id
        self.state = np.hstack([initial_pos, initial_vel])
        self.dt = dt
        
        # State transition matrix (A)
        self.A = np.block([
            [np.eye(3), self.dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        
        # Input matrix (B)
        self.B = np.block([
            [0.5 * self.dt**2 * np.eye(3)],
            [self.dt * np.eye(3)]
        ]).reshape(6, 3)
        
        self.ref_path = None
        self.actual_path = []
        self.safety_radius = safety_radius
        
        # Track previous state
        self.old_state = self.state

    def update_state(self, u: np.ndarray) -> bool:
        """
        Updates the drone's kinematic state, applies spatial and velocity 
        clipping, and logs the new position.

        Parameters
        ----------
        u : np.ndarray
            Control input vector (e.g., accelerations) of shape (3,).

        Returns
        -------
        bool
            Returns True upon successful update.
        """
        self.old_state = self.state
        
        # If control input is exactly zero, hold the previous state entirely
        if (u == np.zeros(3)).all():
            self.state = self.old_state
        else:
            self.state = self.A @ self.state + self.B @ u
        
        # Clip positions and velocities strictly within physical bounds
        self.state[:3] = np.clip(self.state[:3], -2.5, 2.5)
        self.state[3:] = np.clip(self.state[3:], -5.0, 5.0)
        
        self.actual_path.append(self.state[:3].copy())
            
        return True


if __name__ == '__main__':
    # Test instantiation (Fixed dimension inputs to match 3D logic)
    drone1 = Drone(
        drone_id=1, 
        initial_pos=np.zeros(3), 
        initial_vel=np.zeros(3)
    )
    
    # Test a simple update step
    control_input = np.array([0.1, 0.0, -0.05])
    drone1.update_state(control_input)
    print(f"Drone 1 updated state: \n{drone1.state}")