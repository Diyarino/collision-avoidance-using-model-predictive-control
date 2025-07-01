"""
3D Drone Collision Avoidance with MPC and Safety Zones

Key Features:
1. Automatically computes collision paths for drones.
2. Visualizes safety zones as spheres.
3. Clearly shows MPC-corrected trajectories.
4. Extended simulation to ensure collision avoidance.

Author: Diyar Altinses, M.Sc.
Date: 2023-11-20
"""

#%% imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from distutils.spawn import find_executable

#%% configurations

def configure_plt(check_latex = True):
        """
        Set Font sizes for plots.
    
        Parameters
        ----------
        check_latex : bool, optional
            Use LaTex-mode (if available). The default is True.
    
        Returns
        -------
        None.
    
        """
        
        if check_latex:
            
            if find_executable('latex'):
                plt.rc('text', usetex=True)
            else:
                plt.rc('text', usetex=False)
        plt.rc('font',family='Times New Roman')
        plt.rcParams.update({'figure.max_open_warning': 0})
        
        small_size = 13
        small_medium = 14
        medium_size = 16
        big_medium = 18
        big_size = 20
        
        plt.rc('font', size = small_size)          # controls default text sizes
        plt.rc('axes', titlesize = big_medium)     # fontsize of the axes title
        plt.rc('axes', labelsize = medium_size)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize = small_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize = small_size)    # fontsize of the tick labels
        plt.rc('legend', fontsize = small_medium)    # legend fontsize
        plt.rc('figure', titlesize = big_size)  # fontsize of the figure title
        
        plt.rc('grid', c='0.5', ls='-', lw=0.5)
        plt.grid(True)
        plt.tight_layout()
        plt.close()
    
# %%

class Drone:
    """
    A drone agent with dynamics, reference path, and safety zone.
    
    Args:
        id (int): Drone identifier.
        initial_pos (np.ndarray): Initial position [x, y, z].
        initial_vel (np.ndarray): Initial velocity [vx, vy, vz].
        dt (float): Time step.
    """
    def __init__(self, id, initial_pos, initial_vel, dt=0.1, safety_radius = 2.0):
        
        self.id = id
        self.state = np.hstack([initial_pos, initial_vel])
        self.dt = dt
        
        self.A = np.block([
            [np.eye(3), self.dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        
        self.B = np.block([
            [0.5 * self.dt**2 * np.eye(3)],
            [self.dt * np.eye(3)]
        ]).reshape(6, 3)
        
        self.ref_path = None
        self.actual_path = []
        self.safety_radius = safety_radius  # Safety zone radius

    def update_state(self, u):
        """Update state and log position."""
        self.state = self.A @ self.state + self.B @ u
        self.actual_path.append(self.state[:3].copy())

# %%

def compute_collision_paths(pos1, pos2, T=100, collision_time=0.5, post_collision_steps=20):
    """
    Generates reference paths where drones collide at a specified time and continue moving afterward.
    
    Args:
        pos1 (np.array): Starting position of drone 1 [x,y,z]
        pos2 (np.array): Starting position of drone 2 [x,y,z]
        T (int): Total timesteps for the paths
        collision_time (float): Fraction of T when collision should occur (0-1)
        post_collision_steps (int): How many steps to continue after collision
        
    Returns:
        tuple: (path1, path2) reference paths for both drones
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
    path1[t_collision:] = np.linspace(collision_point, 
                                     collision_point + (collision_point - pos1), 
                                     post_collision_steps + T//2 + 1)[1:]
    
    # Drone 2 path: from start to collision point, then continues
    path2[:t_collision] = np.linspace(pos2, collision_point, t_collision)
    path2[t_collision:] = np.linspace(collision_point, 
                                     collision_point + (collision_point - pos2), 
                                     post_collision_steps + T//2 + 1)[1:]
    
    return path1[:T], path2[:T]  # Return paths of length T

# %%

def mpc_control(drone, other_drone, t, N=10, Q=np.eye(3), R_u=0.1*np.eye(3)):
    """
    MPC controller with collision avoidance.
    
    Args:
        drone (Drone): Controlled drone.
        other_drone (Drone): Other drone to avoid.
        N (int): Prediction horizon.
        Q (np.ndarray): State cost matrix.
        R_u (np.ndarray): Control cost matrix.
    
    Returns:
        np.ndarray: Optimal control input (3x1 acceleration).
    """
    def cost_fn(u_flat):
        
        u = np.clip(u_flat.reshape((N, 3)), -5, 5)  # Pre-clip
        u = u_flat.reshape((N, 3))
        cost = 0
        x_pred = np.copy(drone.state)
        current_horizon = min(N, len(drone.ref_path) - t - 1)
        
        for k in range(current_horizon):
            x_pred = drone.A @ x_pred + drone.B @ u[k]
            p_pred = x_pred[:3]
            
            # Standard tracking cost
            cost += (p_pred - drone.ref_path[t + k]).T @ Q @ (p_pred - drone.ref_path[t + k])
            cost += u[k].T @ R_u @ u[k]
            
            # Safety violation (hard penalty)
            p_other = other_drone.state[:3] + k * other_drone.dt * other_drone.state[3:]
            dist = np.linalg.norm(p_pred - p_other)
            cost += 1e2 * max(0, drone.safety_radius - dist)**4  # Quartic penalty
            
        # Target miss penalty (hard penalty on final position)
        target_idx = min(t + N, len(drone.ref_path) - 1)  # Clamp to last index
        cost += 1e4 * np.linalg.norm(x_pred[:3] - drone.ref_path[target_idx])**4
        return cost

    u0 = np.zeros((N, 3))
    bounds = [(-5, 5) for _ in range(N*3)]
    res = minimize(cost_fn, u0.flatten(), bounds=bounds, method='SLSQP')
    return res.x[:3].reshape(3,)

# %%

def plot_safety_zones(ax, drone1, drone2):
    """Draw safety zones as 3D spheres."""
    for drone, color in zip([drone1, drone2], ['b', 'r']):
        center = drone.state[:3]
        radius = drone.safety_radius
        # Create a wireframe sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color=color, alpha=0.2)

# %%

def simulate(T=100):
    """Run the simulation with collision setup and visualization."""
    # Initialize drones with collision paths
    pos1 = np.array([0,0,0])
    pos2 = np.array([10,-4,0])
    
    path1, path2 = compute_collision_paths(pos1, pos2, T, post_collision_steps=10)
    drone1 = Drone(1, path1[0], (path1[1] - path1[0]) / 0.3)
    drone2 = Drone(2, path2[0], (path2[1] - path2[0]) / 0.3)
    drone1.ref_path, drone2.ref_path = path1, path2

    # Colors and labels
    colors = ['b', 'r']
    labels = ['Drone 1', 'Drone 2']

    for t in range(T):
        # Compute control inputs
        u1 = mpc_control(drone1, drone2, t)
        u2 = mpc_control(drone2, drone1, t)
        drone1.update_state(u1)
        drone2.update_state(u2)
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X [m]', fontsize=12)
        ax.set_ylabel('Y [m]', fontsize=12)
        ax.set_zlabel('Z [m]', fontsize=12)
        ax.set_title('Drone Collision Avoidance with MPC', fontsize=14)
        ax.set_xlim(-2, 12)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        ax.grid(True)

        # Clear and replot
        ax.clear()
        ax.set_xlim(-2, 12)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.grid(True)

        # Plot reference paths (collision course)
        ax.plot(path1[:, 0], path1[:, 1], path1[:, 2], '--', color=colors[0], alpha=0.3, label='Ref Path 1')
        ax.plot(path2[:, 0], path2[:, 1], path2[:, 2], '--', color=colors[1], alpha=0.3, label='Ref Path 2')

        # Plot actual paths (MPC-corrected)
        if len(drone1.actual_path) > 1:
            path_actual1 = np.array(drone1.actual_path)
            ax.plot(path_actual1[:, 0], path_actual1[:, 1], path_actual1[:, 2], 
                    '-', color=colors[0], linewidth=2, label='Actual Path 1')
        if len(drone2.actual_path) > 1:
            path_actual2 = np.array(drone2.actual_path)
            ax.plot(path_actual2[:, 0], path_actual2[:, 1], path_actual2[:, 2], 
                    '-', color=colors[1], linewidth=2, label='Actual Path 2')

        # Plot current positions and safety zones
        ax.scatter(drone1.state[0], drone1.state[1], drone1.state[2], 
                  color=colors[0], s=100, marker='o', label=labels[0])
        ax.scatter(drone2.state[0], drone2.state[1], drone2.state[2], 
                  color=colors[1], s=100, marker='o', label=labels[1])
        plot_safety_zones(ax, drone1, drone2)

        # Highlight near-collision
        dist = np.linalg.norm(drone1.state[:3] - drone2.state[:3])
        if dist < drone1.safety_radius * 1.5:
            ax.text(5, 0, 0, "SAFETY ZONE VIOLATED!", color='red', fontsize=12)

        ax.legend(loc='upper right')
        # plt.savefig('figures//sample'+str(t)+'.png', dpi = 150, bbox_inches='tight')

        plt.show()
    
    # %%

if __name__ == "__main__":
    configure_plt()
    simulate(T=50)  # Extended duration to observe avoidance
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    