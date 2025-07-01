# Collision Avoidance for Drones in 3D using Model Predictive Control (MPC)

This repository provides an implementation of a **Model Predictive Control (MPC)** framework for real-time **collision avoidance** in a 3D environment for multiple drones. It ensures safe navigation while achieving trajectory tracking objectives.

- 3D drone dynamics modeling
- Real-time trajectory tracking
- Predictive collision avoidance for multiple UAVs
- Configurable constraints (velocity, acceleration, safety distance)
- Modular and simulation-ready design

## 📽️ Demo

<p align="center">
  <img src="animation_collision.gif" width="600" height="300" alt="til">
</p>

## 🧠 Algorithm Summary

This system uses MPC to:
- Predict future states of each drone over a finite time horizon.
- Optimize control inputs to minimize deviation from a desired trajectory.
- Introduce soft or hard constraints to avoid inter-drone collisions.
- Solve a constrained optimization problem at each time step.


## 📦 Prerequisites

Before you begin, ensure your environment meets the following requirements:

* **Python** ≥ 3.6
* **PyTorch** ≥ 1.0 (CUDA support recommended for faster training)
* **matplotlib, numpy, scipy**

We also recommend using a virtual environment (e.g., `venv` or `conda`) to avoid package conflicts.

## 🚀 Getting Started
```bash
git clone https://github.com/yourusername/collision-avoidance-mpc.git
cd collision-avoidance-mpc
pip install -r requirements.txt
```

## 📌 Citation
If you use this code or build upon our work, please cite our paper:


```bibtex
@article{altinses2025XXX,
  title={Empty},
  author={Altinses, Diyar and Andreas, Schwung},
  journal={Empty},
  volume={XX},
  number={XX},
  pages={XX--XX},
  year={XXXX},
  publisher={IEEE}
}
```


## 📚 References 

This project builds on concepts from multimodal representation learning, attention-based fusion, and anomaly detection in industrial systems. Below are selected related works and projects that inspired or complement this research:

<a id="1">[1]</a> Altinses, D., & Schwung, A. (2023, October). Multimodal Synthetic Dataset Balancing: A Framework for Realistic and Balanced Training Data Generation in Industrial Settings. In IECON 2023-49th Annual Conference of the IEEE Industrial Electronics Society (pp. 1-7). IEEE.

<a id="2">[2]</a> Altinses, D., & Schwung, A. (2025). Performance benchmarking of multimodal data-driven approaches in industrial settings. Machine Learning with Applications, 100691.

<a id="3">[3]</a> Altinses, D., & Schwung, A. (2023, October). Deep Multimodal Fusion with Corrupted Spatio-Temporal Data Using Fuzzy Regularization. In IECON 2023-49th Annual Conference of the IEEE Industrial Electronics Society (pp. 1-7). IEEE.





