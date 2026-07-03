# Advanced Soft Actor-Critic (SAC-v2) with Prioritized Experience Replay for Pendulum-v1 🏎️🚀

This repository contains a state-of-the-art, highly optimized implementation of the **Soft Actor-Critic (SAC-v2)** algorithm, integrated with **Prioritized Experience Replay (PER)**. It is designed to perfectly solve the continuous control challenge of the `Pendulum-v1` environment from Gymnasium.

---

## 🌟 Key Features & Advanced Enhancements

We have upgraded the standard SAC architecture with several advanced reinforcement learning techniques to maximize sample efficiency, stability, and convergence speed:

*   **🏛️ SAC-v2 Architecture (Value Network Removed):** Upgraded to the modern version of SAC by eliminating the explicit Value Network. Target Q-values are computed directly using the twin target critics, reducing computational overhead and memory footprint.
*   **📊 Prioritized Experience Replay (PER):** Instead of uniform random sampling, transitions are sampled based on their **TD-error** magnitude ($|\delta|$). This forces the agent to focus on and learn from the most informative and challenging experiences.
*   **🔄 Automated Temperature Tuning (Dynamic Alpha):** The entropy coefficient ($\alpha$) is automatically optimized online to match a target entropy, balancing exploration and exploitation dynamically throughout the training process.
*   **🛠️ Reward Engineering:** Features a specialized penalty for high angular velocity ($\dot{\theta}^2$) when the pendulum is near the upright position ($cos(\theta) > 0.95$), effectively eliminating micro-vibrations at the top.

---

## 🏛️ Neural Network Architecture

The agent utilizes 3 main active networks and 2 target networks:
1.  **Actor Network 🎭:** Predicts the mean (`mu`) and standard deviation (`sigma`) of actions using a squashed Gaussian policy.
2.  **Twin Critic Networks (Q1 & Q2) ⚖️:** Evaluates state-action values to mitigate overestimation bias.
3.  **Target Twin Critic Networks 🛡️:** Used for stable temporal-difference target updates (updated via soft-updates).

---

## 🎮 Performance & Training Phases

The script runs in a highly efficient two-phase execution loop:
*   **Phase 1: Fast Training (No GUI) ⚡:** Runs for 250 episodes using a headless environment configuration to maximize CPU/GPU training throughput.
*   **Phase 2: Visual Testing (With GUI) 🕹️:** Automatically re-creates the environment with `render_mode='human'` to display 5 evaluation episodes of the fully trained agent using Native Pygame rendering.

---

## ⚙️ Requirements & Installation

Make sure you have Python 3.8+ installed. You can install the required packages using pip:

```bash
pip install gymnasium[classic_control] torch numpy
