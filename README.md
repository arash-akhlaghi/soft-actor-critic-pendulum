# Optimized Soft Actor-Critic (SAC) for Pendulum-v1 🏎️🚀

This repository contains a highly optimized and stable implementation of the **Soft Actor-Critic (SAC)** algorithm, tailored to solve the continuous control challenge of the `Pendulum-v1` environment from Gymnasium. 

---

## 🌟 Key Features & Enhancements

We have improved the vanilla SAC architecture by introducing three advanced reinforcement learning techniques to maximize stability and speed up convergence:

*   **🔄 Automated Temperature Tuning (Dynamic Alpha):** Instead of keeping the entropy coefficient ($\alpha$) constant, the agent automatically learns and adjusts the temperature to maintain a target entropy, balancing exploration and exploitation dynamically.
*   **🛠️ Reward Engineering:** Added a specialized penalty for high angular velocity ($\dot{\theta}^2$) when the pendulum is close to the upright position ($cos(\theta) > 0.95$). This effectively eliminates micro-vibrations at the top.
*   **📐 Architecture Stabilization (LayerNorm & Orthogonal Init):** 
    *   Implemented **Orthogonal Initialization** across all neural networks to prevent vanishing/exploding gradients.
    *   Integrated **Layer Normalization** before activations (`ReLU`) to stabilize data flow throughout training.

---

## 🏛️ Neural Network Architecture

The agent utilizes a total of 5 deep neural networks:
1.  **Actor Network 🎭:** Predicts the mean (`mu`) and standard deviation (`sigma`) of actions using a squashed Gaussian policy.
2.  **Twin Critic Networks (Q1 & Q2) ⚖️:** Evaluates state-action values to mitigate overestimation bias.
3.  **Value & Target Value Networks 🧠:** Estimates state values for stable temporal-difference updates.

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
