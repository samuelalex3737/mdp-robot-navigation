# 🤖 Wall-Following Robot Navigation using MDP

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 📋 Project Overview

This project implements a **Markov Decision Process (MDP)** for autonomous robot navigation using the Wall-Following Robot dataset from Kaggle. The robot learns optimal wall-following behavior through **Value Iteration**.

## 🎯 Problem Statement

A mobile robot uses ultrasonic sensors to navigate a room while following walls. The goal is to learn an optimal policy that enables the robot to:
- Follow walls safely
- Avoid collisions
- Navigate efficiently

## 🔧 MDP Components

| Component | Description |
|-----------|-------------|
| **States (S)** | Discretized sensor readings (Front, Left, Right, Back) |
| **Actions (A)** | Move-Forward, Slight-Right-Turn, Sharp-Right-Turn, Slight-Left-Turn |
| **Transitions P(s'\|s,a)** | Learned from dataset |
| **Rewards R(s,a)** | Designed for wall-following behavior |
| **Discount (γ)** | 0.9 |

## 📊 Dataset

- **Source:** [Kaggle - Robot Navigation Dataset](https://www.kaggle.com/datasets/uciml/wall-following-robot)
- **Samples:** 5,456 sensor readings
- **Features:** 4 ultrasonic sensors
- **Actions:** 4 movement classes

## 🚀 Algorithm

**Value Iteration** is used to compute the optimal value function:

$$V(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$$

## 📁 Files

| File | Description |
|------|-------------|
| `optimal_value_function.csv` | Optimal V(s) and policy for each state |
| `q_values.csv` | Q-values for all state-action pairs |
| `transitions.csv` | Transition probabilities |
| `rewards.csv` | Reward function |
| `convergence.csv` | Convergence history |
| `trajectory.csv` | Robot simulation trajectory |

## 🖥️ Run Locally

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/mdp-robot-navigation.git
cd mdp-robot-navigation

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py