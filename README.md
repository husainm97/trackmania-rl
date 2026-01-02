# 🚀 TrackMania RL

🔴🔴🔴🔴🔴  
⚫️⚫️⚫️⚫️⚫️

Aim: **Train an AI agent to master TrackMania (2020) in real time using reinforcement learning.**  

This project is under **active development**, with updates, modules, experiments, rolling out in my free time. 

---

## 🏎️ Project Overview

The goal of this repository is to develop an RL agent that can autonomously drive, navigate, and optimize performance in TrackMania, an arcade formula racing game. Using Python, PyTorch, and TMRL, the agent's goal will be to use in-game telemetry, including LIDAR, speed, and previous actions, to learn optimal racing behavior.

The core engine establishing communication between the Python modules and the game is established, heuristic models successfully complete tracks. The illustration below shows the agent's view as a LIDAR dashboard in blue and the chosen steering action as a red arc. Throttle is currently limited to release when over 8m/s. Brakes are applied if "too close" to a wall to ease steering. Steering model output (-1, 1) is defined by: 

```steer = -np.tanh(10 * (right - left) / (left + right + 1e-5))```

where ```right``` and ```left``` are the summed ray distances on each side.

<img src="images/LIDAR_Heuristic.gif" width="100%" />
---

## 📈 Progress So Far

1. Imported the **TMRL interface** along with OpenPlanet plugins **DataSender** and **TMRL_data** to enable communication between Python and the game.
2. Implemented basic input control by sending throttle and random steering commands on a test track to verify input functionality.
3. Successfully read in-game telemetry: speed, gear, progress, and LIDAR data.
4. Tested heuristic "wall-avoiding" models at low speeds as rudimentary telemetry-guided inputs.
5. Handed control to an **uninitialized PyTorch network** for preliminary testing.

---
## 🛞 Next Steps

- Fine-tune input frequency and control loop timing for smoother driving.
- Integrate **TMInterface** for faster, more efficient training cycles.
- Build the training loop and define the reward function: maximize distance, minimize lap time, and stay close to an "optimal" trajectory.
- Implement random respwans to prevent overlearning start of the track.
- Expand LIDAR-based perception and temporal memory for more aggressive racing strategies.

---
## 💡 Concept

The core model blueprint is a Convolutional Neural Network (CNN) paired with Long Short-Term Memory (LSTM) units. This allows the model to effectively process both high-dimensional spatial features (visual track data) and temporal dependencies (the car's momentum and history over time).

Models could be trained using only live information, to produce a "generalist" AI, as well as with track information to anticipate upcoming turns. 

---
## 🏁 Finish Line

The project will be considered complete when I am not longer able to beat the AI. 

---

## 🤝 Contributing

Contributions, suggestions, or bug reports are **welcome**! Feel free to open issues or submit pull requests to help improve the project.

---

## ⚡ Status

This project is currently in progress. This repository holds the current status of the work.
