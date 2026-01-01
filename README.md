# 🚀 TrackMania RL

**Train an AI agent to master TrackMania (2020) in real time using reinforcement learning.**  
This project is under **active development**, with frequent updates as new modules, experiments, and breakthroughs are completed.

---

## 🏎️ Project Overview

The goal of this repository is to develop an RL agent that can autonomously drive, navigate, and optimize performance in TrackMania 2020, an arcade formula racing game. Using Python, PyTorch, and TMRL, the agent's goal will be to use in-game telemetry, including LIDAR, speed, and previous actions, to learn optimal racing behavior.

The core engine establishing communication between the Python modules and the game is established, heuristic models successfully complete tracks.

![LIDAR_Heuristic](images/LIDAR_Heuristic.gif)
---

## 🔹 Progress So Far

1. Imported the **TMRL interface** along with OpenPlanet plugins **DataSender** and **TMRL_data** to enable communication between Python and the game.
2. Implemented basic input control by sending throttle and random steering commands on a test track to verify input functionality.
3. Successfully read in-game telemetry: speed, gear, progress, and LIDAR data.
4. Tested heuristic "wall-avoiding" models at low speeds as rudimentary telemetry-guided inputs.
5. Handed control to an **uninitialized PyTorch network** for preliminary testing.

---

## 🔹 Next Steps

- Fine-tune input frequency and control loop timing for smoother driving.
- Integrate **TMInterface** for faster, more efficient training cycles.
- Build the training loop and define the reward function: maximize distance, minimize lap time, and stay close to an "optimal" trajectory.
- Expand LIDAR-based perception and temporal memory for more aggressive racing strategies.

---

## 🤝 Contributing

Contributions, suggestions, or bug reports are **welcome**! Feel free to open issues or submit pull requests to help improve the project.

---

## ⚡ Status

This project is currently under development. This repository holds the current status of the work.
