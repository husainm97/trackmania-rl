# TrackMania RL

This project, currently under active development, aims to train an AI agent to drive in the racing game **TrackMania (2020)**. The repository is updated regularly upon completing exercises or significant units of work toward this goal.

## Progress So Far

1. Imported the **TMRL interface** along with OpenPlanet plugins **DataSender** and **TMRL_data** to enable communication between Python and the game.
2. Implemented basic input control by sending throttle and random steering commands on a test track to verify input functionality.
3. Successfully read basic telemetry data from the game, including speed, gear, progress, and screengrabs. Also tested rudimentary telemetry-guided inputs.

## Next Steps

- Integrate **DataSender** outputs into the data stream for verbose game state information.
- Set input frequency and timing to optimize control loops.
- Transition control to the reinforcement learning (RL) agent for training.
- Integrate **TMInterface** for faster and more efficient training cycles.

---
Please feel free to contribute or raise issues if you encounter any problems or have suggestions.
