# Gym environment for A07-Race

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

from tm_interface.client import TMInterfaceClient


class TrackmaniaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, track_path=None):
        super().__init__()
        
        # Action space: [steering, throttle, brake]
        # Steering: -1 (left) to 1 (right)
        # Throttle: 0 to 1
        # Brake: 0 to 1
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space: simple version = speed + position (x, y, z)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, -np.inf, -np.inf]),
            high=np.array([500.0, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )

        self.track_path = track_path
        self.client = TMInterfaceClient()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Load the track in TM
        if self.track_path:
            self.client.load_map(self.track_path)
        self.client.restart()

        # Give game some time
        time.sleep(0.5)

        # Get initial state
        state = self._get_state()
        return state, {}

    def step(self, action):
        steering, throttle, brake = action

        # Send controls
        self.client.set_controls(steering, throttle, brake)
        time.sleep(0.05)  # 20Hz control rate

        # Get new state
        state = self._get_state()

        # Reward = forward velocity (basic example)
        reward = state[0]

        # Done if car is upside down or race finished
        done = self.client.is_finished() or self.client.is_crashed()

        return state, reward, done, False, {}

    def _get_state(self):
        """Retrieve state from TMInterface."""
        car_state = self.client.get_car_state()
        speed = car_state.speed
        pos = car_state.position  # tuple (x, y, z)
        return np.array([speed, pos[0], pos[1], pos[2]], dtype=np.float32)

    def render(self):
        # No-op, game renders itself
        pass

    def close(self):
        self.client.disconnect()

