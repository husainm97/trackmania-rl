# models/lidar_actor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tmrl.actor import ActorModule
import numpy as np
from tmrl import get_environment

class LidarActor(ActorModule):
    """
    RL Actor for racing with LIDAR.
    
    Input: tuple of (speed, last 4 LIDAR frames, prev action, prev prev action)
    Output: [throttle, brake, steer]
        throttle, brake: binary (0 or 1)
        steer: continuous [-1, 1]
    """
    def __init__(self, observation_space, action_space, hidden_size=256, lstm_hidden=128):
        super().__init__(observation_space, action_space)
        
        # LIDAR feature extractor (convolution over last 4 LIDAR frames)
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lidar_feature_size = 64 * 19  # 64 channels * 19 distances

        # MLP for scalar + previous actions
        self.fc_scalar = nn.Sequential(
            nn.Linear(1 + 3 + 3, hidden_size),  # speed + 2 previous actions
            nn.ReLU()
        )

        # LSTM to capture temporal racing dynamics
        self.lstm = nn.LSTM(input_size=self.lidar_feature_size + hidden_size,
                            hidden_size=lstm_hidden,
                            batch_first=True)
        
        # Actor heads
        self.fc_actor = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_size),
            nn.ReLU()
        )
        self.throttle_head = nn.Linear(hidden_size, 1)
        self.brake_head = nn.Linear(hidden_size, 1)
        self.steer_head = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        """
        obs: tuple (speed, lidar, prev_action, prev_prev_action)
        """
        speed = torch.tensor(obs[0], dtype=torch.float32).view(-1, 1)
        lidar = torch.tensor(obs[1], dtype=torch.float32).unsqueeze(0)  # shape (1, 4, 19)
        prev_actions = torch.tensor(np.concatenate([obs[2], obs[3]]), dtype=torch.float32).view(1, -1)

        # Process LIDAR
        lidar_feat = self.lidar_conv(lidar)  # shape (1, lidar_feature_size)
        scalar_feat = self.fc_scalar(torch.cat([speed, prev_actions], dim=-1))

        x = torch.cat([lidar_feat, scalar_feat], dim=-1).unsqueeze(0)  # add seq dim
        lstm_out, _ = self.lstm(x)  # output shape: (1, 1, lstm_hidden)
        lstm_out = lstm_out[:, -1, :]  # last step

        actor_feat = self.fc_actor(lstm_out)

        # Discrete outputs: throttle, brake
        throttle_prob = torch.sigmoid(self.throttle_head(actor_feat))
        brake_prob = torch.sigmoid(self.brake_head(actor_feat))
        # Convert to 0 or 1
        throttle = (throttle_prob > 0.5).float()
        brake = (brake_prob > 0.5).float()

        # Continuous output: steer
        steer = torch.tanh(self.steer_head(actor_feat))

        action = torch.cat([throttle, brake, steer], dim=-1)
        return action

    def act(self, obs, test=False):
        with torch.no_grad():
            out = self.forward(obs)  # shape (1, 3)
            throttle = 1 if out[0,0].item() > 0 else 0
            brake    = 1 if out[0,1].item() > 0 else 0
            steer    = out[0,2].item()  # pick first (and only) batch element
            print(f'Got {out[0,0].item(),out[0,1].item(),out[0,2].item()}')
            print(f'Doing: {[throttle, brake, steer]}')
            return [throttle, brake, steer]

    # optional: your existing act_ method can stay
    def act_(self, obs, test=False):
        return self.act(obs, test)