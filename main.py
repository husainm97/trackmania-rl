import time
import gym
import numpy as np
from gym_envs.tm_env import TMEnv

def main():
    env = TMEnv()
    num_episodes = 3

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        print(f"Episode {ep + 1} start")
        
        while not done:
            # Random action example (replace with heuristic or agent later)
            action = env.action_space.sample()
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            
            print(f"Step {step}: Reward={reward:.2f} Done={done}")
            
            # Slow down so you can observe actions on screen
            time.sleep(0.1)

        print(f"Episode {ep + 1} finished. Total Reward: {total_reward:.2f}\n")
    
    env.close()

if __name__ == "__main__":
    main()
