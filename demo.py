# demo.py

import time
import numpy as np
from tmrl import get_environment

def main():
    """
    Initializes a TMRL gymnasium environment, drives the car forward with random steering
    for 60 seconds, and logs environment information.
    """
    env = None
    try:
        # Initialize the TMRL environment.
        # Currently this requires the game to be running
        env = get_environment()
        print("TMRL environment initialized.")
        
        # Reset the environment to start a new episode.
        # This places the car at the starting line.
        observation, info = env.reset()
        print("Environment reset to the starting state.")
        
        # Define a simple action: full throttle, no brake, no steering.
        # The action space is a numpy array of [gas, brake, steering].
        drive_forward_action = np.array([1.0, 0.0, steer], dtype=np.float32)
        
        # Run the simulation for a fixed duration (e.g., 60 seconds).
        print("Driving forward for 60 seconds...")
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < 600.0:
            steer = np.random.rand() * 2 - 1 
            throttle = 1.0
            brake = 0.0

            drive_forward_action = np.array([throttle, brake, steer], dtype=np.float32)
            # Send the "drive forward" action to the environment.
            # `env.step` returns the new observation, reward, termination flags, and an info dict.
            observation, reward, terminated, truncated, info = env.step(drive_forward_action)

            # Print default tmrl game state information 
            print(f"Time: {time.perf_counter() - start_time:.2f}s | "
                f"Speed: {observation[0][0]:.2f} m/s | "
                f"Gear: {int(observation[1][0])} | "
                f"Distance: {observation[2][0]:.2f} m")
    
            speed = observation[0][0]

            # Check for termination. If the car crashes or finishes the track, the episode ends.
            if terminated or truncated:
                print("\nEpisode terminated. The car likely crashed or finished.")
                break
                
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
    finally:
        # Clean up by releasing resources and closing the environment
        if env is not None:
            env.close()
            print("\nEnvironment closed successfully.")

if __name__ == "__main__":
    main()