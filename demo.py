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
        # 1. Initialize the TMRL environment.
        # This will launch the game if it's not already running.
        # Ensure your TrackMania instance is ready as per TMRL setup instructions.
        env = get_environment()
        print("TMRL environment initialized.")
        
        # 2. Reset the environment to start a new episode.
        # This places the car at the starting line.
        # The return values are the initial observation and an info dictionary.
        observation, info = env.reset()
        print("Environment reset to the starting state.")
        
        # 3. Define a simple action: full throttle, no brake, no steering.
        # The action space is a numpy array of [gas, brake, steering].
        steer = np.random.rand() * 2 - 1
        drive_forward_action = np.array([1.0, 0.0, steer], dtype=np.float32)
        
        # 4. Run the simulation for a fixed duration (e.g., 5 seconds).
        print("Driving forward for 5 seconds...")
        start_time = time.perf_counter()

        prev_speed = 0 
        crash = False
        steer = 0.3
        while time.perf_counter() - start_time < 600.0:
            steer = np.random.rand() * 2 - 1 
            throttle = 1.0
            brake = 0.0

            if crash and time.perf_counter() - wait_start < 1.5 and False:
                print('Crashed')
                brake = 1.0
                throttle = 0.0
            drive_forward_action = np.array([throttle, brake, steer], dtype=np.float32)
            # Send the "drive forward" action to the environment.
            # `env.step` returns the new observation, reward, termination flags, and an info dict.
            observation, reward, terminated, truncated, info = env.step(drive_forward_action)
            print(env.step(drive_forward_action))

            #print(f"Full Observation: {observation}")
            # You can also check the type and length to understand its structure
            #print(f"Observation is of type: {type(observation)} and length: {len(observation)}")

            # --- WITH THIS NEW LINE ---
            #print(f"Time: {time.perf_counter() - start_time:.2f}s | "
            #    f"Speed: {observation[0][0]:.2f} m/s | "
            #    f"Gear: {int(observation[1][0])} | "
            #    f"Distance: {observation[2][0]:.2f} m")
    
            
            # The observation is a tuple in this environment.
            # The first element is typically a numpy array containing car state info.
            # The exact structure depends on the tmrl config, but the first value is often speed.
            speed = observation[0][0]
            if speed - prev_speed < -10 and False:
                crash = True
                wait_start = time.perf_counter()
            prev_speed = speed 
            # The info dictionary provides additional details.
            # You can inspect `info.keys()` to see what's available.
            # 'location' and 'pitch' are common keys.
            
            # 5. Log the information to the console.
            # We'll print the time elapsed, speed, and other relevant info.
            #print(f"Time: {time.perf_counter() - start_time:.2f}s | "
            #      f"Speed: {speed:.2f} m/s | "
            #      f"Location (X,Y,Z): {info.get('location', 'N/A')}")
            #print(steer)
            # 6. Check for termination. If the car crashes or finishes the track, the episode ends.
            if terminated or truncated:
                print("\nEpisode terminated. The car likely crashed or finished.")
                break
                
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
    finally:
        # 7. Clean up by closing the environment.
        # This is important to release resources and close the game instance.
        if env is not None:
            env.close()
            print("\nEnvironment closed successfully.")

if __name__ == "__main__":
    main()