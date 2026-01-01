from models.lidar_agent import LidarActor


import time
import numpy as np
from tmrl import get_environment

import time

from tm_interface.telemetry import Telemetry 
import matplotlib.pyplot as plt

telem = Telemetry()
telem.start_listener()

from time import sleep
import time
import numpy as np
import matplotlib.pyplot as plt
from tmrl import get_environment



def main():
    env = None

    env = get_environment()
    actor = LidarActor(env.observation_space, env.action_space)
    try:

        print("TMRL environment initialized.")
        obs, info = env.reset()
        print("Environment reset.")

        start_time = time.perf_counter()
        while True:  # drive 60 seconds
            act = actor.act_(obs, test=False)
            obs, reward, terminated, truncated, info = env.step(act)

            # Example telemetry print
            snap = telem.get_snapshot()

            if terminated or truncated:
                print("\nEpisode terminated, resetting environment...")
                obs, info = env.reset()
                from pynput.keyboard import Controller, Key

                keyboard = Controller()
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)
                # press and release 'i'
                sleep(0.5)
                keyboard.press('i')
                keyboard.release('i')
                keyboard.press('i')
                keyboard.release('i')

                start_time = time.perf_counter()  # restart timer if you want
                #break

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if env:
            env.close()
        print("\nEnvironment closed.")

if __name__ == "__main__":
    main()