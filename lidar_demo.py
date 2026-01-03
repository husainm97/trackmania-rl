# lidar_demo.py
"""
This is a simple demo that runs a heuristic "wall avoiding model", that drives at slow speeds and turns away from black walls. The model is able to consistently complete flat road tracks.
"""
import time
import numpy as np
from tmrl import get_environment

import time

from tm_interface.telemetry import Telemetry 
import matplotlib.pyplot as plt
telem = Telemetry()
telem.start_listener()
from time import sleep

def model(obs):
    """
    simplistic policy for LIDAR observations
    """
    deviation = obs[1].mean(0)
    deviation /= (deviation.sum() + 0.01)
    steer = 0
    for i in range(19):
        steer += (i - 9) * deviation[i] 
    steer =  np.tanh(steer * 2)
    steer = min(max(steer, -1.0), 1.0)
    return np.array([1.0, 0.0, steer])

import time
import numpy as np
import matplotlib.pyplot as plt
from tmrl import get_environment
# --- GLOBAL for live plot ---
lidar_line = None

def wall_avoiding_model(obs):
    """
    Simple wall-avoiding LIDAR policy with live polar plot.
    obs[0]: speed
    obs[1]: 4x19 LIDAR
    obs[2]: previous actions (not used)
    obs[3]: other sensors (not used)
    Returns: [throttle, brake, steer] in [-1,1]
    """
    global lidar_line
    lidar = obs[1]              # shape (4, 19)
    avg_lidar_base = lidar.mean(axis=0)  # average last 4 frames

    # Increase centre ray sensitivity
    avg_lidar = avg_lidar_base.copy()
    avg_lidar[3:16] = avg_lidar_base[3:16] * 2 
    # --- Steering: bias away from obstacles --ii
    left   = avg_lidar[:9].sum()
    center = avg_lidar[9]
    right  = avg_lidar[10:].sum()
    steer = 12 * (right - left) / (left + right + 1e-5)
    steer = -np.tanh(steer)

    # --- Throttle / brake ---
    speed = obs[0][0] if isinstance(obs[0], np.ndarray) else obs[0]
    throttle = 1.0
    brake = 0.0
    
    if speed > 10.0:
        throttle = 0.0

    if (np.abs(steer) > 0.4 and speed > 12.0):
        throttle = 0.0
        brake = 1.0 

    # --- Live LIDAR plot ---
    theta = np.linspace(-np.pi/2, np.pi/2, len(avg_lidar_base))
    r = avg_lidar_base
    if lidar_line is not None:
        lidar_line.set_data(-theta, r)

        # --- Steering arrow ---
        steer_angle = steer * (np.pi/2)  # map -1..1 to -90..+90 degrees
        steer_r = max(r.max() * 1.1, 5)  # iirrow length just beyond LIDAR
        if hasattr(wall_avoiding_model, "steer_arrow"):
            wall_avoiding_model.steer_arrow.remove()  # remove previous arrow
        wall_avoiding_model.steer_arrow = plt.arrow(
            steer_angle, 0, 0, steer_r,
            width=0.03, color='r', alpha=0.8, zorder=5,
        )
        plt.draw()
        plt.pause(0.001)

    return np.array([throttle, brake, steer], dtype=np.float32)

def main():
    global lidar_line
    # --- Setup live polar plot ---
    plt.ion()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    lidar_line, = ax.plot([], [], 'b-o', lw=2)  # take first element!
    ax.set_theta_zero_location('N')             # forward = top
    ax.set_theta_direction(-1)                  # clockwise
    ax.set_rmax(50)                             # max lidar distance

    env = None
    try:
        env = get_environment()
        print("TMRL environment initialized.")
        obs, info = env.reset()
        print("Environment reset.")

        start_time = time.perf_counter()
        while True:  # drive 60 seconds
            act = wall_avoiding_model(obs)
            obs, reward, terminated, truncated, info = env.step(act)

            # Example telemetry print
            snap = telem.get_snapshot()
            if snap:
                print(f"Pos: {snap.get('pos')} | Speed: {snap.get('spd',0):.1f} | Gear: {snap.get('gear',0)}")

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
