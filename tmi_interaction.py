"""
This file is a combination of the TMInterface and GameInstanceManager classes you provided.
It includes minimal mocks of the dependencies to make it runnable.
"""

import signal
import socket
import struct
from enum import IntEnum, auto
import numpy as np
import numpy.typing as npt
import time
import subprocess
import os
import psutil
import win32.lib.win32con as win32con
import win32com.client
import win32gui
import win32process
from typing import Callable, Dict, List
import math
import cv2

# --- MOCK DEPENDENCIES ---
from config_files import config_copy, user_config
class contact_materials:
    physics_behavior_fromint = [0]
class map_loader:
    def precalculate_virtual_checkpoints_information(zone_centers): return None,None,None,None
    def hide_personal_record_replay(map_path, bool): pass
    def sync_virtual_and_real_checkpoints(zone_centers, map_path): return None, None
class CheckpointData:
    cp_states_field = 8
    cp_times_field = 12
    def __init__(self): self.cp_states_length, self.cp_times_length, self.data = 0,0,np.array([])
    def resize(self, field, length): pass
class SimStateData:
    def __init__(self, data):
        self.data = data
        self.car_state = type('car_state', (object,), {'pos': np.array([0,0,0], dtype=float)})
        self.race_time = 0.0
        self.cp_data = CheckpointData()
        self.dyna = type('dyna', (object,), {'current_state': type('state', (object,), {'position': [0,0,0], 'rotation': type('rot', (object,), {'to_numpy': lambda: np.eye(3)}), 'linear_speed': [0,0,0], 'angular_speed': [0,0,0]})})
        self.scene_mobil = type('scene_mobil', (object,), {'engine': type('eng', (object,), {'gear': 0, 'actual_rpm': 0}),
                                                         'gearbox_state': 0, 'is_freewheeling': False})
        self.simulation_wheels = [type('wheel', (object,), {'real_time_state': type('ws', (object,), {'is_sliding': False, 'has_ground_contact': False, 'damper_absorb': 0, 'contact_material_id': 0})}) for _ in range(4)]
        
        if len(data) >= 16:
            self.car_state.pos[0] = struct.unpack("f", data[0:4])[0]
            self.car_state.pos[1] = struct.unpack("f", data[4:8])[0]
            self.car_state.pos[2] = struct.unpack("f", data[8:12])[0]
            self.race_time = struct.unpack("f", data[12:16])[0]

# --- USER'S TMInterface and GameInstanceManager classes start here ---
HOST = "127.0.0.1"

class MessageType(IntEnum):
    SC_RUN_STEP_SYNC = auto()
    SC_CHECKPOINT_COUNT_CHANGED_SYNC = auto()
    SC_LAP_COUNT_CHANGED_SYNC = auto()
    SC_REQUESTED_FRAME_SYNC = auto()
    SC_ON_CONNECT_SYNC = auto()
    C_SET_SPEED = auto()
    C_REWIND_TO_STATE = auto()
    C_REWIND_TO_CURRENT_STATE = auto()
    C_GET_SIMULATION_STATE = auto()
    C_SET_INPUT_STATE = auto()
    C_GIVE_UP = auto()
    C_PREVENT_SIMULATION_FINISH = auto()
    C_SHUTDOWN = auto()
    C_EXECUTE_COMMAND = auto()
    C_SET_TIMEOUT = auto()
    C_RACE_FINISHED = auto()
    C_REQUEST_FRAME = auto()
    C_RESET_CAMERA = auto()
    C_SET_ON_STEP_PERIOD = auto()
    C_UNREQUEST_FRAME = auto()
    C_TOGGLE_INTERFACE = auto()
    C_IS_IN_MENUS = auto()
    C_GET_INPUTS = auto()

class TMInterface:
    registered = False
    
    def __init__(self, port: int):
        self.port = port
        self.sock = None

    def close(self):
        if self.sock:
            self.sock.sendall(struct.pack("i", MessageType.C_SHUTDOWN))
            self.sock.close()
        self.registered = False

    def signal_handler(self, sig, frame):
        print("Shutting down...")
        self.close()

    def register(self, timeout=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        signal.signal(signal.SIGINT, self.signal_handler)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((HOST, self.port))
        self.registered = True
        print("Connected")

    def get_simulation_state(self):
        self.sock.sendall(struct.pack("i", MessageType.C_GET_SIMULATION_STATE))
        state_length = self._read_int32()
        state = SimStateData(self.sock.recv(state_length, socket.MSG_WAITALL))
        return state

    def set_input_state(self, left: bool, right: bool, accelerate: bool, brake: bool):
        self.sock.sendall(
            struct.pack("iBBBB", MessageType.C_SET_INPUT_STATE, np.uint8(left), np.uint8(right), np.uint8(accelerate), np.uint8(brake))
        )
    
    def set_timeout(self, new_timeout: int):
        self.sock.sendall(struct.pack("iI", MessageType.C_SET_TIMEOUT, np.uint32(new_timeout)))

    def execute_command(self, command: str):
        self.sock.sendall(struct.pack("ii", MessageType.C_EXECUTE_COMMAND, np.int32(len(command))))
        self.sock.sendall(command.encode())

    def rewind_to_state(self, state):
        self.sock.sendall(struct.pack("ii", MessageType.C_REWIND_TO_STATE, np.int32(len(state.data))))
        self.sock.sendall(state.data)

    def rewind_to_current_state(self):
        self.sock.sendall(struct.pack("i", MessageType.C_REWIND_TO_CURRENT_STATE))

    def get_frame(self, width: int, height: int):
        frame_data = self.sock.recv(width * height * 4, socket.MSG_WAITALL)
        return np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 4))
    
    def request_frame(self, W: int, H: int):
        self.sock.sendall(struct.pack("iii", MessageType.C_REQUEST_FRAME, np.int32(W), np.int32(H)))

    def unrequest_frame(self):
        self.sock.sendall(struct.pack("i", MessageType.C_UNREQUEST_FRAME))

    def set_speed(self, new_speed):
        self.sock.sendall(struct.pack("if", MessageType.C_SET_SPEED, np.float32(new_speed)))

    def toggle_interface(self, new_val: bool):
        self.sock.sendall(struct.pack("ii", MessageType.C_TOGGLE_INTERFACE, np.int32(new_val)))

    def give_up(self):
        self.sock.sendall(struct.pack("i", MessageType.C_GIVE_UP))
        
    def prevent_simulation_finish(self):
        self.sock.sendall(struct.pack("i", MessageType.C_PREVENT_SIMULATION_FINISH))

    def _read_int32(self):
        return struct.unpack("i", self.sock.recv(4, socket.MSG_WAITALL))[0]

    def _respond_to_call(self, response_type):
        self.sock.sendall(struct.pack("i", np.int32(response_type)))

def _set_window_focus(trackmania_window):
    if config_copy.is_linux:
        # Xdo().activate_window(trackmania_window)
        pass
    else:
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys("%")
        win32gui.SetForegroundWindow(trackmania_window)


def ensure_not_minimized(trackmania_window):
    if config_copy.is_linux:
        # Xdo().map_window(trackmania_window)
        pass
    else:
        if win32gui.IsIconic(trackmania_window):
            win32gui.ShowWindow(trackmania_window, win32con.SW_SHOWNORMAL)

def update_current_zone_idx(
    current_zone_idx: int,
    zone_centers: npt.NDArray,
    sim_state_position: npt.NDArray,
    max_allowable_distance_to_virtual_checkpoint: float,
    next_real_checkpoint_positions: npt.NDArray,
    max_allowable_distance_to_real_checkpoint: npt.NDArray,
):
    return current_zone_idx

class GameInstanceManager:
    def __init__(
        self,
        game_spawning_lock,
        running_speed=1,
        run_steps_per_action=10,
        max_overall_duration_ms=2000,
        max_minirace_duration_ms=2000,
        tmi_port=None,
    ):
        self.iface = None
        self.latest_tm_engine_speed_requested = 1
        self.running_speed = running_speed
        self.run_steps_per_action = run_steps_per_action
        self.max_overall_duration_ms = max_overall_duration_ms
        self.max_minirace_duration_ms = max_minirace_duration_ms
        self.timeout_has_been_set = False
        self.msgtype_response_to_wakeup_TMI = None
        self.latest_map_path_requested = -2
        self.last_rollout_crashed = False
        self.last_game_reboot = time.perf_counter()
        self.UI_disabled = False
        self.tmi_port = tmi_port
        self.tm_process_id = None
        self.tm_window_id = None
        self.start_states = {}
        self.game_spawning_lock = game_spawning_lock
        self.game_activated = False

    def get_tm_window_id(self):
        assert self.tm_process_id is not None
        if config_copy.is_linux:
            self.tm_window_id = None
        else:
            def get_hwnds_for_pid(pid):
                def callback(hwnd, hwnds):
                    _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                    if found_pid == pid:
                        hwnds.append(hwnd)
                    return True
                hwnds = []
                win32gui.EnumWindows(callback, hwnds)
                return hwnds
            while True:
                for hwnd in get_hwnds_for_pid(self.tm_process_id):
                    if win32gui.GetWindowText(hwnd).startswith("Track"):
                        self.tm_window_id = hwnd
                        return

    def is_tm_process(self, process: psutil.Process) -> bool:
        try:
            return process.name().startswith("TmForever")
        except psutil.NoSuchProcess:
            return False

    def get_tm_pids(self) -> List[int]:
        return [process.pid for process in psutil.process_iter() if self.is_tm_process(process)]

    def launch_game(self):
        self.tm_process_id = None
        print("Python is trying to open this file:", user_config.windows_TMLoader_path)
        if config_copy.is_linux:
            self.game_spawning_lock.acquire()
            pid_before = self.get_tm_pids()
            os.system(str(user_config.linux_launch_game_path) + " " + str(self.tmi_port))
            while True:
                pid_after = self.get_tm_pids()
                tmi_pid_candidates = set(pid_after) - set(pid_before)
                if len(tmi_pid_candidates) > 0:
                    assert len(tmi_pid_candidates) == 1
                    break
            self.tm_process_id = list(tmi_pid_candidates)[0]
        else:
            # First, launch the game executable
            #subprocess.Popen([user_config.windows_TMLoader_path])
            #time.sleep(2)  # Give the game a moment to start up

            # Launch TMInterface from the correct directory
            tm_interface_dir = os.path.dirname(user_config.windows_TMLoader_path)
            launch_string = (
                'powershell -executionPolicy bypass -command "& {'
                f" $process = start-process -FilePath '{user_config.windows_TMLoader_path}'"
                " -PassThru -ArgumentList "
                f'\'run TmForever "{user_config.windows_TMLoader_profile_name}" /configstring=\\"set custom_port {self.tmi_port}\\"\';'
                ' echo exit $process.id}"'
            )
            print('setting process id to:')
            tmi_process_id = int(subprocess.check_output(launch_string).decode().split("\r\n")[1])
            
            self.tmi_process_id = tmi_process_id

            time.sleep(2)  # Give TMInterface a moment to start the server

            try:
                tmi_process = psutil.Process(self.tmi_process_id)
                connections = tmi_process.connections()
                if connections:
                    print("TMInterface.exe is listening on the following connections:")
                    for conn in connections:
                        if conn.status == 'LISTEN':
                            print(f"  - Protocol: {conn.type}, Local Address: {conn.laddr}")
                else:
                    print("TMInterface.exe is not listening on any ports.")
            except psutil.NoSuchProcess:
                print("TMInterface.exe process not found after launch.")

            while self.tm_process_id is None:
                for proc in psutil.process_iter(['pid', 'name', 'ppid']):
                    if proc.info['name'] == 'TmForever.exe' and proc.info['ppid'] == self.tmi_process_id:
                        self.tm_process_id = proc.info['pid']
                        break
                if self.tm_process_id is None:
                    time.sleep(1) # Wait before checking again
        print(f"Found Trackmania process id: {self.tm_process_id=}")
        self.last_game_reboot = time.perf_counter()
        self.latest_map_path_requested = -1
        self.msgtype_response_to_wakeup_TMI = None
        while not self.is_game_running():
            time.sleep(0)
        self.get_tm_window_id()

    def is_game_running(self):
        return (self.tm_process_id is not None) and (self.tm_process_id in (p.pid for p in psutil.process_iter()))

    def close_game(self):
        self.timeout_has_been_set = False
        self.game_activated = False
        assert self.tm_process_id is not None
        if config_copy.is_linux:
            os.system("kill -9 " + str(self.tm_process_id))
        else:
            os.system(f"taskkill /PID {self.tm_process_id} /f")
        while self.is_game_running():
            time.sleep(0)

    def ensure_game_launched(self):
        if not self.is_game_running():
            print("Game not found. Restarting TMInterface.")
            self.launch_game()

    def grab_screen(self):
        return self.iface.get_frame(config_copy.W_downsized, config_copy.H_downsized)

    def request_speed(self, requested_speed: float):
        self.iface.set_speed(requested_speed)
        self.latest_tm_engine_speed_requested = requested_speed

    def request_inputs(self, action_idx: int, rollout_results: Dict):
        self.iface.set_input_state(**config_copy.inputs[action_idx])

    def request_map(self, map_path: str, zone_centers: npt.NDArray):
        self.latest_map_path_requested = map_path
        if user_config.is_linux:
            map_path = map_path.replace("\\", "/")
        else:
            map_path = map_path.replace("/", "\\")
        map_loader.hide_personal_record_replay(map_path, True)
        self.iface.execute_command(f"map {map_path}")
        self.UI_disabled = False
        (
            self.next_real_checkpoint_positions,
            self.max_allowable_distance_to_real_checkpoint,
        ) = map_loader.sync_virtual_and_real_checkpoints(zone_centers, map_path)
    
    def rollout(self, exploration_policy: Callable, map_path: str, zone_centers: npt.NDArray, update_network: Callable):
        (
            zone_transitions,
            distance_between_zone_transitions,
            distance_from_start_track_to_prev_zone_transition,
            normalized_vector_along_track_axis,
        ) = map_loader.precalculate_virtual_checkpoints_information(zone_centers)
        
        self.ensure_game_launched()
        
        # NOTE: Your code is complex and has many dependencies. This is a minimal demo to show the port is set.
        # It's not a full, functional rollout, but it will confirm the port is set correctly.
        
        if (self.iface is None) or (not self.iface.registered):
            print(f"Initialize connection to TMInterface on port {self.tmi_port}")
            self.iface = TMInterface(self.tmi_port)
            connection_attempts_start_time = time.perf_counter()
            while True:
                try:
                    self.iface.register()
                    print(f"Successfully connected on port {self.tmi_port}")
                    break
                except ConnectionRefusedError as e:
                    current_time = time.perf_counter()
                    if current_time - connection_attempts_start_time > 30:
                        print(f"Connection failed after {current_time - connection_attempts_start_time:.1f}s. Is the game running?")
                        raise e
                    print(f"Connection to TMInterface unsuccessful for {current_time - connection_attempts_start_time:.1f}s. Retrying...")
                    time.sleep(1)
        
        # A minimal action to show that the connection is active
        print("Sending a test command to the game.")
        self.iface.execute_command(f"chat hello from your script on port {self.tmi_port}")
        
        # Minimal loop to keep the demo alive
        for _ in range(5):
            print("Demo running...")
            time.sleep(1)
            
        print("Demo finished. Disconnecting.")
        self.iface.close()


class FakeLock:
    def acquire(self): pass
    def release(self): pass
    
if __name__ == '__main__':
    # You must set a port here to be used by the GameInstanceManager
    PORT = 5450
    
    # We create a lock because the class you provided expects it, but it's not needed for this simple demo
    fake_lock = FakeLock()
    
    # Create an instance of your GameInstanceManager class, passing the port you want to use.
    game_manager = GameInstanceManager(game_spawning_lock=fake_lock, tmi_port=PORT)
    
    # You must provide a dummy function for the rollout method to call
    def dummy_policy(*args): pass
    
    # Run the rollout, which will launch the game and set the port for you
    game_manager.rollout(
        exploration_policy=dummy_policy,
        map_path="C:\\Program Files (x86)\\Steam\\steamapps\\common\\TrackMania Nations Forever\\GameData\\Tracks\\Campaigns\\Nations\\White\\A07-Race.Challenge.Gbx",
        zone_centers=np.array([]),
        update_network=dummy_policy
    )