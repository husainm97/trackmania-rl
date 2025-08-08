# TMInterface connection and control
import time
import threading
from typing import Tuple

# Import TMInterface Python bindings — adjust import if needed
import TMInterface as tmi


class CarState:
    def __init__(self, speed=0.0, position=(0.0, 0.0, 0.0)):
        self.speed = speed
        self.position = position


class TMInterfaceClient:
    def __init__(self):
        self.client = tmi.Client()
        self.client.connect()

        self._controls = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}
        self._car_state = CarState()
        self._finished = False
        self._crashed = False

        # Start background thread to update state continuously
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        while self._running:
            try:
                # Update car state from TMInterface
                car = self.client.car
                self._car_state.speed = car.speed
                pos = car.position
                self._car_state.position = (pos.x, pos.y, pos.z)

                # Update race status
                self._finished = self.client.is_finished()
                self._crashed = self.client.is_crashed()
            except Exception as e:
                print(f"TMInterface update error: {e}")

            time.sleep(0.02)  # 50Hz update rate

    def load_map(self, path: str):
        print(f"Loading map: {path}")
        self.client.load_map(path)
        time.sleep(1)  # Wait for the map to load

    def restart(self):
        self.client.restart()
        self._finished = False
        self._crashed = False
        time.sleep(0.5)

    def set_controls(self, steering: float, throttle: float, brake: float):
        # Clamp inputs
        steering = max(-1.0, min(1.0, steering))
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))

        self._controls.update({"steering": steering, "throttle": throttle, "brake": brake})

        # Send controls to TMInterface
        self.client.set_steering(steering)
        self.client.set_throttle(throttle)
        self.client.set_brake(brake)

    def get_car_state(self) -> CarState:
        return self._car_state

    def is_finished(self) -> bool:
        return self._finished

    def is_crashed(self) -> bool:
        return self._crashed

    def disconnect(self):
        self._running = False
        self._thread.join()
        self.client.disconnect()
