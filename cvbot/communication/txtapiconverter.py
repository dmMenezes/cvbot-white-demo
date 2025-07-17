from typing import Any, Optional

from cvbot.model.camera import Camera
from cvbot.model.counter_motor import CounterMotor
from cvbot.model.device import Device
from cvbot.model.motor import Motor
from cvbot.model.servomotor import Servomotor

try:
    import aiohttp
    import cvtxtclient
    from aiohttp import ClientSession
    from cvtxtclient.api.config import APIConfig
    from cvtxtclient.api.controller import ControllerAPI as TxtApiControllerAPI
    from cvtxtclient.models.camera_config import CameraConfig as TXTApiCameraConfig
    from cvtxtclient.models.counter import Counter as TXTApiCounter
    from cvtxtclient.models.motor import Motor as TXTApiMotor
    from cvtxtclient.models.output import Output as TXTApiOutput
    from cvtxtclient.models.servomotor import Servomotor as TXTApiServomotor
except ImportError:
    cvtxtclient = None
    ClientSession = None
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cvbot.communication.txtapiclient import TxtApiClient
else:
    TxtApiClient = Any

import time


class TxtApiConverter:
    """
    This class is responsible for converting the cvbot model objects to the format that can be used by the cvtxtclient API.
    """

    def __init__(self, client: TxtApiClient):
        self.client = client

    def get_new_id(self, max_id: Optional[int] = None) -> int:
        """
        Get a new id for the device.

        Returns
        -------
        int
            The new id for the device.
        """
        # Get the current max id from the devices in the controller.
        if max_id is None:
            max_id = max(self.client._devices.keys(), default=-1)
        # Increment the max id by 1 to get a new id.
        new_id = max_id + 1
        return new_id

    def to_api(self, device: Device) -> Any:
        """
        Convert the given text to a format that can be used by the API.

        Parameters
        ----------
        device : Device
            The device to be converted.

        Returns
        -------
        Any
            The converted device.
        """
        new_device = None
        if isinstance(device, CounterMotor):
            # Create a new TXTApiMotor object with the values from the cvbot CounterMotor object.
            new_motor = TXTApiMotor(
                name=device.name, enabled=True, values=[abs(device.speed)])
            # Set the direction of the motor based on the speed.
            if device.speed < 0:
                new_motor.direction = "CCW"
            else:
                new_motor.direction = "CW"
            # Create a counter
            new_counter = TXTApiCounter(
                name=device.name.replace("M", "C"), enabled=True, digital=True, count=device.count)
            new_device = (new_motor, new_counter)
        elif isinstance(device, Motor):
            # Create a new TXTApiMotor object with the values from the cvbot Motor object.
            new_device = TXTApiMotor(
                name=device.name, enabled=True, values=[abs(device.speed)])
            # Set the direction of the motor based on the speed.
            if device.speed < 0:
                new_device.direction = "CCW"
            else:
                new_device.direction = "CW"
        elif isinstance(device, Servomotor):
            # Create a new TXTApiServomotor object with the values from the cvbot Servomotor object.
            new_device = TXTApiServomotor(
                name=device.name, enabled=True, value=device.position)
        elif isinstance(device, Camera):
            new_device = TXTApiCameraConfig(fps=device.fps, width=device.width, height=device.height)
        return new_device

    def from_api(self, device: Any, max_id: Optional[int] = None) -> Device:
        mapped_device = None
        if isinstance(device, TXTApiMotor):
            # Check if there is a device with the same name, then use the id of the device.
            mapped_device = next((x for x in self.client._devices if isinstance(
                x, Motor) and x.name == device.name), None)
            if not mapped_device:
                # Create a new device with a new id.
                new_id = self.get_new_id(max_id)
                mapped_device = Motor(id=new_id, name=device.name, speed=0)

            # Update the device with the values from the API.
            speed = device.values[0] if device.values else 0
            # If the direction is CCW, set the speed to negative.
            if device.direction == "CCW" and speed > 0:
                speed = -speed
            mapped_device.speed = speed
        elif isinstance(device, tuple) and len(device) == 2 and isinstance(device[0], TXTApiMotor) and isinstance(device[1], TXTApiCounter):
            # Check if there is a device with the same name, then use the id of the device.
            device_motor: TXTApiMotor = device[0]
            device_counter: TXTApiCounter = device[1]
            mapped_device = next((x for x in self.client._devices if isinstance(
                x, CounterMotor) and x.name == device_motor.name), None)
            if not mapped_device:
                # Create a new device with a new id.
                new_id = self.get_new_id(max_id)
                mapped_device = CounterMotor(
                    id=new_id, name=device_motor.name, speed=0)

            # Update the device with the values from the API.
            speed = device_motor.values[0] if device[0].values else 0
            # If the direction is CCW, set the speed to negative.
            if device_motor.direction == "CCW" and speed > 0:
                speed = -speed
            mapped_device.speed = speed

            mapped_device.last_recorded_at = mapped_device.recorded_at
            mapped_device.last_count = mapped_device.count
            mapped_device.count = device_counter.count
            mapped_device.recorded_at = time.time()
        elif isinstance(device, TXTApiServomotor):
            device: TXTApiServomotor
            # Check if there is a device with the same name, then use the id of the device.
            mapped_device = next((x for x in self.client._devices if isinstance(
                x, Servomotor) and x.name == device.name), None)
            if not mapped_device:
                # Create a new device with a new id.
                new_id = self.get_new_id(max_id)
                mapped_device = Servomotor(id=new_id, name=device.name)

            # Update the device with the values from the API.
            mapped_device.position = device.value
        elif isinstance(device, TXTApiCameraConfig):
            device: TXTApiCameraConfig
            # Check if there is a device with the same name, then use the id of the device.
            mapped_device = next((x for x in self.client._devices if isinstance(x, Camera)), None)
            if not mapped_device:
                # Create a new device with a new id.
                new_id = self.get_new_id(max_id)
                mapped_device = Camera(id=new_id, width=device.width, height=device.height, fps=device.fps)
        else:
            raise ValueError(f"Unsupported device type: {type(device)}")
        return mapped_device
