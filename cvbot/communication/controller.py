from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Type

import numpy as np

from cvbot.model.counter_motor import CounterMotor
from cvbot.model.device import Device
from cvbot.model.sensor import Sensor
from cvbot.model.servomotor import Servomotor


class Controller:
    """A Communication Controller is responsible for managing the communication between the controlling device and the actuators and sensors.
    Given multiple ways of communication, this class is an interface for the communication methods.
    """

    _devices: Dict[int, Device]
    """Dictionary of devices, known and managed by the controller."""

    _devices_by_type: Dict[Type[Device], Dict[int, Device]]
    """Dictionary of devices, known and managed by the controller, grouped by type."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the CommunicationController with the given parameters."""
        self._devices = dict()
        self._devices_by_type = dict()
        self._initialized = False

    def add_devices(self, *devices: Device) -> None:
        """Adds device to the controller.

        Parameters
        ----------
        devices : List[Device]
            List of devices to be added to the controller.
        """
        for device in devices:
            self._devices[device.id] = device
            if type(device) not in self._devices_by_type:
                self._devices_by_type[type(device)] = dict()
            self._devices_by_type[type(device)][device.id] = device

    def remove_devices(self, *devices: Device) -> None:
        """Removes device from the controller.

        Parameters
        ----------
        devices : List[Device]
            List of devices to be removed from the controller.
        """
        for device in devices:
            if device.id in self._devices:
                if type(device) in self._devices_by_type:
                    del self._devices_by_type[type(device)][device.id]
                del self._devices[device.id]

    def get_device(self, value: int) -> Optional[Device]:
        """
        Returns the devices of the controller.

        Parameters
        ----------
        value : int
            The id of the device to be returned.

        Returns
        -------
        Optional[Device]
            The device with the given id.
            None if the device is not found.
        """
        return self._devices.get(value, None)

    def set_devices(self, *devices: Device) -> None:
        """Sets the devices of the controller.

        Existing devices will be removed and the new devices will be added.

        Parameters
        ----------
        devices : List[Device]
            List of devices to be set to the controller.
        """
        self.remove_devices(*self._devices.values())
        self.add_devices(*devices)

    @abstractmethod
    async def discover_devices(self) -> Dict[int, Device]:
        """Triggers a device discovery process.

        Found deevices can in a further step be added to the controller, by calling the add_devices method.

        Returns
        -------
        Dict[int, Device]
            Dictionary of devices found during the discovery process.
        """
        pass

    async def initialize(self) -> None:
        """Initializes the controller, by discovering the devices and setting them to the controller."""
        self.set_devices(*[v for k, v in (await self.discover_devices()).items()])
        self._initialized = True

    @abstractmethod
    async def open_sensor(self, *device: Sensor) -> AsyncGenerator[np.ndarray]:
        """Opens a sensor stream.

        Returns
        -------
        AsyncGenerator
            A generator that yields sensor values.
        """
        pass

    @abstractmethod
    async def update_motors(self, *device: CounterMotor) -> None:
        """
        Updates the speed of the motors in the api.

        Parameters
        ----------
        device : Device
            The device to update.
        """
        pass

    @abstractmethod
    async def update_servomotors(self, *device: Servomotor) -> None:
        """
        Updates the position of the servomotors in the api.

        Parameters
        ----------
        device : Device
            The device to update.
        """
        pass

    @abstractmethod
    async def update_counters(self, *device: CounterMotor) -> None:
        """
        Updates the count of the counters in the api.

        Parameters
        ----------
        device : Device
            The device to update.
        """
        pass

    @abstractmethod
    async def read_counters(self, *device: CounterMotor) -> List[CounterMotor]:
        """
        Reads the count of the counters in the api.

        Parameters
        ----------
        device : Device
            The device to update.
        """
        pass

    def get_devices_by_type(self, device_type: Type[Device]) -> List[Device]:
        """
        Returns the devices of the controller by type.

        Parameters
        ----------
        device_type : str
            The type of the device to be returned.

        Returns
        -------
        List[Device]
            List of devices with the given type.
        """
        return list(self._devices_by_type.get(device_type, dict()).values())
