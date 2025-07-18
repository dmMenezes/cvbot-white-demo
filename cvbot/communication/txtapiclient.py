import io
import time
from collections.abc import AsyncGenerator
from typing import Dict, List, Optional

import numpy as np
import PIL
import PIL.Image
from cvtools.logger.logging import logger
from numpy import ndarray

from cvbot.communication.controller import Controller
from cvbot.model.camera import Camera
from cvbot.model.counter_motor import CounterMotor
from cvbot.model.device import Device
from cvbot.model.servomotor import Servomotor

try:
    import aiohttp
    import cvtxtclient
    from aiohttp import ClientSession
    from cvtxtclient.api.config import APIConfig
    from cvtxtclient.api.controller import ControllerAPI as TxtApiControllerAPI
except ImportError:
    cvtxtclient = None
    logger.warning(
        "cvtxtclient not found. Is the package installed? TxtApiClient will not be available."
    )
    ClientSession = None
import asyncio

from cvbot.communication.txtapiconverter import TxtApiConverter


class TxtApiClient(Controller):
    """TxtApi Client communicates with a TxtAPI controller which is assumed to have all the devices connected."""

    def __init__(
        self,
        host: str,
        port: int,
        api_key: str = None,
        api_base_path: str = "/api/v1",
        # type: ignore[valid-type]
        session: Optional[ClientSession] = None, # type: ignore
    ) -> None:
        """Initialize the TxtApiClient with the given parameters.

        Parameters
        ----------
        host : str
            Hostname or IP address of the TxtAPI controller to reach the api.
        port : int
            The port of the TxtAPI controller to reach the api.
        api_key : str, optional
            The api key to authorize the requests, by default None
        api_base_path : str, optional
            The base path of the api, by default "/api/v1".
        session : Optional[ClientSession], optional
            The aiohttp session to use for the requests, by default None. If None, a new session will be created.
            To effectively use client pooling, its recommended to use a application wide session.
        """
        super().__init__()
        url = r"http://{}:{}{}".format(host, port, api_base_path)
        self._api_config = APIConfig(url, api_key)
        self.api = TxtApiControllerAPI(self._api_config, session=session)
        self.converter = TxtApiConverter(self)

    async def discover_devices(self) -> Dict[int, Device]:
        """Triggers a device discovery process.

        Found devices can in a further step be added to the controller, by calling the add_devices method.

        Returns
        -------
        Dict[int, Device]
            Dictionary of devices found by the discovery process.
        """
        from cvtxtclient.models.camera_config import CameraConfig as TxtApiCameraConfig
        from cvtxtclient.models.motor import Motor as TxtApiMotor
        from cvtxtclient.models.servomotor import Servomotor as TxtApiServoMotor

        max_id = None
        # Get the current max id from the devices in the controller.
        devices = dict()

        # Initialize controller
        await self.api.init_controller_by_id(0)

        ##### Discover actuators #####
        # As the txt api does not support a motor lookup, we need to set the motors to zero speed.
        api_motors = [
            TxtApiMotor(enabled=True, values=[0], direction="CW", name="M" + str(i))
            for i in range(1, 5)
        ]

        counter_tasks = [
            asyncio.create_task(self.api.get_controller_counter_by_id(0, i))
            for i in range(1, 5)
        ]
        await asyncio.gather(*counter_tasks)

        for idx, api_motor in enumerate(api_motors):
            counter = counter_tasks[idx].result()
            motor = self.converter.from_api((api_motor, counter), max_id=max_id)
            if max_id is None:
                max_id = motor.id
            else:
                max_id = max(max_id, motor.id)
            devices[motor.id] = motor

        api_servo_motors = [
            TxtApiServoMotor(enabled=True, value=256, name="S" + str(i))
            for i in range(1, 4)
        ]

        for idx, smot in enumerate(api_servo_motors):
            motor = self.converter.from_api(smot, max_id=max_id)
            if max_id is None:
                max_id = motor.id
            else:
                max_id = max(max_id, motor.id)
            devices[motor.id] = motor

        # Trigger motor updates in parallel
        motor_tasks = [
            asyncio.create_task(
                self.api.update_controller_motor_by_id(0, motor_id, motor)
            )
            for motor_id, motor in enumerate(api_motors, start=1)
        ]

        servo_tasks = [
            asyncio.create_task(
                self.api.update_controller_servomotor_by_id(0, motor_id, motor)
            )
            for motor_id, motor in enumerate(api_servo_motors, start=1)
        ]

        await asyncio.gather(*(motor_tasks + servo_tasks))

        ##### Discover sensors #####
        api_camera = TxtApiCameraConfig()
        camera = self.converter.from_api(api_camera, max_id=max_id)
        if max_id is None:
            max_id = camera.id
        else:
            max_id = max(max_id, camera.id)
        devices[camera.id] = camera
        return devices

    async def open_camera(self, camera: Camera) -> AsyncGenerator[ndarray, None]:
        try:
            await asyncio.wait_for(self.api.start_camera(self.converter.to_api(camera)), timeout=10)
            async for frame_bytes in self.api.camera_image_stream():
                pil_image = PIL.Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                yield np.array(pil_image)
        finally:
            await asyncio.wait_for(self.api.stop_camera(), timeout=10)

    async def update_motors(self, *device: CounterMotor) -> None:
        """
        Updates the speed of the motors in the api.

        Parameters
        ----------
        device : Device
            The device to update.
        """
        tasks = []
        for dev in device:
            mot, cnt = self.converter.to_api(dev)
            tasks.append(
                asyncio.create_task(
                    self.api.update_controller_motor_by_id(0, dev.name[-1], mot)
                )
            )
        return await asyncio.gather(*tasks)

    async def update_servomotors(self, *device: Servomotor) -> None:
        """
        Updates the position of the servomotors in the api.

        Parameters
        ----------
        device : Device
            The device to update.
        """
        tasks = []
        for dev in device:
            smot = self.converter.to_api(dev)
            tasks.append(
                asyncio.create_task(
                    self.api.update_controller_servomotor_by_id(0, dev.name[-1], smot)
                )
            )
        await asyncio.gather(*tasks)

    async def update_counters(self, *device: CounterMotor) -> None:
        """
        Updates the count of the counters in the api.

        Parameters
        ----------
        device : Device
            The device to update.
        """
        tasks = []
        for dev in device:
            mot, cnt = self.converter.to_api(dev)
            tasks.append(
                asyncio.create_task(
                    self.api.update_controller_counter_by_id(0, dev.name[-1], cnt)
                )
            )
        await asyncio.gather(*tasks)

    async def read_counters(self, *device: CounterMotor) -> List[CounterMotor]:
        """
        Reads the count of the counters in the api.

        Parameters
        ----------
        device : Device
            The device to update.
        """
        tasks = []
        for dev in device:
            tasks.append(
                asyncio.create_task(
                    self.api.get_controller_counter_by_id(0, dev.name[-1])
                )
            )
        await asyncio.gather(*tasks)
        ret = []
        for task in tasks:
            cnt = task.result()
            # Find corresponding motor
            mname = cnt.name.replace("C", "M")
            motor = next(
                (
                    m
                    for m in self._devices.values()
                    if m.name == mname and isinstance(m, CounterMotor)
                ),
                None,
            )
            if motor is not None:
                motor.last_count = motor.count
                motor.last_recorded_at = motor.recorded_at
                motor.count = cnt.count
                motor.recorded_at = time.time()
                ret.append(motor)
        return ret

    async def close(self):
        """Clean up the internal aiohttp session if we created it."""
        session = self.api._client.session  # access the actual aiohttp.ClientSession
        if isinstance(session, ClientSession) and not session.closed:
            await session.close()
