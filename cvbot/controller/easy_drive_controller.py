from collections.abc import AsyncGenerator

import numpy as np

from cvbot.communication.controller import Controller
from cvbot.config.drive_robot_configuration import DriveRobotConfiguration
from cvbot.model.camera import Camera
from cvbot.model.counter_motor import CounterMotor


class EasyDriveController:
    """This class is a wrapper around the controller to make it easier to use.
    """

    def __init__(self, control: Controller, config: DriveRobotConfiguration):
        self.control = control
        self.config = config

    async def initialize(self) -> None:
        """
        Initialize the controller.

        Returns
        -------
        bool
            True if the controller is initialized, False otherwise.
        """
        # Initialize the controller.
        await self.control.initialize()

        _ = self.config.kinematic_matrix
        return True

    async def camera(self) -> AsyncGenerator[np.ndarray]:
        """
        Returns a stream of camera frames.
        The frames are of shape (3,H,W) and of dtype float32 with values in [0,1].

        Returns
        -------
        AsyncGenerator
            A generator that yields the camera frames.
        """
        # Set all motors to the same speed.
        camera = self.control.get_devices_by_type(Camera)[0]
        async for frame in self.control.open_camera(camera):
            yield frame

    async def straight(self, speed: int) -> bool:
        """
        Drive straight with a given speed.

        Parameters
        ----------
        speed : int
            The speed of the vehicle in m/s.


        Returns
        -------
        bool
            True if the motors are set to the given speed, False otherwise.
        """
        # Set all motors to the same speed.
        motors = self.control.get_devices_by_type(CounterMotor)
        # Sort according to the motor configuration.
        motors = sorted(
            motors, key=lambda x: self.config.drive_motor_names.index(x.name)
        )

        v = np.array([0.0, speed, 0], dtype=self.config.dtype)
        K = self.config.kinematic_matrix

        w = K @ v

        # If any value is larger than the max speed, scale everything down.
        if np.max(np.abs(w)) > self.config.max_motor_speed:
            w = w / np.max(np.abs(w)) * self.config.max_motor_speed

        for i, motor in enumerate(motors):
            motor.speed = int(float(w[i]))
        res = await self.control.update_motors(*motors)
        return True

    async def side(self, speed: int) -> bool:
        """
        Drive to the side with a given speed.

        Positive speed means to the right, negative speed means to the left.

        Parameters
        ----------
        speed : int
            The speed of the vehicle in m/s.

        Returns
        -------
        bool
            True if the motors are set to the given speed, False otherwise.
        """
        # Set all motors to the same speed.
        motors = self.control.get_devices_by_type(CounterMotor)
        # Sort according to the motor configuration.
        motors = sorted(
            motors, key=lambda x: self.config.drive_motor_names.index(x.name)
        )

        v = np.array([speed, 0, 0], dtype=self.config.dtype)
        K = self.config.kinematic_matrix

        w = K @ v

        # If any value is larger than the max speed, scale everything down.
        if np.max(np.abs(w)) > self.config.max_motor_speed:
            w = w / np.max(np.abs(w)) * self.config.max_motor_speed

        for i, motor in enumerate(motors):
            motor.speed = int(float(w[i]))
        res = await self.control.update_motors(*motors)
        return True

    async def diagonal(self, speed_forward: int, speed_side: int) -> bool:
        """
        Drive to the side with a given speed.

        Parameters
        ----------
        speed_forward : int
            The speed forward of the vehicle in m/s.
            Negative speed means backward.

        speed_side : int
            The speed to the right side of the vehicle in m/s.
            Negative speed means to the left side.

        Returns
        -------
        bool
            True if the motors are set to the given speed, False otherwise.
        """
        # Set all motors to the same speed.
        motors = self.control.get_devices_by_type(CounterMotor)
        # Sort according to the motor configuration.
        motors = sorted(
            motors, key=lambda x: self.config.drive_motor_names.index(x.name)
        )

        v = np.array([speed_side, speed_forward, 0], dtype=self.config.dtype)
        K = self.config.kinematic_matrix

        w = K @ v

        # If any value is larger than the max speed, scale everything down.
        if np.max(np.abs(w)) > self.config.max_motor_speed:
            w = w / np.max(np.abs(w)) * self.config.max_motor_speed

        for i, motor in enumerate(motors):
            motor.speed = int(float(w[i]))
        res = await self.control.update_motors(*motors)
        return True

    async def rotate(self, speed: int) -> bool:
        """
        Rotate the robot with a given speed. Around its y-axis.

        Parameters
        ----------
        speed : int
            The angular speed of the vehicle in rad/s.

        Returns
        -------
        bool
            True if the motors are set to the given speed, False otherwise.
        """
        # Set all motors to the same speed.
        motors = self.control.get_devices_by_type(CounterMotor)
        # Sort according to the motor configuration.
        motors = sorted(
            motors, key=lambda x: self.config.drive_motor_names.index(x.name)
        )

        v = np.array([0.0, 0.0, speed], dtype=self.config.dtype)
        K = self.config.kinematic_matrix

        w = K @ v

        # If any value is larger than the max speed, scale everything down.
        if np.max(np.abs(w)) > self.config.max_motor_speed:
            w = w / np.max(np.abs(w)) * self.config.max_motor_speed

        for i, motor in enumerate(motors):
            motor.speed = int(float(w[i]))
        res = await self.control.update_motors(*motors)
        return True

    async def drive(self, speeds: np.ndarray) -> None:
        """
        Drive the robot by given speeds.


        Parameters
        ----------
        speed : np.ndarray
            The speed of the vehicle in (x - (right), z - (forward), w - (angular)) coordinates.

        Returns
        -------
        bool
            True if the motors are set to the given speed, False otherwise.
        """
        # Set all motors to the same speed.
        motors = self.control.get_devices_by_type(CounterMotor)
        # Sort according to the motor configuration.
        motors = sorted(
            motors, key=lambda x: self.config.drive_motor_names.index(x.name)
        )
        # Set the speed of each motor.
        K = self.config.kinematic_matrix
        w = K @ speeds

        # If any value is larger than the max speed, scale everything down.
        if np.max(np.abs(w)) > self.config.max_motor_speed:
            w = w / np.max(np.abs(w)) * self.config.max_motor_speed

        for i, motor in enumerate(motors):
            motor.speed = int(float(w[i]))

        res = await self.control.update_motors(*motors)
        return True

    async def stop(self) -> None:
        """
        Stop the robot.

        Returns
        -------
        bool
            True if the motors are set to 0, False otherwise.
        """
        # Set all motors to 0.
        motors = self.control.get_devices_by_type(CounterMotor)
        # Sort according to the motor configuration.
        motors = sorted(
            motors, key=lambda x: self.config.drive_motor_names.index(x.name)
        )

        for motor in motors:
            motor.speed = 0
        res = await self.control.update_motors(*motors)
        return True
