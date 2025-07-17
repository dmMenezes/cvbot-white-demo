from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DriveRobotConfiguration:
    """Configuration of the robot. Defines how motors and sensors are aligned.
    The vehicle frame is defined as follows:

    When viewed from the top:
    - The x-axis is pointing to the right,
    - The y-axis is pointing down,
    - The z-axis is pointing in the driving direction, when driving forward.
    - The vehicle frame is defined as the center of the robot. E.g. laying in the center of the motor axis, exactly between all wheels, assuming them to be aligned in a rectangle.

    So, the vehicle coordinate system is right-handed.
    It is also assumed the robot uses Mercanum wheels.
    """

    drive_motor_names: list[str] = field(
        default_factory=lambda: ["M2", "M1", "M3", "M4"])
    """Motor names in order Front-Left, Front-Right, Back-Right, Back-Left. (When viewed from the top)"""

    drive_motor_positions: np.ndarray = field(default_factory=lambda: np.array(
        [[-0.075, 0., 0.05], [0.075, 0., 0.05], [0.075, 0., -0.05], [-0.075, 0., -0.05]]))
    """3D positions of the motors in the robot, in the vehicle frame. Units are in meters.
    Precisely, the positions are the axis center within each wheel.
    Shape (4, 3) (x, y, z)."""

    wheel_radius: float = 0.03
    """Radius of the wheels in meters."""

    gear_ratio: float = 2.0
    """Gear ratio of the motors. E.g. 2.0 means the motor turns twice as fast as the wheel."""

    max_motor_speed: float = 255
    """Max speed of the motors in its control range."""

    dtype: np.dtype = np.float32
    """Data type of the positions."""

    _kinematic_matrix: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)
    """Kinematic matrix of the robot. Shape (4, 3)."""

    @property
    def kinematic_matrix(self) -> np.ndarray:
        """Returns the kinematic matrix of the robot.

        Returns
        -------
        np.ndarray
            Kinematic matrix K of the robot. Shape (4, 3).
            Assumed to by multiplied with the motor speeds, and angle V = (v_x, v_z, w_v) to get the robot speed in the vehicle frame.
            Whereby v_x is the speed in x direction, v_z is the speed in z direction, and w_v is the angular speed around the y axis.    

            Giving the speed control by each motor W = (w_1, w_2, w_3, w_4) in the vehicle frame:
            W = K @ V   

            The velocities in V should be in meters per second.
            The angular speed w_v should be in radians per second.

            Resulting speed is the radians per second of each motor.
        """
        if self._kinematic_matrix is None:
            self._kinematic_matrix = self.get_kinematic_matrix()
        return self._kinematic_matrix

    def get_kinematic_matrix(self) -> np.ndarray:
        """Returns the kinematic matrix of the robot.

        Returns
        -------
        np.ndarray
            Kinematic matrix K of the robot. Shape (3, 4).
            Assumed to by multiplied with the motor speeds, and angle V = (v_x, v_z, w_v) to get the robot speed in the vehicle frame.
            Whereby v_x is the speed in x direction, v_z is the speed in z direction, and w_v is the angular speed around the y axis.    

            Giving the speed control by each motor W = (w_1, w_2, w_3, w_4) in the vehicle frame:
            W = K @ V   

            The velocities in V should be in meters per second.
            The angular speed w_v should be in radians per second.

            Resulting speed is the radians per second of each motor.
        """
        # Get the kinematic matrix of the robot.
        kinematic_matrix = np.zeros((4, 3), self.dtype)

        # Side direction, to drive right.
        kinematic_matrix[:, 0] = np.array([1, 1, -1, -1], dtype=self.dtype)
        # Forward direction of the motors. Left side is inverted
        kinematic_matrix[:, 1] = np.array([1, -1, -1, 1], dtype=self.dtype)

        # Definition of the matrix, rows are motors Fl, Fr, Br, Bl.

        fl = self.drive_motor_positions[0]
        fr = self.drive_motor_positions[1]
        br = self.drive_motor_positions[2]
        bl = self.drive_motor_positions[3]

        # Distance between the wheels in the z-axis
        l = np.abs(fl[2] - bl[2]) / 2
        # Distance between the wheels in the x-axis
        w = np.abs(fl[0] - fr[0]) / 2

        # Clockwise turn (seen from above) around the y-axis.
        kinematic_matrix[0, 2] = (l + w)
        kinematic_matrix[1, 2] = (l + w)
        kinematic_matrix[2, 2] = (l + w)
        kinematic_matrix[3, 2] = (l + w)

        # Multiply with the wheel radius.
        kinematic_matrix *= self.gear_ratio / self.wheel_radius

        return kinematic_matrix
