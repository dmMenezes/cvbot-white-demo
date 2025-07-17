from cvbot.model.motor import Motor
from cvbot.model.sensor import Sensor

from pydantic import Field, BaseModel
from time import time


class CounterMotor(Motor, Sensor):
    """Represents a motor that can count the number of revolutions."""

    count: int = Field(
        0, description="Number of revolutions counted by the motor.")
    """Number of revolutions counted by the motor."""

    recorded_at: float = Field(
        default_factory=time, description="Timestamp when the count was last recorded.")
    """Timestamp when the count was last recorded."""

    last_count: int = Field(0, description="Last recorded count value.")
    """Last recorded count value."""

    last_recorded_at: float = Field(
        default_factory=time, description="Timestamp when the last count was recorded.")
    """Timestamp when the last count was recorded."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, count={self.count})"

    @property
    def velocity(self) -> float:
        """Calculates the velocity of the motor based on the count and time.

        TODO: Need to factor in the gear ratio of the motor.

        Returns
        -------
        float
            The velocity of the motor in revolutions per minute (RPM).
        """
        if self.recorded_at == self.last_recorded_at:
            return 0.0
        rps = (self.count - self.last_count) / \
            (self.recorded_at - self.last_recorded_at)

        return rps * 60  # Convert to RPM (revolutions per minute)
