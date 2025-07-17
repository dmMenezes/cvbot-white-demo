from cvbot.model.actuator import Actuator
from cvbot.model.named_device import NamedDevice
from pydantic import Field, BaseModel


class Motor(Actuator, NamedDevice):
    """Represents a motor output."""

    speed: float = Field(
        0., description="Speed of the motor in Device specific units. May be negative to indicate reverse direction.")
    """Speed of the motor in Device specific units. May be negative to indicate reverse direction."""
