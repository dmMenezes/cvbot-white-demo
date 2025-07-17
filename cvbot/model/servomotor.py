from cvbot.model.actuator import Actuator
from cvbot.model.named_device import NamedDevice
from pydantic import Field, BaseModel


class Servomotor(Actuator, NamedDevice):
    """Represents a Servo motor actuator."""

    position: int = Field(256,
                          description="Position of the servomotor. Can be in range 0-512")
    """Position of the servomotor. Can be in range 0-512."""
