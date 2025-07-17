from pydantic import BaseModel, Field
from cvbot.model.device import Device


class Actuator(Device):
    """Abstract base class for actuators. Any active component where a value can be set is an actuator."""
    pass
