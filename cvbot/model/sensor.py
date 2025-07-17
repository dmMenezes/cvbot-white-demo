from pydantic import BaseModel, Field
from cvbot.model.device import Device


class Sensor(Device):
    """Abstract base class for sensor. Any active component where a value can be read is a sensor."""
    pass
