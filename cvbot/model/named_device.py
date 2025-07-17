from pydantic import Field, BaseModel

from cvbot.model.device import Device


class NamedDevice(Device):
    """Abstract base class for devices with a name. Any device that has a name is a named device."""

    name: str = Field(..., description="Name of the device.")
    """Name of the device."""
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"