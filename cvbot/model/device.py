from pydantic import BaseModel, Field


class Device(BaseModel):
    """Abstract base class for devices. Anything that can be controlled by the CVBot API is a device."""

    id: int = Field(...)
    """Unique identifier for the device."""
