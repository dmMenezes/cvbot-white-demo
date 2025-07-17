from pydantic import Field

from cvbot.model.sensor import Sensor


class Camera(Sensor):
    """Represents a camera."""

    width: int = Field(
        640, description="The width of the camera frames in pixels.", ge=1
    )
    """The width of the camera frames in pixels."""

    height: int = Field(
        480, description="The height of the camera frames in pixels.", ge=1
    )
    """The height of the camera frames in pixels."""

    fps: int = Field(
        30, description="The frames per second captured by the camera.", ge=1
    )
    """The frames per second captured by the camera."""
