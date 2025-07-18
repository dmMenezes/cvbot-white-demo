import asyncio

from cvbot.communication import controller
from cvbot.communication.txtapiclient import TxtApiClient
from cvbot.controller.easy_drive_controller import EasyDriveController
from cvbot.config.drive_robot_configuration import DriveRobotConfiguration

from dotenv import load_dotenv
import os


load_dotenv()

HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
KEY = os.getenv("KEY")

async def connect():
    # The following classes are needed to init the drive controller
    try:
        print("Initializing API client and controller")
         # Initialize the API client and controller
        api_client = TxtApiClient(HOST, PORT, KEY)
        controller = EasyDriveController(api_client, DriveRobotConfiguration())
        await api_client.initialize()
    
        await controller.stop()
    except Exception as e:
        print(f"Error initializing API client or controller: {e}")
        exit()
asyncio.run(connect())