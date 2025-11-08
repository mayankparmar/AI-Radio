# base_channel.py

from abc import ABC, abstractmethod
from enum import Enum
import asyncio

class ChannelState(Enum):
    STOPPED = 0
    RUNNING = 1
    PAUSED = 2
    ERROR = 3

class BaseChannel(ABC):
    def __init__(self, config: dict):
        self.name = config["name"]
        self.config = config
        self.state = ChannelState.STOPPED
        self.encoder_A_value = 0
        self.encoder_B_value = 0
        self.requires_internet = config.get("requires_internet", False)

    async def run(self):
        try:
            while True:
                if self.state == ChannelState.RUNNING:
                    await self.play()
                await asyncio.sleep(1.0)
        except Exception as e:
            self.state = ChannelState.ERROR
            await self.on_error(e)

    async def toggle(self):
        if self.state == ChannelState.RUNNING:
            self.state = ChannelState.STOPPED
            await self.stop()
        else:
            self.state = ChannelState.RUNNING

    async def handle_encoder_A(self, value: int):
        self.encoder_A_value = value
        await self.on_encoder_A_input(value)

    async def handle_encoder_B(self, value: int):
        self.encoder_B_value = value
        await self.on_encoder_B_input(value)



    @abstractmethod
    async def stop(self):
        """called to stop the running audio."""
        pass

    @abstractmethod
    async def play(self):
        """Called repeatedly while channel is in RUNNING state."""
        pass

    @abstractmethod
    async def on_encoder_A_input(self, value: int):
        """Called when encoder sends new input."""
        pass

    @abstractmethod
    async def on_encoder_B_input(self, value: int):
        """Called when encoder sends new input."""
        pass


    async def on_error(self, error: Exception):
        print(f"[{self.name}] Error: {error}")
