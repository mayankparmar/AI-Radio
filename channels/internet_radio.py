# channels/internet_radio.py

import asyncio

import subprocess

from base_channel import BaseChannel

class InternetRadioChannel(BaseChannel):
    def __init__(self, config):
        super().__init__(config)
        self.play_lock = asyncio.Lock()

        self.stream_url = config.get("stream_url")

        print(self.stream_url)

        self.ch_index = 0
        self.process=None

    async def play(self):
        if self.stream_url is None:
            print(f"[{self.name}] No stream URL provided.")
            return

        if self.process is None:
            print(f"[{self.name}] Starting stream: {self.stream_url[self.ch_index]}")
            self.process = await asyncio.create_subprocess_exec(
                "ffplay", "-nodisp", "-autoexit", self.stream_url[self.ch_index],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,         stderr=subprocess.DEVNULL )
        else:
            if self.process.returncode is not None:
                print(f"[{self.name}] System was streaming, but must have had an error")
        
    async def on_encoder_B_input(self, value: int):
         print(f"[{self.name}] Encoder B not implemented")
        

    async def on_encoder_A_input(self, value: int):
        if self.ch_index < len(self.stream_url)-1:
            self.ch_index=self.ch_index+1
        else:
            self.ch_index = 0


        # if we are playing we should stop playing the current channel and restart the new one!! 
        await self.stop()
        print(f"[{self.name}] Starting stream: {self.stream_url[self.ch_index]}")
        self.process = await asyncio.create_subprocess_exec(
                "ffplay", "-nodisp", "-autoexit", self.stream_url[self.ch_index],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,         stderr=subprocess.DEVNULL )



        print(f"[{self.name}] Channel Index adjusted to {self.ch_index}")



    async def stop(self):
#        print(f"[{self.name}] Stop Requested")
        if self.process and self.process.returncode is None:
            print(f"[{self.name}] Stopping stream.")
            self.process.terminate()
            await self.process.wait()
        self.process = None


