# channels/local_mp3_player.py
import asyncio
import subprocess
import os
import random

from base_channel import BaseChannel

class LocalMP3Channel(BaseChannel):
    def __init__(self, config):
        super().__init__(config)
        self.play_lock = asyncio.Lock()

        self.volume = 50
        self.process = None

        # Get the directory where the current script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Relative folder "audio" inside the script directory
        self.audio_dir = os.path.join(script_dir, 'audio')

        supported_exts = {'.mp3', '.mp4'}
        self.media_files = []

        for filename in os.listdir(self.audio_dir):
            if os.path.splitext(filename)[1].lower() in supported_exts:
                self.media_files.append(os.path.join(self.audio_dir, filename))
        print("Media Files", self.media_files)

        self.file_path = None



    async def play(self):
        async with self.play_lock:
            if self.process is None:
                await self.playNewAudio()
            else:
                if self.process.returncode is not None:
                    print(f"[{self.name}] Stopped")
                    await self.playNewAudio()


 

    async def stop(self):
        async with self.play_lock:

            if self.process and self.process.returncode is None:
                print(f"[{self.name}] Stopping playback.")
                self.process.terminate()
                await self.process.wait()
            self.process = None


    async def playNewAudio(self):
        # select a new autio file. 
        if self.file_path is None:
            self.file_path = random.choice(self.media_files)
        else: 
            choices = [f for f in self.media_files if f != self.file_path]
            if choices:
               self.file_path = random.choice(choices)

        # stop the audio playing, if it is playing
        if self.process and self.process.returncode is None:
            print(f"[{self.name}] Stopping playback.")
            self.process.terminate()
            await self.process.wait()
        self.process = None

        print(f"[{self.name}] Starting playback of: {self.file_path}")
        self.process = await asyncio.create_subprocess_exec("ffplay", "-nodisp", "-autoexit", self.file_path,stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


    async def on_encoder_A_input(self, value: int):
         await self.playNewAudio()

    async def on_encoder_B_input(self, value: int):
         print(f"[{self.name}] Encoder B not implemented")


  







