# channels/internet_radio.py
# implements a pseudo analogue tuning for digital channels
# requires pulse audio for mixing of audio signals ( sudo apt install pulseaudio pulseaudio-utils )


import asyncio
import re
import subprocess
import os

from base_channel import BaseChannel

class InternetRadioChannel(BaseChannel):
    def __init__(self, config):
        super().__init__(config)
        self.play_lock = asyncio.Lock()

        self.stream_url = config.get("stream_url")

        print(self.stream_url)
        self.ch_playing_index = 0
        self.ch_index = 0
        self.process=None
        self.enc_actual = 25
        self.vol = 0
        self.ffplay_index = -1
        self.ffplay_noise_index = -1


        self.noise = None
        #subprocess.run( ["ffplay", "-f",  "lavfi", "-i", "anoisesrc=color=white:duration=3600", "-nodisp"], env={**os.environ, "PULSE_PROP": "application.name=AnalogueNoise"}, capture_output=True, text=True )



    async def play(self):
        if self.noise is None: 
            self.noise = await asyncio.create_subprocess_exec(
                "ffplay", "-f",  "lavfi", "-nodisp", "-autoexit",  "-i", "anoisesrc=color=white:duration=3600",
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,         stderr=subprocess.DEVNULL,
                env={**os.environ, "PULSE_PROP": "application.name=AnalogueNoise"} )






        if self.stream_url is None:
            print(f"[{self.name}] No stream URL provided.")
            return

        if self.process is None:
            print(f"[{self.name}] Starting stream: {self.stream_url[self.ch_index]}")
            self.process = await asyncio.create_subprocess_exec(
                "ffplay", "-nodisp", "-autoexit", self.stream_url[self.ch_index],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,         stderr=subprocess.DEVNULL,
                env={**os.environ, "PULSE_PROP": "application.name=AnalogueTuner"} )

            await self.RecordID() 

        else:
            if self.process.returncode is not None:
                print(f"[{self.name}] System was streaming, but must have had an error")
        
        
    async def on_encoder_B_input(self, value: int):
        
        
        self.vol = self.vol + value

        if(self.vol  > 99):
            self.vol = 99
        if(self.vol < 0):
            self.vol = 0

        print(f"[{self.name}] Encoder B value {self.vol}")

        #await asyncio.create_subprocess_exec(
        #        "amixer", "set", "Master", f"{self.vol}%", 
        #        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,         stderr=subprocess.DEVNULL )

        if self.ffplay_index > 0:
             subprocess.run(["pactl", "set-sink-input-volume", str(self.ffplay_index), f"{self.vol}%"])





    async def on_encoder_A_input(self, value: int):
        
        self.enc_actual = self.enc_actual + value

        if(self.enc_actual > 99):
            self.enc_actual = 99
        if(self.enc_actual < 0):
            self.enc_actual = 0
        
        #record the channel we should be on... 
        self.ch_index = await self.get_channel_index(self.enc_actual)

        print(f"[{self.name}] Encoder A value {self.enc_actual} Channel Index {self.ch_index}")
        
        #set the volumes for each source, one is the inverse of the other.... 

        if self.ffplay_index > 0:
             subprocess.run(["pactl", "set-sink-input-volume", str(self.ffplay_index), f"{await self.get_volume_val(self.enc_actual)}%"])


        if self.ffplay_noise_index > 0:
             noise_vol = 100  - (await self.get_volume_val(self.enc_actual)) 
             subprocess.run(["pactl", "set-sink-input-volume", str(self.ffplay_noise_index), f"{noise_vol}%"])



        if self.ch_index == self.ch_playing_index:
            return


        # if we are playing we should stop playing the current channel and restart the new one!! 
        await self.stop()
        print(f"[{self.name}] Starting stream: {self.stream_url[self.ch_index]}")
        self.process = await asyncio.create_subprocess_exec(
                "ffplay", "-nodisp", "-autoexit", self.stream_url[self.ch_index],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,         stderr=subprocess.DEVNULL,
                env={**os.environ, "PULSE_PROP": "application.name=AnalogueTuner"} )

        self.ch_playing_index= self.ch_index

        print(f"[{self.name}] Channel Index adjusted to {self.ch_index}")

        # store the pulse ctl audio number here so we can use it for adjusting the volume on the stream... 
        await self.RecordID()

        
    async def RecordID(self):

        pid = self.process.pid
        await asyncio.sleep(2)  # Give PulseAudio time to register the stream

        # Run pactl to get sink inputs
        pactl_output = subprocess.check_output(["pactl", "list", "sink-inputs"], text=True)
        #print(pactl_output)

        self.ffplay_index = -1

        lines = pactl_output.splitlines()
        current_index = None
        found_blocks = []
        block = []
        for line in lines:
            if line.startswith("Sink Input #"):
                if block:
                    found_blocks.append((current_index, block))
                    block = []
                current_index = int(line.split("#")[1])
            block.append(line)
        if block:
            found_blocks.append((current_index, block))
        for index, block in found_blocks:
            for line in block:
                if f'application.name = "AnalogueTuner"' in line:
                    self.ffplay_index = index 
                if f'application.name = "AnalogueNoise"' in line:
                    self.ffplay_noise_index = index 


        if self.ffplay_index == -1:    
            print(f"[{self.name}] No matching sink input found.")

        if self.ffplay_noise_index == -1:    
            print(f"[{self.name}] No matching sink input found.")



    async def RecordID2(self):

        pid = self.process.pid
        await asyncio.sleep(2)  # Give PulseAudio time to register the stream

        # Run pactl to get sink inputs
        pactl_output = subprocess.check_output(["pactl", "list", "sink-inputs"], text=True)
        #print(pactl_output)

        self.ffplay_index = -1

        # Parse sink input index by matching the PID
        sink_inputs = pactl_output.split("Sink Input #")
        for entry in sink_inputs:
            
            if f'application.process.id = "{pid}"' in entry:
                match = re.search(r"^(\d+)", entry.strip())
                if match:
                    index = int(match.group(1))
                    print(f"ffplay stream index: {index}")
                    self.ffplay_index = index
    
        if self.ffplay_index == -1:    
            print(f"[{self.name}] No matching sink input found.")





    async def stop(self):
#        print(f"[{self.name}] Stop Requested")
        if self.process and self.process.returncode is None:
            print(f"[{self.name}] Stopping stream.")
            self.process.terminate()
            await self.process.wait()
        self.process = None

       # if self.noise:
       #     print(f"[{self.name}] Stopping noise.")
       #     self.noise.terminate()
       #     await self.noise.wait()
       # self.noise = None



    async def get_volume_val(self, encoder_value):
         # Lookup table with 100 entries
        lookup_table = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 40, 60, 80, 100, 80, 60, 40, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 40, 60, 80, 100, 80, 60, 40, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 40, 60, 80, 100, 80, 60, 40, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if 0 <= encoder_value < len(lookup_table):
            return lookup_table[encoder_value]
        else:
            raise ValueError("Encoder value must be between 0 and 99 inclusive.")

	

    
    async def get_channel_index(self, encoder_value):
         # Lookup table with 100 entries
        lookup_table = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1,2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        if 0 <= encoder_value < len(lookup_table):
            return lookup_table[encoder_value]
        else:
            raise ValueError("Encoder value must be between 0 and 99 inclusive.")


