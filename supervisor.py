import importlib
import asyncio
from input.keyboard_handler import KeyboardHandler
from input.encoder_handler import EncoderHandler

class Supervisor:
    def __init__(self, config):
        self.config = config
        self.channels = {}
        self.encoder_map = {}
        self.keyboard = KeyboardHandler()
        self.encoder = EncoderHandler(config)
        self._load_channels()
        
        self.encoderA=0
        self.encoderB=0
        self.encoder_type = config["encoder_mapping"]["type"]
        
        #check for autostart flag in yaml file, if present auto start the first channel with it set
        
        #for ch_config in self.config["channels"]:
        #    auto_start = ch_config.get("auto_start","")
        #    if auto_start != None:
        #        if auto_start == True:
        #            ch_config.toggle()
        

# Inside Supervisor class

    def _load_channels(self):
        self.channels = {}
        self.key_to_channel = {}

        for ch_config in self.config["channels"]:
            module_path, class_name = ch_config["class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            channel_class = getattr(module, class_name)

            name = ch_config["name"]
            button = ch_config.get("button", "").strip().upper()

            instance = channel_class(ch_config)

            self.channels[name] = instance
            if button:
                self.key_to_channel[button] = name

            
            
            

    async def run(self):
        tasks = [
            self.keyboard.listen(self._on_key),
            self.encoder.listen(self._on_encoder),
            *[ch.run() for ch in self.channels.values()]
        ]
        await asyncio.gather(*tasks)
        
    async def stop_all(self, except_name=None):
        for name, ch in self.channels.items():
            if ch.state.name == "RUNNING" and name != except_name:
                await ch.toggle()

    async def _on_key(self, key):
        #key = self._normalize_key(key)
        key = key.strip().upper().replace(" ", "")
        
        if key == "Q":
            print("Quitting: stopping all channels.")
            await self.stop_all()
            return
            
        if key == "+":
            self.encoderA=+1
            for name, ch in self.channels.items():
                if ch.state.name == "RUNNING":
                    await ch.handle_encoder_A(self.encoderA) 
        if key == "-":
            self.encoderA=-1
            for name, ch in self.channels.items():
                if ch.state.name == "RUNNING":
                    await ch.handle_encoder_A(self.encoderA) 

        if key == ".":
            self.encoderB=+1
            for name, ch in self.channels.items():
                if ch.state.name == "RUNNING":
                    await ch.handle_encoder_B(self.encoderB) 
        if key == ",":
            self.encoderB=-1
            for name, ch in self.channels.items():
                if ch.state.name == "RUNNING":
                    await ch.handle_encoder_B(self.encoderB)

        if key in self.key_to_channel:
            new_channel_name = self.key_to_channel[key]
            new_channel = self.channels[new_channel_name]

            # Stop currently running channels
            for name, ch in self.channels.items():
                if ch.state.name == "RUNNING" and name != new_channel_name:
                    await ch.toggle()  # Stop the current one

            # Start new channel (or toggle it if it's already running)
            await new_channel.toggle()

    async def _on_encoder(self, event_type, value):
        if event_type == "rotateA":
            print(f"Rotary encoder A moved: {value}")
            for name, ch in self.channels.items():
                if ch.state.name == "RUNNING":
                    await ch.handle_encoder_A(value) 
        elif event_type == "rotateB":
            print(f"Rotary encoder B moved: {value}")
            for name, ch in self.channels.items():
                if ch.state.name == "RUNNING":
                    await ch.handle_encoder_B(value) 
        elif event_type == "button":
	    #rotate around channels on button press!!!!! 
            #quit the current channel and go to the next!
            #print("Button pressed!")
            running = False

            for name, ch in self.channels.items():
                if ch.state.name == "RUNNING":
                    running = True
                   

            if not running:
               #print("requesting first channel to start play ")
               await self.channel_list[0].toggle()
               self.channel_index = 0
            else:
            
                if self.channel_index == len(self.channel_list)-1:
                    await self.channel_list[self.channel_index].toggle()
                else:
                    #print("playing next")
                    await self.channel_list[self.channel_index].toggle() #stop current
                    self.channel_index = self.channel_index+1
                    await self.channel_list[self.channel_index].toggle() #start next... 

