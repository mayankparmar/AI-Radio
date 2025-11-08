# channels/MorseCode.py
import asyncio
import subprocess
import os
import random
import numpy as np
import sounddevice as sd
from enum import Enum



from base_channel import BaseChannel

class PlayBackState(Enum):
    STOPPED = 0
    READY = 1
    RUNNING = 2
    FINISHED = 3
    

class MorseCode(BaseChannel):
    def __init__(self, config):
        super().__init__(config)
        self.words = config.get("text")
        self.static = config.get("static")
        self.word_count = len(self.words)
        self.word_index = 0
        self.audio_index = 1
       
        #p = pyaudio.PyAudio()
        #for i in range(p.get_device_count()):
        #    print(f"[{self.name}] {i} {p.get_device_info_by_index(i)['name']}")


        print(f"[{self.name}]  [{self.words}] [{self.word_count}]")
        
        self.index = 0
        self.playback_state = PlayBackState.STOPPED 
        
        self.morse_code = {
                    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 
                    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 
                    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 
                    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 
                    'Y': '-.--', 'Z': '--..',
                    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-', 
                    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
                    ' ': '/'
                }
                
        self.audio = None

    def text_to_morse(self, text):
        morse = ''
        for char in text.upper():
            if char in self.morse_code:
                morse += self.morse_code[char] + ' '
        return morse
        
    def generate_morse_audio(self, morse):
        sample_rate = 44100
        frequency = 1000  # Hz
        duration_dot = 0.2  # seconds
        duration_dash = 2.2  * duration_dot
        duration_gap = 3.0    #gap between chars
        duration_blanking = 0.25 #blanking period either side of a char... 

        audio = np.array([], dtype=np.float32)

	
        static_before = np.random.uniform(-0.5, 0.5, int(sample_rate * duration_blanking)).astype(np.float32)
        static_after = np.random.uniform(-0.5, 0.5, int(sample_rate * duration_blanking)).astype(np.float32)
        blanking  = np.zeros(int(sample_rate * duration_blanking), dtype=np.float32) 
	
        for char in morse:
            if char == '.':
                signal = np.sin(2 * np.pi * np.arange(int(sample_rate * duration_dot)) * frequency / sample_rate).astype(np.float32)
                noise = np.random.uniform(-0.5, 0.5, int(sample_rate*duration_dot)).astype(np.float32)

                if self.static:
                    audio = np.concatenate((audio, static_before, signal+noise, static_after))
                else:
                    audio = np.concatenate((audio, blanking, signal, blanking))

            elif char == '-':

                signal = np.sin(2 * np.pi * np.arange(int(sample_rate * duration_dash)) * frequency / sample_rate).astype(np.float32)
                noise = np.random.uniform(-0.5, 0.5, int(sample_rate * duration_dash)).astype(np.float32)

                if self.static:
                    audio = np.concatenate((audio, static_before, signal+noise, static_after))
                else:
                    audio = np.concatenate((audio, blanking, signal, blanking))
               

            elif char == ' ':
                if self.static:
                    static = np.random.uniform(-0.5, 0.5, int(sample_rate * duration_gap)).astype(np.float32)
                else:
                    static = np.zeros(int(sample_rate * duration_gap), dtype=np.float32)
                    
                audio = np.concatenate((audio, static))
            elif char == '/':
                static = np.random.uniform(-0.5, 0.5, int(sample_rate * 7 * duration_gap)).astype(np.float32)
                audio = np.concatenate((audio, static))

        return audio

    async def play(self):
        if self.playback_state == PlayBackState.READY:
            print(f"[{self.name}] Play Starting")
            self.playback_state= PlayBackState.RUNNING
            sd.play(self.audio, samplerate=44100)
            sd.wait()  # Wait until playback is finished
#            p = pyaudio.PyAudio()
#            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=True, output_device_index=self.audio_index)
#            stream.write(self.audio.tobytes())
#            stream.stop_stream()
#            stream.close()
#            p.terminate()
            self.playback_state = PlayBackState.STOPPED

            
        


 

    async def stop(self):
        print(f"[{self.name}] Stop called")


    async def on_encoder_B_input(self, value: int):
         print(f"[{self.name}] Encoder B not implemented") 

    async def on_encoder_A_input(self, value: int):
        if self.playback_state== PlayBackState.STOPPED:
            
            #if we are idle we can prepare another word for transmission... 
	    


            #then get the play function to play it!!! 
            text = self.words[self.word_index]
            morse = self.text_to_morse(text)
            self.audio = self.generate_morse_audio(morse)

            print(f"[{self.name}] Next word Selected: {self.words[self.word_index]} {morse}")
	    
            if self.word_index< self.word_count-1:
                self.word_index= self.word_index+ 1
            else:
                self.word_index= 0

            self.playback_state = PlayBackState.READY




  







