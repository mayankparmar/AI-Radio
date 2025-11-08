# vradio
Vintage Radio Python Framework for comment/feedback. 

The idea is that as channels are added the user will create new classes from a template in the channels folder. These will all have a similar format. 

A yaml file is used (config.yaml) to configure the radio and link the various channels to buttons and encoder functions etc. 

Requires python 3.11 and greater, need to consider using poetry to manage python dependencies. 

Now tested on a PI4 and Pi5, can be started from dp@raspberrypi:~/vradio $ python main.py
The libraries are not system-wide installation, instead, on virtual environment. Before launching main.py, activate venv using: `source env/bin/activate`. Ensure you are in vradio folder when you activate venv.

Currently only works using the keyboard and has 2 channel "buttons" mapped in the yaml file. Button "a" = Audio stream radio and button "b" plays MP3 files from the local folder. The "+" and "-" encoder inputs do interact with the audio playing on the "b" channel. 

Uses FFMPEG to play the audio from python, so FFMPEG must be installed. 


