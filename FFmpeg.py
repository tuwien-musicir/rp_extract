import numpy as np

import subprocess
import os
import tempfile
from scipy.io import wavfile


class FFmpeg(object):
    
    
    def __init__(self, bin_path):
        
        self.binary_path = bin_path
    

    def convert(self, input, output):
        
        # check if file exists
        if not os.path.exists(input):
            raise NameError("File not existing:" + input)
        
        
        cmd = ["-i", input, output]

        so, se = self.call_binary(cmd)
        
    def convertAndRead(self, input):
        
        tmp_file_name = "{0}.wav".format(self.make_temp_file(suffix="wav"))
        
        self.convert(input, tmp_file_name)
        
        fs, dat = self.read_wave_file(tmp_file_name)
        
        self.remove_temp_file(tmp_file_name)
        
        return fs, dat
    
    def read_wave_file(self, audio_file):
    
        samplerate, data = wavfile.read(audio_file)
        
        return samplerate, data
    
    def make_temp_file(self, suffix="", prefix=""):
        return tempfile.mktemp(suffix, prefix)
    
    def remove_temp_file(self, path):
        os.remove(path)
        
    def call_binary(self, cmd):
        
        try:
            
            cmd.insert(0,self.binary_path)
            
            # open pipe to shell and execute command
            process = subprocess.Popen(cmd, 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE)
            
            # store cmd-line output
            std_out, std_err = process.communicate()
            
            # convert output to utf-8
            std_out = unicode(std_out, errors='ignore')
            std_err = unicode(std_err, errors='ignore')

        except UnicodeDecodeError as e:
            pass
            #todo handle this
        
        return std_out, std_err
        

if __name__ == '__main__':
    
    ffmpeg = FFmpeg("D:/Research/Tools/ffmpeg/bin/ffmpeg.exe")
    
    #dat = ffmpeg.convert("C:/Users/Public/Music/Sample Music/Kalimba.mp3", "D:/tmp/test.wav")
    
    fs, dat = ffmpeg.convertAndRead("D:/Research/Data/MIR/MVD/Audio/MV-VIS/Dance/Dance_1.mp3")
    
    print dat.shape
    
    
    
    
    
    