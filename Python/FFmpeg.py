import numpy as np

import subprocess
import os

from scipy.io.arff import loadarff
from Tools.external.WrapperLib import Wrapper

from Utils.Temporary_Utils import make_temp_file, remove_temp_file

from Utils.AudioUtils import read_wave_file

class FFmpeg(Wrapper):
    
    
    def __init__(self, bin_path):
        
        self.binary_path = bin_path
    

    def convert(self, input, output):
        
        # check if file exists
        if not os.path.exists(input):
            raise NameError("File not existing:" + input)
        
        
        cmd = ["-i", input, output]

        so, se = self.call_binary(cmd)
        
    def convertAndRead(self, input):
        
        tmp_file_name = "{0}.wav".format(make_temp_file(suffix="wav"))
        
        self.convert(input, tmp_file_name)
        
        fs, dat = read_wave_file(tmp_file_name)
        
        remove_temp_file(tmp_file_name)
        
        return fs, dat
        

if __name__ == '__main__':
    
    ffmpeg = FFmpeg("D:/MIR/Tools/ffmpeg/bin/ffmpeg.exe")
    
    #dat = ffmpeg.convert("C:/Users/Public/Music/Sample Music/Kalimba.mp3", "D:/tmp/test.wav")
    
    fs, dat = ffmpeg.convertAndRead("C:/Users/Public/Music/Sample Music/Kalimba.mp3")
    
    print dat.shape
    
    
    
    
    
    