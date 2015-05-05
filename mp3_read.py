# 2015-04 by Thomas Lidy

# MP3 READ: mini function to decode mp3 using external program
# as there is no Python library for it, we need to use external tools (mpg123, lame, ffmpeg)

import os # for calling external program for mp3 decoding
from scipy.io import wavfile


def mp3_read(filename):
    cmd = 'mpg123'
    tempfile = "temp.wav" # TODO: use system temp dir
    args = '-q -w "' + tempfile + '" "' + filename + '"'
    print 'Decoding mp3 ...'

    # execute external command:
    # import subprocess
    #return_code = call('mpg123') # does work
    #return_code = call(['mpg123',args]) # did not work
    #print return_code

    os.popen(cmd + ' ' + args)
    fs, data = wavfile.read(tempfile)
    os.remove(tempfile)
    return (fs, data)
