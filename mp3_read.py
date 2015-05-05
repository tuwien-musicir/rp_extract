# 2015-04 by Thomas Lidy

# MP3 READ: mini function to decode mp3 using external program
# as there is no Python library for it, we need to use external tools (mpg123, lame, ffmpeg)

import os # for calling external program for mp3 decoding
import tempfile
from scipy.io import wavfile


def mp3_read(filename):

    temp = tempfile.NamedTemporaryFile(suffix='.wav')
    # print 'gettempdir():', tempfile.gettempdir()

    cmd = 'mpg123'
    args = '-q -w "' + temp.name + '" "' + filename + '"'
    print 'Decoding mp3: ', cmd, args

    try:
        # execute external command:
        # import subprocess
        #return_code = call('mpg123') # does work
        #return_code = call(['mpg123',args]) # did not work
        #print return_code

        os.popen(cmd + ' ' + args)
        fs, data = wavfile.read(temp.name)
        #os.remove(tempfile) # now done by finally part after temp.close() automtacally be tempfile class

    finally:
        # Automatically cleans up (deletes) the temp file
        temp.close()
        # print 'Exists after close:', os.path.exists(temp.name)

    return (fs, data)