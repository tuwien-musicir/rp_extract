# 2015-04 by Thomas Lidy

# MP3 READ: mini function to decode mp3 using external program
# as there is no Python library for it, we need to use external tools (mpg123, lame, ffmpeg)

import os # for calling external program for mp3 decoding
import subprocess # for subprocess calls
import tempfile
from scipy.io import wavfile


# check if a command exists on the system (without knowing the path, i.e. like Linux 'which')
def cmd_exists(cmd):
    return subprocess.call("type " + cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0

def mp3_read(filename):

    temp = tempfile.NamedTemporaryFile(suffix='.wav')
    # print 'gettempdir():', tempfile.gettempdir()

    # check a number of MP3 Decoder tools if they are available
    cmd=[]
    args=[]

    cmd.append('mpg123')
    args.append ('-q -w "' + temp.name + '" "' + filename + '"')

    cmd.append('lame')
    args.append ('--quiet --decode "' + filename + '" "' + temp.name + '"')

    # TODO ADD ffmpeg

    success = False

    for i in range(len(cmd)):

        if cmd_exists(cmd[i]):

            print 'Decoding mp3 with: ', cmd[i], args[i]

            try:
                # execute external command:
                # import subprocess
                #return_code = call('mpg123') # does work
                #return_code = call(['mpg123',args]) # did not work
                #print return_code

                os.popen(cmd[i] + ' ' + args[i])
                fs, data = wavfile.read(temp.name)
                #os.remove(tempfile) # now done by finally part after temp.close() automtacally be tempfile class
                success = True

            finally:
                # Automatically cleans up (deletes) the temp file
                temp.close()
                # print 'Exists after close:', os.path.exists(temp.name)
        if success:
            break

    # TODO:
    # if not success:
    # print error message: no decoder found

    return (fs, data)