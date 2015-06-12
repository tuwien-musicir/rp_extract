# 2015-04 by Thomas Lidy

# MP3 READ: mini function to decode mp3 using external program
# as there is no Python library for it, we need to use external tools (mpg123, lame, ffmpeg)

import os # for calling external program for mp3 decoding
import subprocess # for subprocess calls
import tempfile

# Reading WAV files
# scipy.io.wavfile does not support 24 bit Wav files
# from scipy.io import wavfile
# therefore we switch to wavio by Warren Weckesser
# https://github.com/WarrenWeckesser/wavio -  BSD 3-Clause License
import wavio


# check if a command exists on the system (without knowing the path, i.e. like Linux 'which')

def cmd_exists(cmd):
    return subprocess.call("type " + cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0


# read wav files
# returns samplereate (e.g. 44100), samplewith (e.g. 2 for 16 bit) and wavedata (simple array for mono, 2-dim. array for stereo)

def wav_read(filename):

    # check if file exists
    if not os.path.exists(filename):
        raise NameError("File does not exist:" + filename)

    samplerate, samplewidth, wavedata = wavio.readwav(filename)

    return (samplerate, samplewidth, wavedata)


# convert mp3 to wav and read from wav file
# returns samplereate (e.g. 44100), samplewith (e.g. 2 for 16 bit) and wavedata (simple array for mono, 2-dim. array for stereo)

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

    cmd.append('ffmpeg')
    args.append ('-y -v 1 -i "' + filename + '" "' + temp.name + '"')
    # -v adjusts log level, -y option overwrites output file, because it has been created already by tempfile above

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

                samplerate, samplewidth, wavedata = wav_read(temp.name)

                #os.remove(tempfile) # now done automatically by finally part after temp.close() by tempfile class
                success = True

            except: # catch *all* exceptions
                raise OSError("Problem appeared during decoding.")

            finally:
                # Automatically cleans up (deletes) the temp file
                temp.close()

        if success:
            break  # no need to loop further

    if not success:
        raise OSError("No MP3 decoder found. Check if any of these is on your system path: " + ", ".join(cmd) + \
                       " and if not add the path using os.environ['PATH'] += os.pathsep + path.")

    return (samplerate, samplewidth, wavedata)


# generic function capable of reading both .wav and .mp3 files
# returns samplereate (e.g. 44100), samplewith (e.g. 2 for 16 bit) and wavedata (simple array for mono, 2-dim. array for stereo)

def audiofile_read(filename):

    # check if file exists
    if not os.path.exists(filename):
        raise NameError("File does not exist:" + filename)

    basename, ext = os.path.splitext(filename)

    if ext.lower() == '.wav':
        return(wav_read(filename))
    elif ext.lower() == '.mp3':
        return(mp3_read(filename))
    else:
        raise NameError("File name extension must be either .wav or .mp3 when using audiofile_read. Extension found: " + ext)


# main routine: to test if decoding works properly

if __name__ == '__main__':

    # if your MP3 decoder is not on the system PATH, add it like this:
    # path = '/path/to/ffmpeg/'
    # os.environ['PATH'] += os.pathsep + path


    file = "Lamb - Five.mp3"
    #file = "Acrassicauda_-_02_-_Garden_Of_Stones.wav"
    samplerate, samplewidth, wavedata = audiofile_read(file)

    print "Successfully read audio file:"
    print samplerate, "Hz,", samplewidth*8, "bit,", wavedata.shape[1], "channels,", wavedata.shape[0], "samples"
