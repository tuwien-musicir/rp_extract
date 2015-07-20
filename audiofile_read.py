# 2015-04 by Thomas Lidy

# MP3 READ: mini function to decode mp3 using external program
# as there is no Python library for it, we need to use external tools (mpg123, lame, ffmpeg)

import os         # for calling external program for mp3 decoding
import subprocess # for subprocess calls
import tempfile
import uuid

# Reading WAV files
# from scipy.io import wavfile
# scipy.io.wavfile does not support 24 bit Wav files
# therefore we switch to wavio by Warren Weckesser - https://github.com/WarrenWeckesser/wavio - BSD 3-Clause License
import wavio



class DecoderException(Exception):
    
    def __init__(self, message, command=[], orig_error=None):

        # Call the base class constructor with the parameters it needs
        super(DecoderException, self).__init__(message)
        self.command        = command
        self.original_error = orig_error




# Normalize integer WAV data to float in range (-1,1)
# Note that this works fine with Wav files read with Wavio
# when using scipy.io.wavfile to read Wav files, use divisor = np.iinfo(wavedata.dtype).max + 1
# but in this case it will not work with 24 bit files due to scipy scaling 24 bit up to 32bit
def normalize_wav(wavedata,samplewidth):

    # samplewidth in byte (i.e.: 1 = 8bit, 2 = 16bit, 3 = 24bit, 4 = 32bit)
    divisor  = 2**(8*samplewidth)/2
    wavedata = wavedata / float(divisor)
    return (wavedata)


# read wav files
# returns samplereate (e.g. 44100), samplewith (e.g. 2 for 16 bit) and wavedata (simple array for mono, 2-dim. array for stereo)

def wav_read(filename,normalize=True):

    # check if file exists
    if not os.path.exists(filename):
        raise NameError("File does not exist:" + filename)

    samplerate, samplewidth, wavedata = wavio.readwav(filename)

    if (normalize):
        wavedata = normalize_wav(wavedata,samplewidth)

    return (samplerate, samplewidth, wavedata)


def get_temp_filename(suffix=None):
    
    temp_dir      = tempfile.gettempdir()
    rand_filename = str(uuid.uuid4())

    if suffix != None:
        rand_filename = "%s%s" % (rand_filename, suffix)
        
    return os.path.join(temp_dir, rand_filename)

# convert mp3 to wav and read from wav file
# returns samplereate (e.g. 44100), samplewith (e.g. 2 for 16 bit) and wavedata (simple array for mono, 2-dim. array for stereo)

def mp3_read(filename,normalize=True):

    temp = get_temp_filename(suffix='.wav')

    # check a number of external MP3 decoder tools whether they are available on the system

    # for subprocess.call, we prepare the commands and the arguments as a list
    # cmd_list is a list of commands with their arguments, which will be iterated over to try to find each tool
    cmd_list = []

    cmd1 = ['mpg123','-q', '-w', temp, filename]
    cmd2 = ['ffmpeg','-v','1','-y','-i', filename,  temp]    # -v adjusts log level, -y option overwrites output file, because it has been created already by tempfile above
    cmd3 = ['lame','--quiet','--decode', filename, temp]

    cmd_list.append(cmd1)
    cmd_list.append(cmd2)
    cmd_list.append(cmd3)

    success = False

    try:

        for cmd in cmd_list:
            
            try:
    
                print cmd
                return_code = subprocess.call(cmd)  # subprocess.call takes a list of command + arguments
                
                if return_code != 0:
                    raise DecoderException("Problem appeared during decoding.", command=cmd)
                #print 'Decoding mp3 with: ', " ".join(cmd)
                samplerate, samplewidth, wavedata = wav_read(temp,normalize)
    
                success = True
    
            except OSError as e:
                print e.errno
                if e.errno != 2: #  2 = No such file or directory (i.e. decoder not found, which we want to catch at the end below)
                    raise DecoderException("Problem appeared during decoding.", cmd=cmd, orig_error=e)
                
            if success:
                break  # no need to loop further
            
    finally:
       
       if os.path.exists(temp):
           os.remove(temp)
            

    if not success:
        commands = ", ".join( c[0] for c in cmd_list)
        raise OSError("No MP3 decoder found. Check if any of these is on your system path: " + commands + \
                       " and if not add the path using os.environ['PATH'] += os.pathsep + path.")

    return (samplerate, samplewidth, wavedata)


# generic function capable of reading both .wav and .mp3 files
# returns samplereate (e.g. 44100), samplewith (e.g. 2 for 16 bit) and wavedata (simple array for mono, 2-dim. array for stereo)

def audiofile_read(filename,normalize=True):

    # check if file exists
    if not os.path.exists(filename):
        raise NameError("File does not exist:" + filename)

    basename, ext = os.path.splitext(filename)

    if ext.lower() == '.wav':
        return(wav_read(filename,normalize))
    elif ext.lower() == '.mp3':
        return(mp3_read(filename,normalize))
    else:
        raise NameError("File name extension must be either .wav or .mp3 when using audiofile_read. Extension found: " + ext)


# main routine: to test if decoding works properly

if __name__ == '__main__':

    # if your MP3 decoder is not on the system PATH, add it like this:
    # path = '/path/to/ffmpeg/'
    # os.environ['PATH'] += os.pathsep + path
    
    # test audio file: "Epic Song" by "BoxCat Game"
    # Epic Song by BoxCat Games is licensed under a Creative Commons Attribution License.
    # http://freemusicarchive.org/music/BoxCat_Games/Nameless_the_Hackers_RPG_Soundtrack/BoxCat_Games_-_Nameless-_the_Hackers_RPG_Soundtrack_-_10_Epic_Song
    file = "E:/Data/MIR/EU_SOUNDS/2023601/oai_eu_dismarc_ARTP_GBCRW0158204.mp3"
    
    samplerate, samplewidth, wavedata = audiofile_read(file)

    print "Successfully read audio file:"
    print samplerate, "Hz,", samplewidth*8, "bit,", wavedata.shape[1], "channels,", wavedata.shape[0], "samples"
