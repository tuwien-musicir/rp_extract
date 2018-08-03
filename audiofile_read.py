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



def get_audioformat_info(input_file, split_list = True):
    import magic    # pip install python-magic
    from string import split
    info = magic.from_file(input_file)
    if split_list:
        info = split(info, ', ')
    return info



# Normalize integer WAV data to float in range (-1,1)
# Note that this works fine with Wav files read with Wavio
# when using scipy.io.wavfile to read Wav files, use divisor = np.iinfo(wavedata.dtype).max + 1
# but in this case it will not work with 24 bit files due to scipy scaling 24 bit up to 32bit
def normalize_wav(wavedata,samplewidth):

    # samplewidth in byte (i.e.: 1 = 8bit, 2 = 16bit, 3 = 24bit, 4 = 32bit)
    divisor  = 2**(8*samplewidth)/2
    wavedata = wavedata / float(divisor)
    return (wavedata)



def wav_read(filename,normalize=True,verbose=True,auto_resample=True):
    '''read WAV files

    :param filename: input filename to read from
    :param normalize: normalize the read values (usually signed integers) to range (-1,1)
    :param verbose: output some information during reading
    :param auto_resample: auto-resampling: if sample rate is different than 11, 22 or 44 kHz it will resample to 44 khZ
    :return: tuple of 3 elements: samplereate (e.g. 44100), samplewith (e.g. 2 for 16 bit) and wavedata (simple array for mono, 2-dim. array for stereo)
    '''

    # check if file exists
    if not os.path.exists(filename):
        raise NameError("File does not exist: " + filename)

    samplerate, samplewidth, wavedata = wavio.readwav(filename)

    if auto_resample and samplerate != 11025 and samplerate != 22050 and samplerate != 44100:
        #print original file info
        if verbose:
            print samplerate, "Hz,", wavedata.shape[1], "channel(s),", wavedata.shape[0], "samples"

        to_samplerate = 22050 if samplerate < 22050 else 44100
        filename2 = resample(filename, to_samplerate, normalize=True, verbose=verbose)
        samplerate, samplewidth, wavedata = wavio.readwav(filename2)
        os.remove(filename2) # delete temp file

    if (normalize):
        wavedata = normalize_wav(wavedata,samplewidth)

    return (samplerate, samplewidth, wavedata)


def get_temp_filename(suffix=None):
    
    temp_dir      = tempfile.gettempdir()
    rand_filename = str(uuid.uuid4())

    if suffix != None:
        rand_filename = "%s%s" % (rand_filename, suffix)
        
    return os.path.join(temp_dir, rand_filename)


def resample(filename, to_samplerate=44100, normalize=True, verbose=True):

    tempfile = get_temp_filename(suffix='.wav')

    try:
        cmd = ['ffmpeg','-v','1','-y','-i', filename, '-ar', str(to_samplerate), tempfile]
        if verbose:
            print "Resampling to", to_samplerate, "..."
            #print " ".join(cmd)

        return_code = subprocess.call(cmd)  # subprocess.call takes a list of command + arguments

        if return_code != 0:
            raise DecoderException("Problem appeared during resampling.", command=cmd)
        #if verbose: print 'Resampled with:', " ".join(cmd)

    except OSError as e:
        if os.path.exists(tempfile):
            os.remove(tempfile)

        if e.errno == 2:     # probably ffmpeg binary not found
            try:
                subprocess.call(cmd[0])  # check if we can just call the binary
            except OSError as e:
                raise DecoderException("Decoder not found. Please install " + cmd[0], command=cmd, orig_error=e)

        raise DecoderException("Unknown problem appeared during resampling.", command=cmd, orig_error=e)

    return tempfile



def mp3_decode(in_filename, out_filename=None, verbose=True):
    ''' mp3_decode

    decoding of MP3 files

    now handled by decode function (for parameters see there)
    kept for code compatibility
    '''
    return decode(in_filename, out_filename, verbose)


def decode(in_filename, out_filename=None, verbose=True, no_extension_check=False, force_mono=False, force_resampling=None):
    ''' calls external decoder to convert an MP3, FLAC, AIF(F) or M4A file to a WAV file

    One of the following decoder programs must be installed on the system:

    ffmpeg: for mp3, flac, aif(f), or m4a
    mpg123: for mp3
    lame: for mp3

    (consider adding their path  using os.environ['PATH'] += os.pathsep + path )

    in_filename: input audio file name to process
    out_filename: output filename after conversion; if omitted, the input filename is used, replacing the extension by .wav
    verbose: print decoding command line information or not
    no_extension_check: does not check file format extension. means that *first* specified decoder is called on ANY files type
    force_mono: force mono output when decoding (works with FFMPEG only!)
    force_resampling: force a target sampling rate (provided in Hz) when decoding (works with FFMPEG only!)
    returns: decoder command used (without parameters)
    '''

    basename, ext = os.path.splitext(in_filename)
    ext = ext.lower()

    if out_filename == None:
        out_filename = basename + '.wav'

    # check a number of external MP3 decoder tools whether they are available on the system

    # for subprocess.call, we prepare the commands and the arguments as a list
    # cmd_list is a list of commands with their arguments, which will be iterated over to try to find each tool
    # cmd_types is a list of file types supported by each command/tool

    cmd1 = ['ffmpeg','-v','1','-y','-i', in_filename] # -v adjusts log level, -y option overwrites output file, because it has been created already by tempfile before when passed
    if force_resampling: cmd1.extend(['-ar',str(force_resampling)])  # add option to resample to targate Hz
    if force_mono: cmd1.extend(['-ac','1'])  # add option to force to mono (1 output channel)
    cmd1.append(out_filename)
    cmd1_types = ['.mp2','.mp3','.mp4','.m4a','.aif','.aiff','.flac']

    cmd2 = ['mpg123','-q', '-w', out_filename, in_filename]
    cmd2_types = ['.mp3']

    cmd3 = ['lame','--quiet','--decode', in_filename, out_filename]
    cmd3_types = ['.mp3']

    cmd_list = [cmd1,cmd2,cmd3]
    cmd_types = [cmd1_types,cmd2_types,cmd3_types]

    success = False

    for cmd, types in zip(cmd_list,cmd_types):

        # we decode only if the current command supports the file type that we are having (except no_extension_check is True)
        if ext in types or no_extension_check:
            try:
                return_code = subprocess.call(cmd)  # subprocess.call takes a list of command + arguments

                if return_code != 0:
                    raise DecoderException("Problem appeared during executing decoder. Return_code: " + str(return_code), command=cmd)
                if verbose: print 'Decoded', ext, 'with:', " ".join(cmd)
                success = True

            except OSError as e:
                if e.errno != 2: #  2 = No such file or directory (i.e. decoder not found, which we want to catch at the end below)
                    raise DecoderException("Problem appeared during decoding: " + str(e), command=cmd, orig_error=e)

        if success:
            break  # no need to loop further

    if not success:
        commands = ", ".join( c[0] for c in cmd_list)
        raise OSError("No appropriate decoder found for " + ext + " file. Check if any of these programs is on your system path: " + commands + \
                       ". Otherwise install one of these and/or add them to the path using os.environ['PATH'] += os.pathsep + path.")

    return cmd[0]

def decode_video(in_filename, out_filename=None, verbose=False, no_extension_check=False, force_mono=False):
    ''' calls external decoder to extract the audio stream from a video file and to store it to a WAV file (works with FFMPEG only!)

    The following decoder program must be installed on the system:

    ffmpeg: for all relevant video formats
    
    (consider adding their path  using os.environ['PATH'] += os.pathsep + path )

    in_filename: input video file name to process
    out_filename: output filename after conversion; if omitted, the input filename is used, replacing the extension by .wav
    verbose: print decoding command line information or not
    no_extension_check: does not check file format extension. means that *first* specified decoder is called on ANY files type
    force_mono: force mono output when decoding 
    force_resampling: force a target sampling rate (provided in Hz) when decoding 
    returns: decoder command used (without parameters)
    '''

    basename, ext = os.path.splitext(in_filename)
    ext = ext.lower()

    if out_filename == None:
        out_filename = basename + '.wav'

    # check a number of external MP3 decoder tools whether they are available on the system

    # for subprocess.call, we prepare the commands and the arguments as a list
    # cmd_list is a list of commands with their arguments, which will be iterated over to try to find each tool
    # cmd_types is a list of file types supported by each command/tool

    cmd = ['ffmpeg','-i', in_filename,'-acodec','pcm_s16le','-ar','44100'] # -v adjusts log level, -y option overwrites output file, because it has been created already by tempfile before when passed
    if force_mono: cmd.extend(['-ac','1'])  # add option to force to mono (1 output channel)
    else: cmd.extend(['-ac','2'])
    cmd.append(out_filename)

    try:
        return_code = subprocess.call(cmd)  # subprocess.call takes a list of command + arguments

        if return_code != 0:
            raise DecoderException("Problem appeared during executing decoder. Return_code: " + str(return_code), command=cmd)
        if verbose: print 'Decoded', ext, 'with:', " ".join(cmd)
        success = True

    except OSError as e:
        if e.errno != 2: #  2 = No such file or directory (i.e. decoder not found, which we want to catch at the end below)
            raise DecoderException("Problem appeared during decoding: " + str(e), command=cmd, orig_error=e)

    return cmd


def get_supported_audio_formats():
    # TODO: update this list here every time a new format is added; to avoid this, make a more elegant solution getting the list of formats from where the commands are defined above
    return ('.wav','.mp2','.mp3','.mp4','.m4a','.aif','.aiff','.flac','.au')


# testing decoding to memory instead of file: did NOT bring any speedup!
# Also note:  sample rate and number of channels not returned with this method. can be derived with
# ffprobe -v quiet -show_streams -of json <input_file>
# which already converts plain text to json, but then the json needs to be parsed.
def decode_to_memory(in_filename, verbose=True):
    cmd1 = ['ffmpeg','-v','1','-y','-i', in_filename, "-f", "f32le", "pipe:1"]    # -v adjusts log level, -y option overwrites output file, because it has been created already by tempfile above
    # "pipe:1" sends output to std_out (probably Linux only)
    # original:  call = [cmd, "-v", "quiet", "-i", infile, "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "pipe:1"]
    # for Windows: \\.\pipe\from_ffmpeg  # http://stackoverflow.com/questions/32157774/ffmpeg-output-pipeing-to-named-windows-pipe
    cmd1_types = ('.mp3','.aif','.aiff','.m4a')

    ext = ''
    if verbose: print 'Decoding', ext, 'with:', " ".join(cmd1)

    import numpy as np
    decoded_wav = subprocess.check_output(cmd1)
    wavedata = np.frombuffer(decoded_wav, dtype=np.float32)
    return wavedata


def mp3_read(filename,normalize=True,verbose=True):
    ''' mp3_read:
    call mp3_decode and read from wav file ,then delete wav file
    returns samplereate (e.g. 44100), samplewith (e.g. 2 for 16 bit) and wavedata (simple array for mono, 2-dim. array for stereo)
    '''

    try:
        tempfile = get_temp_filename(suffix='.wav')
        decode(filename,tempfile,verbose)
        samplerate, samplewidth, wavedata = wav_read(tempfile,normalize,verbose)

    finally: # delete temp file

        if os.path.exists(tempfile):
            os.remove(tempfile)

    return (samplerate, samplewidth, wavedata)


def videofile_read(filename,normalize=True,verbose=False,include_decoder=False,no_extension_check=False):
    ''' audiofile_read

    generic function capable of reading audio from video files

    :param filename: file name path to video file
    :param normalize: normalize to (-1,1) if True (default), or keep original values (16 bit, 24 bit or 32 bit)
    :param verbose: whether to print a message while decoding files or not
    :param include_decoder: includes a 4th return value: string which decoder has been used to decode the audio file
    :param no_extension_check: does not check file format via extension. means that decoder is called on ALL files.
    :param force_resampling: force a target sampling rate (provided in Hz) when decoding (works with FFMPEG only!)
    :return: a tuple with 3 or 4 entries: samplerate in Hz (e.g. 44100), samplewidth in bytes (e.g. 2 for 16 bit),
            wavedata (simple array for mono, 2-dim. array for stereo), and optionally a decoder string

    Example:
    >>> samplerate, samplewidth, wavedata = videofile_read("music/BoxCat_Games_-_10_-_Epic_Song.mp4",verbose=False)
    >>> print samplerate, "Hz,", samplewidth*8, "bit,", wavedata.shape[1], "channels,", wavedata.shape[0], "samples"
    44100 Hz, 16 bit, 2 channels, 2421504 samples

    '''

    # check if file exists or has 0 bytes
    if not os.path.exists(filename):
        raise NameError("File does not exist: " + filename)
    if os.path.getsize(filename) == 0:
        raise ValueError("File has 0 bytes: " + filename)

    basename, ext = os.path.splitext(filename)
    ext = ext.lower()


    try: # try to decode
        tempfile = get_temp_filename(suffix='.wav')
        decoder  = decode_video(filename,tempfile,verbose,no_extension_check)
        samplerate, samplewidth, wavedata = wav_read(tempfile,normalize,verbose)

    finally: # delete temp file in any case
        if os.path.exists(tempfile):
            os.remove(tempfile)

    if include_decoder:
        return samplerate, samplewidth, wavedata, decoder
    else:
        return samplerate, samplewidth, wavedata


def audiofile_read(filename,normalize=True,verbose=True,include_decoder=False,no_extension_check=False,force_resampling=None):
    ''' audiofile_read

    generic function capable of reading WAV, MP3 and AIF(F) files

    :param filename: file name path to audio file
    :param normalize: normalize to (-1,1) if True (default), or keep original values (16 bit, 24 bit or 32 bit)
    :param verbose: whether to print a message while decoding files or not
    :param include_decoder: includes a 4th return value: string which decoder has been used to decode the audio file
    :param no_extension_check: does not check file format via extension. means that decoder is called on ALL files.
    :param force_resampling: force a target sampling rate (provided in Hz) when decoding (works with FFMPEG only!)
    :return: a tuple with 3 or 4 entries: samplerate in Hz (e.g. 44100), samplewidth in bytes (e.g. 2 for 16 bit),
            wavedata (simple array for mono, 2-dim. array for stereo), and optionally a decoder string

    Example:
    >>> samplerate, samplewidth, wavedata = audiofile_read("music/BoxCat_Games_-_10_-_Epic_Song.mp3",verbose=False)
    >>> print samplerate, "Hz,", samplewidth*8, "bit,", wavedata.shape[1], "channels,", wavedata.shape[0], "samples"
    44100 Hz, 16 bit, 2 channels, 2421504 samples

    '''

    # check if file exists or has 0 bytes
    if not os.path.exists(filename):
        raise NameError("File does not exist: " + filename)
    if os.path.getsize(filename) == 0:
        raise ValueError("File has 0 bytes: " + filename)

    basename, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == '.wav' and not no_extension_check==True:
        samplerate, samplewidth, wavedata = wav_read(filename,normalize,verbose)
        decoder = 'wavio.py' # for log file
    else:
        try: # try to decode
            tempfile = get_temp_filename(suffix='.wav')
            decoder = decode(filename,tempfile,verbose,no_extension_check,force_resampling=force_resampling)
            samplerate, samplewidth, wavedata = wav_read(tempfile,normalize,verbose)

        finally: # delete temp file in any case
            if os.path.exists(tempfile):
                os.remove(tempfile)

    if include_decoder:
        return samplerate, samplewidth, wavedata, decoder
    else:
        return samplerate, samplewidth, wavedata


# function to self test audiofile_read if working properly
def self_test():
    import doctest
    #doctest.testmod()
    doctest.run_docstring_examples(audiofile_read, globals())


# main routine: to test if decoding works properly

if __name__ == '__main__':

    # to run self test:
    #self_test()
    #exit()
    # (no output means that everything went fine)

    import sys

    # if your MP3 decoder is not on the system PATH, add it like this:
    # path = '/path/to/ffmpeg/'
    # os.environ['PATH'] += os.pathsep + path
    
    # test audio file: "Epic Song" by "BoxCat Game" (included in repository)
    # Epic Song by BoxCat Games is licensed under a Creative Commons Attribution License.
    # http://freemusicarchive.org/music/BoxCat_Games/Nameless_the_Hackers_RPG_Soundtrack/BoxCat_Games_-_Nameless-_the_Hackers_RPG_Soundtrack_-_10_Epic_Song
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        file = "music/BoxCat_Games_-_10_-_Epic_Song.mp3"

    #print get_audioformat_info(file)

    # import time
    # start = time.time()
    samplerate, samplewidth, wavedata = audiofile_read(file)
    # print time.time() - start
    # print wavedata.shape
    #
    # start = time.time()
    # wavedata2 = decode_to_memory(file)
    # print time.time() - start
    # print wavedata2.shape
    #
    # print "EQUAL" if wavedata == wavedata2 else "NOT EQUAL"

    print "Successfully read audio file:"
    print samplerate, "Hz,", samplewidth*8, "bit,", wavedata.shape[1], "channels,", wavedata.shape[0], "samples"