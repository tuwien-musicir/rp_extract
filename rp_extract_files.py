# RP_EXTRACT_FILES:

# wrapper around rp_extract_python.py to sequentially extract features from all files in a given directory

# 2015-04 by Thomas Lidy

import csv
import unicsv # unicode csv library (installed via pip)
import sys
import os # for mp3 decoding
# import pandas as pd
from scipy.io import wavfile
import time # for time measuring

#sys.path.append('../RP_extract_python')
import rp_extract_python as rp

#np.set_printoptions(suppress=True)


# MP3 DECODING:
# as there is no Python library for it, we need to use external tools (mpg123, lame, ffmpeg)

def mp3_read(filename):
    cmd = 'mpg123'
    tempfile = "temp.wav"
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


# SET WHICH FEATURES TO EXTRACT

ext = ['rp','ssd','rh','mvd'] # sh, tssd, trh   # must be lower case due to check below
#ext = ['rh'] # testing

# INITIALIZE OUTPUT FILES

out_path = 'feat'
if not os.path.exists(out_path):
    os.mkdir(out_path)

out_file = 'features'
files = {}  # files is a dict of one file handle per extension
writer = {} # files is a dict of one file writer per extension

for e in ext:
    filename = out_path + os.sep + out_file + '.' + e
    files[e] = open(filename, 'w') # or 'a' to append
    writer[e] = unicsv.UnicodeCSVWriter(files[e]) #, quoting=csv.QUOTE_ALL)


# iterate through all files

start_abs = time.time()

# TODO: restrict to *.mp3

main_dir = "./mp3" #/Rock"
n=0

for d in os.walk(main_dir):    # finds all subdirectories and gets a list of files therein
    path = d[0]
    # dir_list = d[1]
    filelist = d[2]
    print path, len(filelist), "files"
    

    for fil in filelist:  # iterate over all files in a dir
        n += 1
        filename = path + os.sep + fil
        print '#',n,':', filename

        start = time.time()

        # read MP3
        fs, data = mp3_read(filename)

        end = time.time()
        print end - start

        # audio file info
        print fs, "Hz,", data.shape[1], "channels,", data.shape[0], "length"

        start = time.time()

        # convert data space to MATLAB value range
        data = data / float(32768)

        # extract features
        # Note: the True/False flags are determined by checking if a feature is listed in 'ext' (see settings above)

        feat = rp.rp_extract(data,
                          fs, # / 1,
                          extract_rp   = ('rp' in ext),          # extract Rhythm Patterns features
                          extract_ssd  = ('ssd' in ext),           # extract Statistical Spectrum Descriptor
                          extract_sh   = ('sh' in ext),          # extract Statistical Histograms
                          extract_tssd = ('tssd' in ext),          # extract temporal Statistical Spectrum Descriptor
                          extract_rh   = ('rh' in ext),           # extract Rhythm Histogram features
                          extract_trh  = ('trh' in ext),          # extract temporal Rhythm Histogram features
                          extract_mvd  = ('mvd' in ext),        # extract Modulation Frequency Variance Descriptor 
                          spectral_masking=True,
                          transform_db=True,
                          transform_phon=True,
                          transform_sone=True,
                          fluctuation_strength_weighting=True,
                          skip_leadin_fadeout=1,
                          step_width=1)

        end = time.time()

        print "Features extracted:", feat.keys(), end - start

        #type(feat["rp"])
        #numpy.ndarray

        #print feat["rp"].shape
        #(1440,)

        # WRITE each features to a CSV

        # TODO check if ext and feat.keys are consistent

        start = time.time()

        for e in ext: # or feat.keys()
            f=feat[e].tolist()  
            f.insert(0,fil)        # add filename before vector (to include path, change fil to filename)
            writer[e].writerow(f)

        end = time.time()

        print "Data written", end-start



# OPTIONAL PLOTTING

## This is how to RESHAPE in case needed
# rpf = feat["rp"].reshape(24,60,order='F')  # order='F' means Fortran compatible; Alex uses it in rp_extract flatten() to be Matlab compatible
# print rpf.shape
# plotmatrix(rpf)

# ssd = feat["ssd"].reshape(24,7,order='F')  # order='F' means Fortran compatible; Alex uses it in rp_extract flatten() to be Matlab compatible
# print ssd.shape

# plotssd(ssd)

# plotrh(feat["rh"])


# close all output files

for e in ext:
    files[e].close()

    
end = time.time()
    
print "FEATURE EXTRACTION FINISHED.", end-start_abs

