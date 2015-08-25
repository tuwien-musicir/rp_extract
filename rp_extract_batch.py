''' RP_Extract_Batch

(c) 2015 by Thomas Lidy

Batch extraction of RP features:
  wrapper around rp_extract.py to sequentially extract features from all audio files in a given directory
  and store them into CSV feature files

Batch MP3 to WAV conversion
  use one of three external decoders to batch convert folders with mp3 to wav files
'''

import os
import unicsv # unicode csv library (installed via pip install unicsv)
import time # for time measuring

from audiofile_read import * # reading wav and mp3 files
import rp_extract as rp # Rhythm Pattern extractor


# function to find all files of a particular file type in a given path
# path: input path to start searching
# file_types: a tuple of file extensions (e.g.'.wav','.mp3') (case-insensitive) or 'None' in which case ALL files in path will be returned
# relative_path: if False, absolute paths will be returned, otherwise the path relative to the given path

def find_files(path,file_types=('.wav','.mp3'),relative_path = False,verbose=False):

    if path.endswith(os.sep):
        path = path[0:-1]   # we need to remove the file separator at the end otherwise the path handling below gets confused

    # lower case the file types for comparison
    if file_types: # if we have file_types (otherwise 'None')
        if type(file_types) == tuple:
            file_types = tuple((f.lower() for f in file_types))
            file_type_string = ' or '.join(file_types) # for print message only
        else: # single string
            file_types = file_types.lower()
            file_type_string = file_types # for print message only
    else:
        file_type_string = 'any file type'  # for print message only

    all_files = []

    for d in os.walk(path):    # finds all subdirectories and gets a list of files therein
        # subpath: complete sub directory path (full path)
        # filelist: files in that sub path (filenames only)
        (subpath, _, filelist) = d

        if file_types:   # FILTER FILE LIST by FILE TYPE
            filelist = [ file for file in filelist if file.lower().endswith(file_types) ]

        if (verbose): print subpath,":", len(filelist), "files found (" + file_type_string + ")"

        # add full absolute path
        filelist = [ subpath + os.sep + file for file in filelist ]

        if relative_path: # cut away full path at the beginning (+/- 1 character depending if path ends with path separator)
            filelist = [ filename[len(path)+1:] for filename in filelist ]

        all_files.extend(filelist)

    return all_files


# these 3 functions are to incrementally save the features line by line into CSV files

def initialize_feature_files(base_filename,ext,append=False):
    files = {}  # files is a dict of one file handle per extension
    writer = {} # files is a dict of one file writer per extension

    if append:
        mode = 'a' # append
    else:
        mode = 'w' # write new (will overwrite)

    for e in ext:
        filename = base_filename + '.' + e
        files[e] = open(filename, mode)
        writer[e] = unicsv.UnicodeCSVWriter(files[e]) #, quoting=csv.QUOTE_ALL)

    return (files,writer)


def write_feature_files(id,feat,writer,id2=None):
    # id: string id (e.g. filename) of extracted file
    # feat: dict containing 1 entry per feature type (must match file extensions)

    for e in feat.keys():
        f=feat[e].tolist()
        f.insert(0,id)        # add filename before vector (to include path, change fil to filename)
        if not id2==None:
            f.insert(1,id2)
        writer[e].writerow(f)


def close_feature_files(files,ext):
    for e in ext:
        files[e].close()


# read_feature_files:
# reads pre-analyzed features from CSV files
# in write_feature_files we use unicsv to store the features
# it will quote strings containing , and other characters if needed only (automatically)
# here we use pandas to import CSV as a pandas dataframe,
# because it handles quoted filenames (containing ,) well (by contrast to other CSV readers)

# parameters:
# filenamestub: full path to feature file name WITHOUT .extension
# ext: a list of file .extensions (e.g. 'rh','ssd','rp') to be read in
# separate_ids: if False, it will return a single matrix containing the id column
#               if True, it will return a tuple: (ids, features) separating the id column from the features
# id_column: which of the CSV columns contains the ids (default = 0, i.e. first column)
#
# returns: single numpy matrix including ids, or tuple of (ids, features) with ids and features separately
#          each of them is a python dict containing an entry per feature extension (ext)

# NOTE: this function has been moved to rp_feature_io.py and is maintained here for backwards compatibility

def read_feature_files(filenamestub,ext,separate_ids=True,id_column=0):
    from rp_feature_io import read_csv_features
    return read_csv_features(filenamestub,ext,separate_ids,id_column)



# mp3_to_wav_batch:
# finds all MP3s in a given directory in all subdirectories
# and converts all of them to WAV
# if outdir is specified it will replicate the entire subdir structure from within input path to outdir
# otherwise the WAV file will be created in the same dir as the MP3 file
# in both cases the file name is maintained and the extension changed to .wav

def mp3_to_wav_batch(path,outdir=None):

    get_relative_path = (outdir!=None) # if outdir is specified we need relative path otherwise absolute

    filenames = find_files(path,'.mp3',get_relative_path)

    n_files = len(filenames)
    n = 0

    for file in filenames:

        n += 1
        wav_file = os.path.splitext(file)[0] + '.wav'

        if outdir: # if outdir is specified we add it in front of the relative file path
            file = path + os.sep + file
            wav_file = outdir + os.sep + wav_file

            # recreate same subdir path structure as in input path
            out_subpath = os.path.split(wav_file)[0]

            if not os.path.exists(out_subpath):
                os.makedirs(out_subpath)

        # future option: (to avoid recreating the input path subdir structure in outdir)
        #filename_only = os.path.split(wav_file)[1]

        try:
            if not os.path.exists(wav_file):
                print "Decoding:", n, "/", n_files, ":"
                mp3_decode(file,wav_file)
            else:
                print "Already existing: " + wav_file
        except:
            print "Not decoded " + file



# finds all files of a certain type (e.g. .wav and/or .mp3) in a path and all sub directories in it
# extracts selected RP feature types
# and saves them into separate CSV feature files (one per feature type)

# path: input file path to search for audio files (including subdirectories)
# out_file: output file name stub for feature files to write
# feature_types: RP feature types to extract. see rp_extract.py
# audiofile_types: a string or tuple of suffixes to look for file extensions to consider (include the .)


def extract_all_files_in_path(path,out_file,feature_types,audiofile_types=('.wav','.mp3')):

    ext = feature_types

    # get file list of all files in a path (filtered by audiofile_types)
    filelist = find_files(path,audiofile_types,relative_path=True)

    n = 0  # counting the files that were actually analyzed
    err = 0 # counting errors
    n_files = len(filelist)

    start_abs = time.time()

    files, writer = initialize_feature_files(out_file,ext)

    for fil in filelist:  # iterate over all files
        try:

            n += 1
            filename = path + os.sep + fil
            print '#',n,'/',n_files,':', filename

            start = time.time()

            # read audio file (wav or mp3)
            samplerate, samplewidth, data = audiofile_read(filename)

            end = time.time()
            print end - start, "sec"

            # audio file info
            print samplerate, "Hz,", data.shape[1], "channels,", data.shape[0], "samples"

            # extract features
            # Note: the True/False flags are determined by checking if a feature is listed in 'ext' (see settings above)

            start = time.time()

            feat = rp.rp_extract(data,
                              samplerate,
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

            print "Features extracted:", feat.keys(), end - start, "sec"

            # WRITE each feature set to a CSV

            # TODO check if ext and feat.keys are consistent

            start = time.time()

            # add filename before vector. 3 choices:

            id = fil  # filename only
            # id = filename   # full filename incl. full path
            # id = filename[len(path)+1:] # relative filename only
            write_feature_files(id,feat,writer)

            end = time.time()

            print "Data written." #, end-start
        except:
            print "ERROR analysing file: " + fil
            err += 1

    # close all output files

    close_feature_files(files,ext)

    end = time.time()

    print "FEATURE EXTRACTION FINISHED.", n, "files,", end-start_abs, "sec"
    if err > 0:
        print err, "files had ERRORs during feature extraction."


# EXAMPLE CALL: please adapt to your needs (esp. in_path , out_path and out_file)

if __name__ == '__main__':

    # Example for MP3 to WAV batch conversion:
    # mp3_to_wav_batch('/data/music/ISMIRgenre/mp3_44khz_128kbit_stereo','/data/music/ISMIRgenre/wav')


    # EXAMPLE FOR BATCH RP FEATURE EXTRACTION:

    # SET WHICH FEATURES TO EXTRACT (must be lower case)

    #feature_types = ['rp','ssd','rh','mvd'] # sh, tssd, trh
    feature_types = ['rp','ssd','rh','mvd','tssd','trh']
    feature_types = ['rh']

    # SET PATH WITH AUDIO FILES (INPUT)

    in_path = "./music"

    # OUTPUT FEATURE FILES

    out_path = './feat'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    out_file = 'GTZAN_test'

    out_filename = out_path + os.sep + out_file

    extract_all_files_in_path(in_path,out_filename,feature_types)

    # EXAMPLE ON HOW TO READ THE FEATURE FILES

    filenamestub = out_file
    ext = feature_types

    in_filenamestub = out_path + os.sep + filenamestub

    ids, features = read_feature_files(in_filenamestub,ext)
    e = ext[0]
    print ids[e].shape
    print features[e].shape
    print ids