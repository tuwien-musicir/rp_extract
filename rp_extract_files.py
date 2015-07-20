# RP_EXTRACT_FILES:

# wrapper around rp_extract.py to sequentially extract features from all files in a given directory

# 2015-04, 2015-06 by Thomas Lidy

from audiofile_read import * # reading wav and mp3 files
import rp_extract as rp # Rhythm Pattern extractor

import unicsv # unicode csv library (installed via pip install unicsv)
import time # for time measuring
# import pandas as pd # only needed in read_feature_files -> import has been moved there


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

def read_feature_files(filenamestub,ext,separate_ids=True,id_column=0):

    import numpy as np
    import pandas as pd

    # initialize empty dicts
    feat = {}
    ids = {}

    for e in ext:
        filename = filenamestub + "." + e

        # we use pandas to import CSV as pandas dataframe,
        # because it handles quoted filnames (containing ,) well (by contrast to other CSV readers)
        dataframe = pd.read_csv(filename, sep=',',header=None)

        # convert to numpy matrix/array
        feat[e] = dataframe.as_matrix(columns=None)

        if separate_ids:
           ids[e] = feat[e][:,id_column]
           feat[e] = np.delete(feat[e],id_column,1)

        print "Read:", e,":\t", feat[e].shape[0], "vectors", feat[e].shape[1], "dimensions (excl. id)"

    if separate_ids:
        return(ids,feat)
    else:
        return feat




def extract_all_files_in_path(path,out_file,feature_types):

    ext = feature_types

    files, writer = initialize_feature_files(out_file,ext)

    # iterate through all files

    start_abs = time.time()

    n = 0  # counting the files that were actually analyzed

    for d in os.walk(path):    # finds all subdirectories and gets a list of files therein
        print path
        subpath = d[0]
        # dir_list = d[1]
        filelist = d[2]
        print subpath, len(filelist), "files found (any file type)"

        filelist2 = [ file for file in filelist if not file.lower().endswith( ('.wav','.mp3') ) ]
        print subpath, len(filelist2), "files found (wav or mp3)."
        exit()

        for fil in filelist:  # iterate over all files in a dir
            try:

                # restrict to mp3 and wav files (remove this if and unindent all subsequent code in case you do not want to check for file extension)
                if os.path.splitext(fil)[1].lower() in ['.wav','.mp3']:

                    n += 1
                    filename = subpath + os.sep + fil
                    print '#',n,':', filename

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

                    #type(feat["rp"])
                    #numpy.ndarray

                    #print feat["rp"].shape
                    #(1440,)

                    # WRITE each features to a CSV

                    # TODO check if ext and feat.keys are consistent

                    start = time.time()

                    # id = fil -> add filename before vector (to include path, change fil to filename)
                    id = fil # filename
                    write_feature_files(id,feat,writer)

                    end = time.time()

                    print "Data written." #, end-start
            except:
                print "Error analysing file: " + file

    # close all output files

    close_feature_files(files,ext)

    end = time.time()

    print "FEATURE EXTRACTION FINISHED.", n, "files,", end-start_abs, "sec"



if __name__ == '__main__':

    # SET WHICH FEATURES TO EXTRACT (must be lower case)

    #feature_types = ['rp','ssd','rh','mvd'] # sh, tssd, trh
    feature_types = ['rp','ssd','rh','mvd','tssd','trh']
    feature_types = ['rh']

    # SET PATH WITH AUDIO FILES (INPUT)

    in_path = "/data/music/GTZAN/wav"

    # OUTPUT FEATURE FILES

    out_path = '/data/music/GTZAN/vec'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    out_file = 'dummytest'

    out_filename = out_path + os.sep + out_file

    extract_all_files_in_path(in_path,out_filename,feature_types)

    # EXAMPLE ON HOW TO READ THE FEATURE FILES

    # filenamestub = out_filename
    # filenamestub = 'features'
    # ext = ['rh']
    # ids, features = read_feature_files(filenamestub,ext)

