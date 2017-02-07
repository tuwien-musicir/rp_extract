#!/usr/bin/env python

''' RP_Extract_Batch

(c) 2015 by Thomas Lidy

Batch extraction of RP features:
  wrapper around rp_extract.py to sequentially extract features from all audio files in a given directory
  and store them into CSV feature files

Batch MP3 to WAV conversion
  use one of three external decoders to batch convert folders with mp3 to wav files
'''

import os
import gc # garbage collector
import unicsv # unicode csv library (installed via pip install unicsv)
import time # for time measuring
import datetime # for time printing
import argparse
import numpy as np

from audiofile_read import * # reading wav and mp3 files
from rp_feature_io import CSVFeatureWriter, HDF5FeatureWriter, read_csv_features, load_multiple_hdf5_feature_files
import rp_extract as rp # Rhythm Pattern extractor


# NOTE: this function has been moved to rp_feature_io.py and is maintained here for backwards compatibility

def read_feature_files(filenamestub,ext,separate_ids=True,id_column=0):
    from rp_feature_io import read_csv_features
    return read_csv_features(filenamestub,ext,separate_ids,id_column)


def timestr(seconds):
    ''' returns HH:MM:ss formatted time string for given seconds
    (seconds can be a float with milliseconds included, but only the integer part will be used)
    :return: string
    '''
    if seconds is None:
        return "--:--:--"
    else:
        return str(datetime.timedelta(seconds=int(seconds)))


def find_files(path,file_types=('.wav','.mp3'),relative_path = False,verbose=False,ignore_hidden=True):
    ''' function to find all files of a particular file type in a given path

    path: input path to start searching
    file_types: a tuple of file extensions (e.g.'.wav','.mp3') (case-insensitive) or 'None' in which case ALL files in path will be returned
    relative_path: if False, absolute paths will be returned, otherwise the path relative to the given path
    verbose: will print info about files found in path if True
    ignore_hidden: if True (default) will ignore Linux hidden files (starting with '.')
    '''

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

    for d in os.walk(unicode(path)):    # finds all subdirectories and gets a list of files therein
        # subpath: complete sub directory path (full path)
        # filelist: files in that sub path (filenames only)
        (subpath, _, filelist) = d

        if ignore_hidden:
            filelist = [ file for file in filelist if not file[0] == '.']

        if file_types:   # FILTER FILE LIST by FILE TYPE
            filelist = [ file for file in filelist if file.lower().endswith(file_types) ]

        if (verbose): print subpath,":", len(filelist), "files found (" + file_type_string + ")"

        # add full absolute path
        filelist = [ subpath + os.sep + file for file in filelist ]

        if relative_path: # cut away full path at the beginning (+/- 1 character depending if path ends with path separator)
            filelist = [ filename[len(path)+1:] for filename in filelist ]

        all_files.extend(filelist)

    return all_files



# mp3_to_wav_batch:
# finds all MP3s in a given directory in all subdirectories
# and converts all of them to WAV
# if outdir is specified it will replicate the entire subdir structure from within input path to outdir
# otherwise the WAV file will be created in the same dir as the MP3 file
# in both cases the file name is maintained and the extension changed to .wav

# Example for MP3 to WAV batch conversion (in a new Python script):
# from rp_extract_batch import mp3_to_wav_batch
# mp3_to_wav_batch('/data/music/ISMIRgenre/mp3_44khz_128kbit_stereo','/data/music/ISMIRgenre/wav')

def mp3_to_wav_batch(path,outdir=None,audiofile_types=('.mp3','.aif','.aiff')):

    get_relative_path = (outdir!=None) # if outdir is specified we need relative path otherwise absolute

    filenames = find_files(path,audiofile_types,get_relative_path)

    n_files = len(filenames)
    n = 0

    for file in filenames:

        n += 1
        basename, ext = os.path.splitext(file)
        wav_file = basename + '.wav'

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
                if ext.lower() == '.mp3':
                    mp3_decode(file,wav_file)
                elif ext.lower() == '.aif' or ext.lower() == '.aiff':
                    cmd = ['ffmpeg','-v','1','-y','-i', file,  wav_file]
                    return_code = subprocess.call(cmd)  # subprocess.call takes a list of command + arguments
                    if return_code != 0:
                        raise DecoderException("Problem appeared during decoding.", command=cmd)
            else:
                print "Already existing: " + wav_file
        except:
            print "Not decoded " + file





def extract_all_files_in_path(in_path,
                              out_file = None,
                              feature_types = ['rp','ssd','rh'],
                              audiofile_types=('.wav','.mp3'),
                              label=False,
                              verbose=True):
    """
    finds all files of a certain type (e.g. .wav and/or .mp3) in a path and all sub-directories in it
    extracts selected RP feature types
    and saves them into separate CSV feature files (one per feature type)


    # path: input file path to search for audio files (including subdirectories)
    # out_file: output file name stub for feature files to write (if omitted, features will be returned from function)
    # feature_types: RP feature types to extract. see rp_extract.py
    # audiofile_types: a string or tuple of suffixes to look for file extensions to consider (include the .)
    # label: use subdirectory name as class label
    """

    # get file list of all files in a path (filtered by audiofile_types)
    filelist = find_files(in_path,audiofile_types,relative_path=True)

    return extract_all_files(filelist, in_path, out_file, feature_types, label, verbose)



def extract_all_files_generic(in_path,
                              out_file = None,
                              feature_types = ['rp','ssd','rh'],
                              audiofile_types=('.wav','.mp3'),
                              path_prefix=None,
                              label=False,
                              append=False,
                              append_diff=False,
                              no_extension_check=False,
                              force_resampling=None,
                              out_HDF5 = False,
                              log_AudioTypes = True,
                              log_Errors = True,
                              verbose=True):
    """
    finds all files of a certain type (e.g. .wav and/or .mp3) in a path (+ sub-directories)
    OR loads a list of files to extract from a given .txt file

    extracts selected RP feature types
    and saves them into separate CSV feature files (one per feature type)

    # in_path: input file path to search for audio files (including subdirectories) OR .txt file containing a list of filenames
    # out_file: output file name stub for feature files to write (if omitted, features will be returned from function)
    # feature_types: RP feature types to extract. see rp_extract.py
    # audiofile_types: a string or tuple of suffixes to look for file extensions to consider (include the .)
    # path_prefix: prefix to be added to relative filenames (used typically together with .txt input files)
    # label:
    # append: append features to existing feature files
    # append_diff: append new features to existing output file(s) only if they are not in it/them yet
    # no_extension_check: does not check file format via extension. means that decoder is called on ALL files.
    # force_resampling: force a target sampling rate (provided in Hz) when decoding (works with FFMPEG only!)
    # out_HDF5: whether to store as HDF5 file format (otherwise CSV)
    # log_AudioTypes: creates a log file with audio format info
    # log_Errors: creates an error log file collecting all errors that appeared during feature extraction
    # verbose: verbose output or not
    """

    if in_path.lower().endswith('.txt'):  # treat as input file list
        from classes_io import read_filenames
        filelist = read_filenames(in_path)
        in_path = path_prefix # in case path_prefix is passed it is added to files in extract_all_files
    elif os.path.isdir(in_path): # find files in path
        if no_extension_check: audiofile_types = None # override filetypes to include all files (no extension check)
        filelist = find_files(in_path,audiofile_types,relative_path=True)
        # filelist will be relative, so we provide in_path below
    elif in_path.lower().endswith(audiofile_types) or no_extension_check: # treat as single audio input file
        filelist = [in_path]
        in_path = None # no abs path to add below
    else:
        raise ValueError("Cannot not process this kind of input file: " + in_path)

    if append_diff:
        # get differential filelist to extract only new feature files
        filelist = get_diff_filelist(out_file, filelist, feature_types, out_HDF5)
        append = True

    return extract_all_files(filelist, in_path, out_file, feature_types, label, append,
                             no_extension_check, force_resampling, out_HDF5, log_AudioTypes, log_Errors, verbose)


def get_diff_filelist(feature_filename, filelist, feature_types, useHDF5=False, verbose=True):
    '''from a filelist to extract features, check if we previously have extracted features from these files before and
    return only subset of filelist containing files we haven't analyzed yet'''

    # check if file exists, otherwise we do not append
    check_filename = feature_filename + '.' + feature_types[0] # check just the first feature type, assuming the others will exist too
    if useHDF5: check_filename += '.h5'
    if not os.path.isfile(check_filename):
        if verbose:
            print "WARNING: No previous feature file " + check_filename + " found. Will create new feature files."
        return filelist # unchanged, as is

    if not useHDF5:
        filelist_previous = read_csv_features(feature_filename, feature_types, ids_only=True, single_id_list=True)
    else:
        filelist_previous = load_multiple_hdf5_feature_files(feature_filename, feature_types, verbose=verbose, ids_only=True)

    filelist_diff = list(set(filelist) - set(filelist_previous))

    if verbose:
        print "Filelist has", len(filelist), "entries, found", len(filelist_previous), "previously analyzed files in feature file(s)."
        print "Analyzing only", len(filelist_diff), "new files."

    return filelist_diff


def extract_all_files(filelist,
                      path,
                      out_file=None,
                      feature_types =['rp','ssd','rh'],
                      label=False,
                      append=False,
                      no_extension_check=False,
                      force_resampling=None,
                      out_HDF5=False,
                      log_AudioTypes=True,
                      log_Errors=True,
                      verbose=True):
    """
    finds all files of a certain type (e.g. .wav and/or .mp3) in a path and all sub-directories in it
    extracts selected RP feature types
    and saves them into separate CSV feature files (one per feature type)

    # filelist: list of files for features to be extracted
    # path: absolute path that will be added at beginning of filelist (can be '')
    # out_file: output file name stub for feature files to write (if omitted, features will be returned from function)
    # feature_types: RP feature types to extract. see rp_extract.py
    # label: use subdirectory name as class label
    # no_extension_check: does not check file format via extension. means that decoder is called on ALL files.
    # force_resampling: force a target sampling rate (provided in Hz) when decoding (works with FFMPEG only!)
    # out_HDF5: whether to store as HDF5 file format (otherwise CSV)
    """

    ext = feature_types

    n = 0   # counting the files being processed
    n_extracted = 0   # counting the files that were actually analyzed
    err = 0 # counting errors
    n_files = len(filelist)

    # initialize filelist_extracted and dict containing all accumulated feature arrays
    filelist_extracted = []
    feat_array = {}
    audio_logwriter = None
    error_logwriter = None
    audio_logwriter_wrote_header = False

    start_time = time.time()

    if out_file: # only if out_file is specified

        if log_AudioTypes:
            log_filename = out_file + '.audiotypes.log'
            audio_logfile = open(log_filename, 'w') # TODO allow append mode 'a'
            audio_logwriter = unicsv.UnicodeCSVWriter(audio_logfile) #, quoting=csv.QUOTE_ALL)

        if log_Errors:
            err_log_filename = out_file + '.errors.log'
            error_logfile = open(err_log_filename, 'w') # TODO allow append mode 'a'
            error_logwriter = unicsv.UnicodeCSVWriter(error_logfile) #, quoting=csv.QUOTE_ALL)

        if out_HDF5:
            FeatureWriter = HDF5FeatureWriter()
        else:
            FeatureWriter = CSVFeatureWriter()
            FeatureWriter.open(out_file,ext,append=append)
    

    for fil in filelist:  # iterate over all files
        try:
            if n > 0:
                elaps_time = time.time() - start_time
                remain_time = elaps_time * n_files / n - elaps_time # n is the number of files done here
            else:
                remain_time = None

            n += 1

            if path:
                filename = path + os.sep + fil
            else:
                filename = fil
            if verbose:
                print '#',n,'/',n_files,'(ETA: ' + timestr(remain_time) + "):", filename

            # read audio file (wav or mp3)
            samplerate, samplewidth, data, decoder = audiofile_read(filename, verbose=verbose, include_decoder=True, no_extension_check=no_extension_check, force_resampling=force_resampling)

            # audio file info
            if verbose: print samplerate, "Hz,", data.shape[1], "channel(s),", data.shape[0], "samples"

            # extract features
            # Note: the True/False flags are determined by checking if a feature is listed in 'ext' (see settings above)

            feat = rp.rp_extract(data,
                              samplerate,
                              extract_rp   = ('rp' in ext),          # extract Rhythm Patterns features
                              extract_ssd  = ('ssd' in ext),           # extract Statistical Spectrum Descriptor
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
                              step_width=1,
                              verbose = verbose)

            # TODO check if ext and feat.keys are consistent

            # WHAT TO USE AS ID (based on filename): 3 choices:
            id = fil  # rel. filename as from find_files
            # id = filename   # full filename incl. full path
            # id = filename[len(path)+1:] # relative filename only (extracted from path)

            if out_file:
                # WRITE each feature set to a CSV or HDF5 file
                
                id2 = None
                
                if label:
                    id2 = id.replace("\\","/").split("/")[-2].strip()

                if out_HDF5 and n_extracted==0:
                    # for HDF5 we need to know the vector dimension
                    # thats why we cannot open the file earlier
                    FeatureWriter.open(out_file,ext,feat,append=append) # append not working for now but possibly in future

                FeatureWriter.write_features(id,feat,id2)
            else:
                # IN MEMORY: add the extracted features for 1 file to the array dict accumulating all files
                # TODO: only if we don't have out_file? maybe we want this as a general option

                if feat_array == {}: # for first file, initialize empty array with dimension of the feature set
                    for e in feat.keys():
                        feat_array[e] = np.empty((0,feat[e].shape[0]))

                # store features in array
                for e in feat.keys():
                    feat_array[e] = np.append(feat_array[e], feat[e].reshape(1,-1), axis = 0) # 1 for horizontal vector, -1 means take original dimension

                filelist_extracted.append(id)

            n_extracted += 1

            # write list of analyzed audio files alongsize audio metadata (kHz, bit, etc.)
            if audio_logwriter:
                if not audio_logwriter_wrote_header: # write CSV header
                    log_info = ["filename","decoder","samplerate (kHz)","samplewidth (bit)","n channels","n samples"]
                    audio_logwriter.writerow(log_info)
                    audio_logwriter_wrote_header = True

                log_info = [filename,decoder,samplerate,samplewidth*8,data.shape[1],data.shape[0]]
                audio_logwriter.writerow(log_info)

            gc.collect() # after every file we do garbage collection, otherwise our memory is used up quickly for some reason

        except Exception as e:
            print "ERROR analysing file: " + fil + ": " + str(e)
            err += 1
            if error_logwriter:
                error_logwriter.writerow([fil,str(e)])

    try:
        if out_file:  # close all output files
            FeatureWriter.close()

            if audio_logwriter:
                audio_logfile.close()

        if error_logwriter:
            error_logfile.close()

    except Exception as e:
        print "ERROR closing the output or log files: " + str(e)

    end_time = time.time()

    if verbose:
        print "FEATURE EXTRACTION FINISHED.", n, "file(s) processed,", n_extracted, "successful. Duration:", timestr(end_time-start_time)
        if err > 0:
            print err, "file(s) had ERRORs during feature extraction.",
            if log_Errors:
                print "See", err_log_filename
            else:
                print
        if out_file:
            opt_ext = '.h5' if out_HDF5 else ''
            print "Feature file(s):", out_file + "." + str(ext) + opt_ext

    if out_file is None:
        return filelist_extracted, feat_array



if __name__ == '__main__':

    argparser = argparse.ArgumentParser() #formatter_class=argparse.ArgumentDefaultsHelpFormatter) # formatter_class adds the default values to print output

    argparser.add_argument('input_path', help='audio file, audio filelist (.txt), or input path to search for audio files to analyze') # nargs='?' to make it optional
    argparser.add_argument('output_filename', nargs='?', help='output path + filename for feature file (without extension) [default: features/features]', default='features/features') # nargs='?' to make it optional

    argparser.add_argument('-pre','--pathprefix',help='optional path prefix, if input_path is a filelist containing relative paths', default='')
    argparser.add_argument('-noext','--noextensioncheck', action='store_true',help='do not check audio file format via extension; means that decoder is called on ALL files',default=False) # boolean opt
    argparser.add_argument('-force','--forceresampling', type=int,help='force a target sampling rate (in Hz) for decoding (works with FFMPEG only)',default=None)

    argparser.add_argument('-rp',   action='store_true',help='extract Rhythm Patterns (default)',default=False) # boolean opt
    argparser.add_argument('-rh',   action='store_true',help='extract Rhythm Histograms (default)',default=False) # boolean opt
    argparser.add_argument('-ssd',  action='store_true',help='extract Statistical Spectrum Descriptors (default)',default=False) # boolean opt
    argparser.add_argument('-trh',  action='store_true',help='extract Temporal Rhythm Histograms',default=False) # boolean opt
    argparser.add_argument('-tssd', action='store_true',help='extract Temporal Statistical Spectrum Descriptors',default=False) # boolean opt
    argparser.add_argument('-mvd',  action='store_true',help='extract Modulation Frequency Variance Descriptors',default=False) # boolean opt
    argparser.add_argument('-a','--all', action='store_true',help='extract ALL of the aforementioned features',default=False) # boolean opt

    argparser.add_argument('-ap', '--append', action='store_true', help='append new features to existing output file(s)', default=False)  # boolean opt
    argparser.add_argument('-adiff', '--appenddiff', action='store_true', help='append new features to existing output file(s) only if they are not in it yet', default=False)  # boolean opt
    argparser.add_argument('-h5','--hdf5', action='store_true',help='store output to HDF5 files instead of CSV',default=False) # boolean opt

    argparser.add_argument('-label',action='store_true',help='use subdirectory name as class label',default=False) # boolean opt

    args = argparser.parse_args()

    # check if outpath contains path that yet needs to be created
    outpath, _ = os.path.split(args.output_filename)
    if not outpath == '' and not os.path.exists(outpath):
        os.mkdir(outpath)

    # select the feature types according to given option(s) or default
    feature_types = []
    if args.rp: feature_types.append('rp')
    if args.rh: feature_types.append('rh')
    if args.trh: feature_types.append('trh')
    if args.ssd: feature_types.append('ssd')
    if args.tssd: feature_types.append('tssd')
    if args.mvd: feature_types.append('mvd')
    if args.all: feature_types = ['rp','ssd','rh','tssd','trh','mvd']

    # if none was selected set default feature set
    if feature_types == []: feature_types = ['rp','ssd','rh']

    audiofile_types = get_supported_audio_formats()

    print "Extracting features:", feature_types
    print "From files in:", args.input_path
    print "File types:",
    print "ALL FILES (NO EXTENSION CHECK)" if args.noextensioncheck else audiofile_types


    # BATCH RP FEATURE EXTRACTION:
    extract_all_files_generic(args.input_path,args.output_filename,feature_types, audiofile_types,
                              args.pathprefix, args.label, args.append, args.appenddiff, args.noextensioncheck, args.forceresampling,
                              args.hdf5, log_AudioTypes = True)

    # EXAMPLE ON HOW TO READ THE FEATURE FILES
    #ids, features = read_feature_files(args.output_filename,feature_types)
