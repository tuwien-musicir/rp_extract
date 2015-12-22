'''RP_Feature_IO

2015 by Thomas Lidy

load and save feature files from RP_extract
manipulate, merge, cut, align features and more helpful functions

supported formats:
read and write: CSV, ARFF
read only currently: HDF5, NPZ (Numpy Pickle)
'''

import os
import sys
import pandas as pd


# == CSV ==


def check_duplicates(file_ids):
    '''check for duplicates in file_ids from CSV or feature files'''
    dup = set([x for x in file_ids if file_ids.count(x) > 1])
    if len(dup) > 0:
        print >> sys.stderr, "DUPLICATE(S):", "; ".join(list(dup))
        raise ValueError("Duplicate entries in file ids!")


def read_csv_features1(filename,separate_ids=True,id_column=0):
    ''' Read_CSV_features1

    read features (and optionally one or more ids) from a CSV file

    Note: in write_feature_files we use unicsv to store the features.
    It will quote strings containing , and other characters if needed only (automatically)
    Here we use pandas to import CSV as a pandas dataframe,
    because it handles quoted filenames (containing ,) well (by contrast to other CSV readers)

    Parameters:
    :param filename: filename path to CSV file to read from
    :param separate_ids: will split off the id column(s) from the features (containing eg. and index or filename string)
    :param id_column: specify which is/are the id column(s) as integer or list, e.g. 0 (= first column) or [0,1] (= first two columns)
            negative integers identify the columns backwards, i.e. -1 is the last column, -2 the second last column, and so on
    :return: if separate_ids is True, it will return a tuple (ids, features)
             with ids containing usually identifiers as list of strings and features the numeric data as numpy array;
             if separate_ids is False, just the features array will returned (containing everything read from the CSV)
    '''

    import numpy as np
    import pandas as pd

    # we use pandas to import CSV as pandas dataframe,
    # because it handles quoted filenames (containing ,) well (by contrast to other CSV readers)
    dataframe = pd.read_csv(filename, sep=',',header=None)

    # future option: this would be a way to set the file ids as index in the dataframe
    # dataframe = pd.read_csv(filename, sep=',',header=None,index_col=0) # index_col=0 makes 1st column the rowname (index)

    # convert to numpy matrix/array
    feat = dataframe.as_matrix(columns=None)

    if separate_ids:
        ids = feat[:,id_column]
        feat = np.delete(feat,id_column,1).astype(np.float) # delete id columns and return feature vectors as float type
        return (ids,feat)
    else:
        return feat


def read_csv_features(filenamestub,ext,separate_ids=True,id_column=0,verbose=True):
    ''' Read_CSV_features:

    read pre-analyzed features from multiple CSV files (with feature name extensions)

    Parameters:
    # filenamestub: full path to feature file name WITHOUT .extension
    # ext: a list of file .extensions (e.g. 'rh','ssd','rp') to be read in
    # separate_ids: if False, it will return a single matrix containing the id column
    #               if True, it will return a tuple: (ids, features) separating the id column from the features
    # id_column: which of the CSV columns contains the ids (default = 0, i.e. first column)
    #
    # returns: single numpy matrix including ids, or tuple of (ids, features) with ids and features separately
    #          each of them is a python dict containing an entry per feature extension (ext)
    '''

    # initialize empty dicts
    feat = {}
    ids = {}

    for e in ext:
        filename = filenamestub + "." + e

        if separate_ids:
            ids[e], feat[e] = read_csv_features1(filename,separate_ids,id_column)
        else:
            feat[e] = read_csv_features1(filename,separate_ids,id_column)

        if verbose: print "Read:", e + ":\t", feat[e].shape[0], "audio file vectors,", feat[e].shape[1], "dimensions"

    if separate_ids:
        # check for ID consistency
        ext = ids.keys()
        for e in ext[1:]:
            if not len(ids[ext[0]]) == len(ids[e]):
                raise ValueError("Feature files have different number of entries! " +
                                 ", ".join([e + ': ' + str(len(ids[e])) for e in ext]) )
            if not all(ids[ext[0]] == ids[e]):
                raise ValueError("Ids not matching across feature files!")

        # once consistent, check for duplicates
        check_duplicates(ids[ext[0]].tolist())

        return(ids,feat)

    else:
        return feat



# == ARFF ==

# load_arff
# read arff file and bring to Numpy array format
# returns a tuple (features,classes)

def load_arff(arff_file):

    import scipy.io.arff as arff
    import numpy as np

    arffdata, metadata = arff.loadarff(arff_file)

    # GET CLASS DATA # TODO: check if we even have a class attribute
    classes = arffdata['class']

    # GET FEATURE DATA
    # strip off class column and get correct NP 2D array with data
    features = arffdata[metadata.names()[:-1]] #everything but the last column # TODO: check which column is the class column and remove that one
    features = features.view(np.float).reshape(arffdata.shape + (-1,)) #converts the record array to a normal numpy array

    return (features,classes)


# save_arff
# save data in Weka ARFF format
# adds typcial ARFF file headers (@relation, @attribute), optionally adds class label + writes data into ARFF format
# based on npz2arff arff writing code by Alexander Schindler

def save_arff(filename,dataframe,relation_name=None):
    
    if relation_name is None:
        relation_name = filename
    
    out_file = open(filename, 'w')
    
    out_file.write("@Relation {0}\n".format(relation_name))
    
    for column in dataframe:
        if column == "ID": 
            out_file.write("@Attribute ID string\n")
        elif column == "class":
            class_list = dataframe["class"].unique()
            out_file.write("@Attribute class {{{0}}}\n".format(",".join(class_list)))   
        else:
            # assume all other columns are numeric
            out_file.write("@Attribute {0} numeric\n".format(column))
    
    # now for the feature data
    out_file.write("\n@Data\n")
    dataframe.to_csv(out_file, header=None, index=None)
    
    # NumPy variant:
    #np.savetxt(out_file, a, delimiter=",")
    
    out_file.close()
    
    
# == NPZ (Numpy Pickle) ==



# == HDF5 ==


def load_hdf5(hdf_filename):
    store = pd.HDFStore(hdf_filename)
    # .as_matrix(columns=None) converts to Numpy array (of undefined data column types)
    data = store['data'].as_matrix(columns=None)
    store.close()
    return(data)


# == CONVERSION ==


# Csv2Arff
# convert feature files that are stored in CSV format to Weka ARFF format
# in_filenamestub, in_filenamestub: full file path and filname but without .rp, .rh etc. extension (will be added from feature types) for input and output feature files
# feature_types = ['rp','ssd','rh','mvd']

def csv2arff(in_filenamestub,out_filenamestub,feature_types,add_class=True):

    from classes_io import classes_from_filename

    ids, features = read_csv_features(in_filenamestub,feature_types)

    for ext in feature_types:

        # derive the class labels from the audio filenames (see function above)
        # THIS VERSION splits by first / or \ (os.sep)
        #classes = classes_from_filename(ids[ext]) if add_class else None

        # THIS VERSION splits by first '.' as used e.g. in GTZAN collection: pop.00001.wav
        classes = classes_from_filename(ids[ext],'.') if add_class else None

        # CREATE DATAFRAME
        # with ids
        #df = to_dataframe(features[ext], None, ids[ext], classes)
        # without ids
        df = to_dataframe_for_arff(features[ext], classes=classes)

        # WRITE ARFF
        out_filename = out_filenamestub + "." + ext + ".arff"
        print "Saving " + out_filename + " ..."
        save_arff(out_filename,df)

    print "Finished."


# convert NPZ (Numpy Pickle) to ARFF format
# adapted from Alex Schindlers npz2arff.py # untested: TODO: test!

def npz2arff(in_file, out_file, relation_name, include_filenames=False):

    import numpy as np
    npz = np.load(in_file)

    # load data
    data = pd.DataFrame(npz["data"], columns=npz["attribute_names"])

    if include_filenames:
        data["ID"] = npz["filenames"]

    data["class"] = npz["labels"]

    npz.close()

    ordered_indexes = data.columns.tolist()
    ordered_indexes.remove("labels")

    if include_filenames:
        ordered_indexes.remove("filenames")


    save_arff(out_file,data,relation_name)


def csv2hdf5(csv_filename,hdf_filename,chunk_size=1000,verbose=True):
    ''' Csv2Hdf5

    converting CSV files to HDF5 file format (using Pandas HDFStore)

    Parameters:
    csv_filename: input filename of file to convert
    hdf_filename: output HDF5 filename
    chunk_size: number of files to read chunk-wise from CSV and store iteratively in HDF5 (default: 1000)
    verbose: if False no output will be printed
    '''
    import os
    #import numpy as np
    import pandas as pd

    # we check and delete the filename, otherwise we would always append
    if os.path.exists(hdf_filename):
        os.remove(hdf_filename)

    store = pd.HDFStore(hdf_filename)

    # read all at once:
    #dataframe = pd.read_csv(csv_filename, sep=',',header=None)

    cnt = 0
    csv_reader = pd.read_csv(csv_filename, sep=',',header=None,chunksize=chunk_size)

    for chunk in csv_reader:
        store.append('data', chunk)
        cnt += chunk.shape[0]
        if verbose: print "processed", cnt, "rows"

    store.close()
    if verbose: print "Finished."


# == FEATURE MANIPULATION ==

def concatenate_features(feat, feature_types = ('ssd', 'rh')):
    ''' concatenate features vectors of various feature types together
    (all entries in feature dictionary  must have same number of instances, but can have different dimensions
    feat: feature dictionary, containing np.arrays for various feature types (named 'ssd', 'rh', etc.)
    feature_types: tuple of strings with the names of feature types to concatenate (1 entry tuple or string is
        allowed, will return this feature type only, without concatenation)
    '''
    import numpy as np

    # in case only 1 feature type instad of tuple was passed
    if isinstance(feature_types, str):
        new_feat = feat[feature_types]
    else:
        # take the first feature type
        new_feat = feat[feature_types[0]]
        # and iteratively horizontally stack the remaining feature types
        for e in feature_types[1:]:
            new_feat = np.hstack((new_feat, feat[e]))
    return new_feat


def sorted_feature_subset(features, ids_orig, ids_select):
    '''
    selects a (sorted) subset of the original features array
    features: a feature dictionary, containing multiple np.arrays one for each feature type ('rh', 'ssd', etc.)
    ids_orig: original labels/ids for the featurs, MUST be same length AND order as feature arrays
    ids_select: set of ids in different order or subset of ids to be selected from original feature arrays
    returns: sorted subset of the original features array
    '''
    new_feat = {}
    for e in features.keys():
        dataframe = pd.DataFrame(features[e], index = ids_orig)
        new_feat[e] = dataframe.ix[ids_select].values
    return new_feat


# == HELPER FUNCTIONS ==

def to_dataframe(feature_data, ids=None, attribute_labels=None):
    '''converts np.array to Pandas dataframe with optional
    ids (e.g. audio filenames) to be added as rownames (index)
    and/or attribute names to be added as column labels
    '''
    dataframe = pd.DataFrame(feature_data, index = ids, columns=attribute_labels)
    return dataframe


def to_dataframe_for_arff(feature_data, attribute_labels=None, ids=None, classes=None):
    '''converts np.array + extra ids and/or classes to Pandas dataframe
    ids (e.g. audio filenames) and classes can be provided optionally as list (will be excluded if omitted)
    feature attribute labels also optionally as a list (will be generated if omitted)
    '''

    if attribute_labels is None:
        attribute_labels = feature_data.dtype.names
        if attribute_labels == None:
            # if nothing is passed and nothing is stored in np array we create the attribute names
            fdim = feature_data.shape[1]
            attribute_labels = [("feat" + str(x)) for x in range(fdim)]

    if feature_data.dtype == object: # convert to float for proper PD output to arff
        feature_data = feature_data.astype(float)

    dataframe = pd.DataFrame(feature_data, columns=attribute_labels)

    if not ids is None:
        dataframe["ID"] = ids

    if not classes is None:
        dataframe["class"] = pd.Categorical(classes) # classes

    return dataframe



def load_or_analyze_features(input_path, feature_types = ['rp','ssd','rh'], save_features = False, output_file = None):
    """convenient function that can either load or freshly extract features

    depending if input_path is...
    a) a path: will recursively look for .wav and .mp3 files in path and freshly extract features
    b) a .txt file: will take a list of filenames (one per line) and freshly extract features
    c) another file: will load features in multiple csv file feature format
    TODO:
    d) a .npz, .h5 or .hdf5 file: will load the features from that file
    e) a .wav or .mp3 file: will freshly extract features from that file

    TODO: The audio analysis parameters of this function call are only needed when features are to be freshly extracted.

    :param input_path:
    :return:
    """
    from rp_extract_batch import extract_all_files_generic

    # not possible because of omitted file extensions in read_csv_features below
    #    if not os.path.exists(input_path):
    #        raise NameError("File or path does not exist: " + input_path)

    if save_features and output_file is None:
        raise ValueError("output_file must be specified if save_features is set to True!")

    # if we got a directory, we do analysis, if we got a file, we load it

    if os.path.isdir(input_path) or input_path.lower().endswith(('.txt','.wav','.mp3','.aif')):  # FRESH ANALYSIS from input path or .txt file

        print "Performing feature extraction from ", input_path

        # BATCH RP FEATURE EXTRACTION:
        # if output_file is given, will save features, otherwise not
        ids, feat = extract_all_files_generic(input_path,output_file,feature_types,audiofile_types=('.wav','.mp3','.aif'))

    else:
        # LOAD from Feature File
        ids, feat = read_csv_features(input_path,feature_types)

        # from the ids dict, we take only the first entry and convert numpy array to list
        ids = ids.values()[0].tolist()

        # TODO: read .npz, .h5 or .hdf5

    return ids, feat



if __name__ == '__main__':

    # test CSV to ARFF

    in_path = '/data/music/GTZAN/vec'
    out_path = './feat'
    filenamestub = 'GTZAN.python'

    feature_types = ['rp','ssd','rh','mvd']

    in_filenamestub = in_path + os.sep + filenamestub
    out_filenamestub = out_path + os.sep + filenamestub

    csv2arff(in_filenamestub,out_filenamestub,feature_types)

    # try to load ARFF

    feat_type = 'ssd'

    arff_file = out_filenamestub + '.' + feat_type + '.arff'

    features, classes  = load_arff(arff_file)

    print "Reading ", arff_file
    print "classes:" , classes.shape
    print "feature dimensions:", features.shape