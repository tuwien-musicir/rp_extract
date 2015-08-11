# 20.07. and 06.08.2015 by Thomas Lidy

# load and save feature files from RP_extract

# supported formats: CSV, ARFF, NPZ(load)

import os
import pandas as pd
from rp_extract_files import read_feature_files  # for csv_to_arff


# == CSV ==


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

def read_csv_features(filenamestub,ext,separate_ids=True,id_column=0):

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
    
    

# converts np.array + extra ids and/or classes to Pandas dataframe
# ids (e.g. audio filenames) and classes can be provided optionally as list (will be excluded if omitted)
# feature attribute labels also optionally as a list (will be generated if omitted)

def to_dataframe(feature_data, attribute_labels=None, ids=None, classes=None):

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



# convert npz to arff format
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



# classes_from_filename:
# derive class label from filename or relative file path
# this function derives class labels from the document file names (ids) given in the original feature files

# examples:
# split class by first / or \ (os.sep)
#classes = classes_from_filename(ids[ext]) if add_class else None
# split class by first '.' as used e.g. in GTZAN collection: pop.00001.wav
#classes = classes_from_filename(ids[ext]) if add_class else None

# TODO: adapt to enable splitting by LAST appearance of split_char instead of first

def classes_from_filename(filenames,split_char=os.sep):
    # this example works for GTZAN collection: class is first part of filename before '.'
    classes = [x.split(split_char, 1)[0] for x in filenames]
    return classes


# convert feature files that are stored in CSV format to Weka ARFF format
# in_filenamestub, in_filenamestub: full file path and filname but without .rp, .rh etc. extension (will be added from feature types) for input and output feature files
# feature_types = ['rp','ssd','rh','mvd']

def csv2arff(in_filenamestub,out_filenamestub,feature_types,add_class=True):

    ids, features = read_feature_files(in_filenamestub,feature_types)

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
        df = to_dataframe(features[ext], classes=classes)

        # WRITE ARFF
        out_filename = out_filenamestub + "." + ext + ".arff"
        print "Saving " + out_filename + " ..."
        save_arff(out_filename,df)

    print "Finished."


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