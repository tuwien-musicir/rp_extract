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

# see further imports in CSVFeatureWriter.open() and HDF5FeatureWriter.open()


# === PART 1: new FeatureWriter classes ===

ABSTRACT_NOT_IMPLEMENTED_MSG = 'Not implemented in abstract class. Use a subclass of FeatureWriter!'

class FeatureWriter(object):
    '''BaseModel class which defines and initializes rudimentary network parameters and methods'''

    def __init__(self):
        raise NotImplementedError(ABSTRACT_NOT_IMPLEMENTED_MSG)

    def open(self,base_filename,ext,append):
        raise NotImplementedError(ABSTRACT_NOT_IMPLEMENTED_MSG)


class CSVFeatureWriter(FeatureWriter):

    def __init__(self):
    #    super(CSVFeatureWriter, self).__init__()
        self.isopen = False
        self.files = None
        self.writer = None
        self.ext = None  # file extensions i.e. feature types

    def open(self,base_filename,ext,append=False):
        '''
        base_filename: path and filename that will be extended by . and feature extension
        ext: list of file extensions i.e. feature types to open files for
        append: whether to append to existing feature files (or overwrite)
        '''

        import unicsv # unicode csv library (installed via pip install unicsv)

        self.ext = ext   # keep extensions
        self.files = {}  # files is a dict of one file handle per extension
        self.writer = {} # writer is a dict of one file writer per extension

        # append or write new (will overwrite)
        mode = 'a' if append else 'w'

        for e in ext:
            filename = base_filename + '.' + e
            self.files[e] = open(filename, mode)
            self.writer[e] = unicsv.UnicodeCSVWriter(self.files[e]) #, quoting=csv.QUOTE_ALL)

        self.isopen = True

    def write_features(self,id,feat,id2=None,flush=True):
        '''
        id: string id (e.g. filename) of extracted file
        feat: dict containing 1 entry per feature type (must match file extensions)
        id2: optional secondary identifier to be stored alongside id
        flush: flush data to disk after every write (i.e. prevent data loss in case of premature termination)
        '''
        if self.isopen is False or self.writer=={}:
            raise RuntimeError("File or writer is not open yet. Call open first!")
        # TODO: check if all feat.keys() == self.ext

        for e in feat.keys():
            f=feat[e].tolist()
            f.insert(0,id)      # add filename/id before vector (to include path, change fil to filename)
            if id2 is not None: # add secondary identifier
                f.insert(1,id2)
            self.writer[e].writerow(f)

            if flush:
                self.files[e].flush()  # flush file after writing, otherwise data is not written until termination of program

    def close(self):
        if self.isopen:
            for e in self.ext:
                self.files[e].close()
            self.isopen = False

class HDF5FeatureWriter(FeatureWriter):

    def __init__(self,float32=False):
        '''float32: if set to True, 32 bit data is stored, otherwise 64 bit'''
        self.isopen = False
        self.append = False
        self.files = None
        self.h5tables = None
        self.idtables = None
        self.idtables2 = None
        self.ext = None  # file extensions i.e. feature types

        import tables  # pytables HDF5 library (installed via pip install tables)
        # define data types to store in HDF5
        self.data_type = tables.Float32Atom() if float32 else tables.Float64Atom()
        # alternative: derive from np array: tables.Atom.from_dtype(myarray.dtype)
        self.string_type = tables.StringAtom(itemsize=256)
        # NOTE: hard-coded character limit of 256 chars for file ids (can be increased if needed)
        # alternative solution for avoiding String limit is:
        #class FileInfo(tables.IsDescription):
        #    file_id = tables.StringCol(256)
        #    seg_pos = tables.IntCol()
        #table = h5file.create_table(h5file.root, 'file_ids_table', FileInfo)

    def open(self,base_filename,ext,feat=None,append=False):
        '''
        base_filename: path and filename that will be extended by . and feature extension and .h5
        ext: list of file extensions i.e. feature types to open files for
        feat: (deprecated, not needed anymore) an example feature vector dict containing 1 entry of a feature vector per
            feature type the vectors are not written here, just used to determine their dimensions
        append: whether to append to existing feature files (or overwrite) (not implemented)
        '''

        import tables
        self.ext = ext   # keep extensions
        self.files = {}  # files is a dict of one file handle per extension

        for e in ext:
            # create file
            outfile = base_filename + '.' + e + '.h5'
            mode = 'r+' if append else 'w'   #'r+' is similar to 'a', but the file must already exist

            if append and not os.path.isfile(outfile):
                # fallback: if file is not existing we cannot append and create new file + tables
                mode = 'w'
                append = False

            #self.files[e] =  = tables.openFile(outfile, mode) # tables <= 3.1.1
            self.files[e] = tables.open_file(outfile, mode) # tables >= 3.2

        self.append = append
        self.isopen = True

    def _init_tables(self, feat):
        '''initalize HDF5 tables: means to create them (in case of a new file) or read the root of the tables
        (in case of appending). for new creation we need a vector example to create the tables with proper vector dimensions.

        feat: an example feature vector dict containing 1 entry of a feature vector per feature type
              the vectors are not written here, just used to determine their dimensions
        '''
        self.h5tables = {} # dict of 1 vector table per extension
        self.idtables = {} # dict of 1 file_id table per extension
        self.idtables2 = {} # dict of 1 secondary file_id table per extension

        for e in self.ext:
            h5file = self.files[e]

            if self.append:
                if h5file.root.__contains__('vec'):
                    self.h5tables[e] = h5file.root.vec
                else:
                    raise AttributeError("HDF5 file does not contain 'vec' table! Cannot append.")
                if h5file.root.__contains__('file_ids'):
                    self.idtables[e] = h5file.root.file_ids
                else:
                    raise AttributeError("HDF5 file does not contain 'file_ids' table! Cannot append.")
                if h5file.root.__contains__('file_ids2'):
                    self.idtables2[e] = h5file.root.file_ids2
            else:
                # create table for vectors
                vec_dim = len(feat[e]) if feat[e].ndim == 1 else feat[e].shape[1]
                shape = (0, vec_dim)  # define feature dimension but not yet number of instances (0)

                if hasattr(h5file, "createEArray"):    # tables 2.x to 3.1
                    h5table = h5file.createEArray(h5file.root, 'vec', self.data_type, shape)
                elif hasattr(h5file, "create_earray"): # tables >= 3.2
                    h5table = h5file.create_earray(h5file.root, 'vec', self.data_type, shape)
                else:
                    raise ValueError("No method createEArray or create_earray in PyTables!")

                h5table.attrs.vec_dim = vec_dim
                h5table.attrs.vec_type = e.upper()
                self.h5tables[e] = h5table

                # create table for file_ids (strings)
                shape = (0,)  # growing dimension 0, undefined other dimension

                if hasattr(h5file, "createEArray"):  # tables 2.x to 3.1
                    self.idtables[e] = h5file.createEArray(h5file.root, 'file_ids', self.string_type, shape)
                    self.idtables2[e] = h5file.createEArray(h5file.root, 'file_ids2', self.string_type, shape)
                elif hasattr(h5file, "create_earray"): # tables >= 3.2
                    self.idtables[e] = h5file.create_earray(h5file.root, 'file_ids', self.string_type, shape)
                    self.idtables2[e] = h5file.create_earray(h5file.root, 'file_ids2', self.string_type, shape)


    #def store_attribures(self,attributes):
        # TODO store audio analysis parameters as table attributes
        #for e in ext:
            #table = self.h5tables[e]
            #h5table.attrs.bands = n_bands
            #h5table.attrs.frames = frames
            #h5table.attrs.Mel = useMel
            #h5table.attrs.Bark = useBark
            #h5table.attrs.transform = transform
            #h5table.attrs.log_transform = log_transform

    def write_features(self,id,feat,id2=None,flush=True):
        '''
        write a single feature vector (for each file type, to multiple feature files)

        id: string id (e.g. filename) of extracted file
        feat: dict containing 1 entry per feature type (must match file extensions)
        id2: optional secondary identifier to be stored alongside id
        flush: flush data to disk after every write (i.e. prevent data loss in case of premature termination)
        '''
        if not self.isopen:
            raise RuntimeError("HDF5FeatureWriter is not open yet. Call open first!")

        if self.h5tables is None:
            self._init_tables(feat)

        for e in feat.keys():
            # write features and file_ids
            self.h5tables[e].append(feat[e].reshape((1,-1))) # make it a row vector instead of column
            self.idtables[e].append([id])  # it's important to have the list brackets here
            if id2 is not None:
                self.idtables2[e].append([id2])

            if flush:
                self.files[e].flush() # flush file after writing, otherwise data is not written until termination of program

    def write_features_batch(self,ids,feat,ids2=None,flush=True):
        '''
        write multiple feature vectors (+ ids) in a batch (for each file type, to multiple feature files)

        ids: list of string ids (e.g. filename) of analyzed files
        feat: dict containing 1 entry per feature type (must match file extensions) with multiple feature vectors
        id2: optional secondary identifier list to be stored alongside id
        flush: flush data to disk after every write (i.e. prevent data loss in case of premature termination)
        '''
        if not self.isopen:
            raise RuntimeError("HDF5FeatureWriter is not open yet. Call open first!")

        if self.h5tables is None:
            self._init_tables(feat)

        for e in feat.keys():
            # write features and file_ids
            self.h5tables[e].append(feat[e]) # no reshape here
            self.idtables[e].append(ids)  # assumed to be list already
            if ids2 is not None:
                self.idtables2[e].append(ids2) # assumed to be list already

            if flush:
                self.files[e].flush() # flush file after writing, otherwise data is not written until termination of program


    def close(self):
        if self.isopen and self.ext is not None: # if it's None, files are not open yet
            for e in self.ext:
                self.files[e].close()
        self.isopen = False



# === PART 2: old individual functions for reading/writing features ===


# == Helper Functions ==


def check_duplicates(file_ids,raise_error=True):
    '''check for duplicates in file_ids from CSV or feature files'''
    dup = set([x for x in file_ids if file_ids.count(x) > 1])
    if len(dup) > 0:
        message = "Duplicate entries in file ids! " + \
                  str(len(dup)) + " duplicate(s):\n" + "; ".join(list(dup))
        if raise_error:
            raise ValueError(message)
        else:
            import warnings
            warnings.warn(message)

def check_id_consistency(ids):
    '''check for ID consistency
    ids: dict containing multiple lists of ids, which should all be the same
    '''
    ext = ids.keys()
    for e in ext[1:]:
        if not len(ids[ext[0]]) == len(ids[e]):
            raise ValueError("Feature files have different number of ID entries! " +
                             ", ".join([e + ': ' + str(len(ids[e])) for e in ext]))
        if not ids[ext[0]] == ids[e]:
            raise ValueError("IDs not matching across feature files!")


# == CSV ==


def read_csv_features1(filename,separate_ids=True,id_column=0,sep=',',as_dataframe=False,ids_only=False):
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
    :param sep: separator in CSV file (default: ',')
    :param as_dataframe: returns Pandas dataframe, otherwise returns Numpy matrix and ids optionally separately
    :param ids_only: return only Id column(s)
    :return: if separate_ids is True, it will return a tuple (ids, features)
             with ids containing usually identifiers as list of strings and features the numeric data as numpy array;
             if separate_ids is False, just the features array will returned (containing everything read from the CSV)
    '''

    import numpy as np
    import pandas as pd

    # we use pandas to import CSV as pandas dataframe,
    # because it handles quoted filenames (containing ,) well (by contrast to other CSV readers)
    # the id_column will be made directly the database index if it is a simple one (otherwise it will be split off from matrix below)
    index_col = id_column if isinstance(id_column, int) else None
    dataframe = pd.read_csv(filename, sep=sep, header=None, index_col=index_col)

    if as_dataframe:
        if ids_only: return dataframe.index.tolist()
        return dataframe

    # convert to numpy matrix/array
    feat = dataframe.as_matrix(columns=None)

    if separate_ids or ids_only:
        if index_col is not None:
            ids = dataframe.index.tolist()
        else:
            ids = feat[:, id_column].tolist() # TODO tolist() will not make sense when multiple columns are chosen

        if ids_only:
            return ids

        if index_col is None:
            # this means we still have the id column in the feature matrix and need to remove it
            feat = np.delete(feat, id_column, 1).astype(np.float)   # convert feature vectors to float type
        return ids, feat

    # in all other cases we return the raw feature matrix
    return feat


def read_csv_features(filenamestub,ext=('rh','ssd','rp'),separate_ids=True,id_column=0,single_id_list=False,
                      as_dataframe=False,ids_only=False,error_on_duplicates=True,verbose=True):
    ''' Read_CSV_features:

    read pre-analyzed features from multiple CSV files (with feature name extensions)

    Parameters:
    # filenamestub: full path to feature file name WITHOUT .extension
    # ext: a tuple or list of file .extensions (e.g. ('rh','ssd','rp')) to be read in
    # separate_ids: if False, it will return a single matrix containing the id column
    #               if True, it will return a tuple: (ids, features) separating the id column from the features
    # id_column: which of the CSV columns contains the ids (default = 0, i.e. first column)
    # single_id_list: if separate_ids and single_id_list are True, this will return a single id list instead of a dictionary
    :param as_dataframe: returns Pandas dataframe, otherwise returns Numpy matrix and ids optionally separately
    :param ids_only: return only Id column(s)
    # returns:  if separate_ids == False:
                    a Python dict containing one entry per feature extension (ext), which is
                    a NumPy matrix containing all data including ids
                if separate_ids == True:
                    a tuple of (ids, features) where both are Python dicts containing one entry per feature extension (ext)
                    ids:  with all the ideas as a Numpy array containing strings
                          (if single_id_list == True, ids will be reduced to a single Python list, cause the arrays are supposed to be all identical)
                    features: NumPy matrix containing all numeric feature data

    '''

    # initialize empty dicts
    feat = {}
    ids = {}

    # make list if only 1 string is passed
    if isinstance(ext, str):
        ext = [ext]

    for e in ext:
        filename = filenamestub + "." + e

        if separate_ids and not as_dataframe:
            ids[e], feat[e] = read_csv_features1(filename,separate_ids,id_column)
            check_id_consistency(ids)
        else:
            feat[e] = read_csv_features1(filename,separate_ids,id_column,as_dataframe=as_dataframe)
            # TODO check consistency also in dataframes
            # ids[e] = dataframe.index.tolist()
            # AFTER FOR LOOP: check_id_consistency(ids)

        if verbose:
            print "Read", feat[e].shape[0], "feature vectors with dimension", feat[e].shape[1], ", type " + e.upper()

    # check if we have duplicates in the ids
    if separate_ids and not as_dataframe:
        check_duplicates(ids[ext[0]],raise_error=error_on_duplicates)
    elif as_dataframe:
        ids = feat[ext[0]].index.tolist() # take index of first entry in feature dict
        check_duplicates(ids, raise_error=error_on_duplicates)
    # if we dont have separate ids we cant check duplicates as of now

    if separate_ids and single_id_list:
        # from the ids dict, we take only the first entry
        ids = ids[ext[0]]

    if ids_only:
        if as_dataframe:
            ids = feat[ext[0]].index.tolist() # take index of first entry in feature dict
        return ids
    if separate_ids and not as_dataframe:
        return ids, feat
    else:
        return feat


def read_multiple_feature_files(list_of_filenames, common_path = '', feature_types =('rh','ssd','rp'), verbose=True):
    '''Reads multiple feature input files and appends them to make a single feature matrix per feature type
    and a single list of file ids.

    list_of_filenames: Python list with filenames (without extension) of multiple feature files (absolute or relative path)
    common_path: will be added at the beginning of each file name, unless it is '' (default), then file names are treated as absolulte
    feature_types: feature types (i.e. file extensions) to be read in
    returns: tuple of (ids, features), where ids is a list of filenames that belong to the feature vects,
        features is a dict which contains one element per feature type which is a Numpy array which contains feature vectors row-wise
    '''
    import numpy as np

    if isinstance(list_of_filenames, str): # make list if only 1 string is passed
        list_of_filenames = [list_of_filenames]

    ids_out = []
    feat_out = {}

    for filen in list_of_filenames:
        ids, feat = read_csv_features(common_path + os.sep + filen, feature_types, verbose=verbose, single_id_list=True)
        ids_out.extend(ids)

        for e in feat.keys(): # for each element in dict add to the feat_out dict
            if e not in feat_out.keys(): # first file: create array
                feat_out[e] = feat[e]
            else: # others: append
                feat_out[e] = np.append(feat_out[e],feat[e],axis=0)

    return ids_out, feat_out


def write_features_csv_batch(ids, feat, out_path, verbose=True):
    '''write entire feature matrices to multiple feature files in CSV file format'''

    for ft in feat.keys():

        if isinstance(ids, list):
            ids_df = ids  # one single list with filenames
        else:  # dict
            ids_df = ids[ft]

        # create pandas dataframe for each feature type
        dataframe = pd.DataFrame(feat[ft], index=ids_df)

        # write to output CSV
        outfile = out_path + '.' + ft  # + '.csv'
        if verbose:
            print "Writing", outfile
        dataframe.to_csv(outfile, header=None)


def concat_multiple_csv_feature_files(list_of_filenames, out_path, feature_types):
    '''combine multiple feature files in CSV format to a single one (for each feature type)'''

    ids, feat = read_multiple_feature_files(list_of_filenames, '', feature_types, verbose=True)
    write_features_csv_batch(ids, feat, out_path, verbose=True)


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
    
    


# == HDF5 ==


def load_hdf5_features(hdf_filename, verbose=True, ids_only=False, return_id2=False):
    '''read HFD5 file written with HDF5FeatureWriter() class'''
    import tables  # pytables HDF5 library (installed via pip install tables)
    #hdf5_file = tables.openFile(hdf_filename, mode='r') # tables <= 3.1.1
    hdf5_file = tables.open_file(hdf_filename, mode='r') # tables >= 3.2

    if not ids_only:
        # feature vector table is called 'vec' in HDF5FeatureWriter() class
        # we slice [:] all the data back into memory
        feat = hdf5_file.root.vec[:]

        if verbose: # just for info purposes
            print "Read", feat.shape[0], "features with dimension", feat.shape[1],
            if hdf5_file.root.vec.attrs.__contains__('vec_type'):
                print "type", hdf5_file.root.vec.attrs.vec_type,
            if hdf5_file.root.vec.attrs.__contains__('vec_dim'):
                print "dim", hdf5_file.root.vec.attrs.vec_dim,
            print

    # check if we also have file_ids or file_ids2 tables (see HDF5FeatureWriter() class)
    ids = ids2 = None # default

    if hdf5_file.root.__contains__('file_ids'):
        ids = hdf5_file.root.file_ids[:].tolist()  # [:] = slicing
        if len(ids) == 1 and isinstance(ids[0],list):
            ids = ids[0]  # compatibility to older format where ids were stored in batch as hdf5_file.root.file_ids[:][0].tolist()
        if not ids_only and len(ids) != feat.shape[0]:  # check if length matches feat shape 0
            hdf5_file.close() # close before raising error
            raise ValueError("Number of file ids in file_ids table (" + str(len(ids)) + ") does not match number of features in vec table (" +
                             str(feat.shape[0]) + ").")

    if hdf5_file.root.__contains__('file_ids2'): # check if file_ids2 is present and also read and return
        #ids2 = hdf5_file.root.file_ids2[:][0].tolist()  # old format, before HDF5FeatureWriter() was changed to write file_ids consecutively
        ids2 = hdf5_file.root.file_ids2[:].tolist()
        if len(ids2) == 1 and isinstance(ids2[0],list):
            ids2 = ids2[0]  # compatibility to older format where ids were stored in batch as hdf5_file.root.file_ids[:][0].tolist()
        if not ids_only and len(ids2) != feat.shape[0] and len(ids2) != 0:  # check if length matches feat shape 0 (we accept 0 for empty table here)
            hdf5_file.close()  # close before raising error
            raise ValueError("Number of file ids in file_ids2 table (" + str(len(ids2)) + ") does not match number of features in vec table (" +
                             str(feat.shape[0]) + ").")
        if len(ids2) == 0:
            ids2 = None

    hdf5_file.close()

    if ids_only:
        if return_id2:
            return ids, ids2
        else:
            return ids

    if return_id2:
        return ids, feat, ids2
    else:
        return ids, feat


def load_multiple_hdf5_feature_files(filename_stub, feature_types, h5ext='h5', as_dataframe=False, ids_only=False, verbose=True): # , return_id2=False
    '''load multiple hdf5 feature files into dicts of features and ids'''

    # make list if only 1 string is passed
    if isinstance(feature_types, str):
        feature_types = [feature_types]

    # create result dicts
    feat = {}
    ids = {}
    for e in feature_types:
        filename = filename_stub + '.' + e + '.' + h5ext
        result = load_hdf5_features(filename, verbose, ids_only, return_id2=False) # return_id2=False hardcoded; enable as param if needed
        if ids_only:
            ids[e] = result
        else:
            ids[e], feat[e] = result

            if as_dataframe:
                feat[e] = pd.DataFrame(feat[e], index=ids[e])

    # check ids for consistency
    check_id_consistency(ids)

    if ids_only:
        return ids[feature_types[0]]

    return ids, feat


def load_hdf5_pandas(hdf_filename):
    '''read HFD5 file written with csv2hdf5()'''
    store = pd.HDFStore(hdf_filename)
    # .as_matrix(columns=None) converts to Numpy array (of undefined data column types)
    data = store['data'].as_matrix(columns=None)
    store.close()
    return(data)


def combine_multiple_hdf5_files(input_filelist_stubs, output_filestub, feature_types):
    hdf_writer = HDF5FeatureWriter()
    hdf_writer.open(output_filestub,feature_types)

    for i, filename in enumerate(input_filelist_stubs):
        print "Reading file", filename
        ids, feat = load_multiple_hdf5_feature_files(filename, feature_types)

        # check ids for consistency
        check_id_consistency(ids)

        # if ok, collapse to just 1 ids list
        ids = ids[feature_types[0]]

        print "Writing part", i + 1, "..."
        hdf_writer.write_features_batch(ids,feat)

    hdf_writer.close()
    print "DONE:", output_filestub + ".*"


# == GENERIC LOAD FUNCTIONS ==


def load_features(input_path, feature_types, as_dataframe=False, verbose=True):
    '''Generic load function for loading features from CSV or HDF5 files'''
    ids = None

    # 1) check if we have HDF5 files
    import glob
    h5extensions = ['h5', 'hdf5', 'H5', 'HDF5']
    for h5ext in h5extensions:
        if len(glob.glob(input_path + ".*." + h5ext)) > 0:
            ids, feat = load_multiple_hdf5_feature_files(input_path, feature_types, as_dataframe=as_dataframe, h5ext=h5ext, verbose=verbose)
            break

    # TODO: add reading NPZ files

    # 2) otherwise try to read in CSV format
    if ids == None:
        ids, feat = read_csv_features(input_path, feature_types, as_dataframe=as_dataframe, error_on_duplicates=False, verbose=verbose)

    # from the ids dict, we take only the first entry
    ids = ids.values()[0]
    return ids, feat



def load_or_analyze_features(input_path, feature_types = ['rp','ssd','rh'], save_features = False, output_file = None, verbose=True):
    """convenient function that can either load or freshly extract features

    depending if input_path is...
    a) a path: will recursively look for .wav and .mp3 files in path and freshly extract features
    b) a .txt file: will take a list of filenames (one per line) and freshly extract features
    c) a .wav, .mp3, flac or .aif(f) file: will freshly extract features from that file
    d) a set of *.h5 or *.hdf5: will load features in HDF5 format (from multiple files)
    e) another file: will try to load features in multiple csv file feature format
    TODO: .npz
    e) a .npz, .h5 or .hdf5 file: will load the features from that file

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


    # we accept and check for all audio file types supported
    from audiofile_read import get_supported_audio_formats
    audiofile_types = get_supported_audio_formats()

    # these file times mean fresh extract
    extract_file_types = list(audiofile_types)
    extract_file_types.append('.txt')
    extract_file_types = tuple(extract_file_types)

    # if we got a directory, we do analysis, if we got a file of one of the accepted file_types, we load it
    if os.path.isdir(input_path) or input_path.lower().endswith(extract_file_types):  # FRESH ANALYSIS from input path or .txt file

        if verbose:
            print "Performing feature extraction from ", input_path

        # BATCH RP FEATURE EXTRACTION:
        # if output_file is given, will save features, otherwise not
        ids, feat = extract_all_files_generic(input_path,output_file,feature_types,audiofile_types=audiofile_types,verbose=verbose)

    else:
        # LOAD features from feature file(s) (can be standard CSV files or HDF5 files)
        ids, feat = load_features(input_path, feature_types, verbose=verbose)

    return ids, feat




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

    # TODO: is not writing audio filenames from CSV file correctly into HDF5

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

    # TODO: write audio filenames from CSV file correctly into HDF5

    for chunk in csv_reader:
        store.append('data', chunk)
        cnt += chunk.shape[0]
        if verbose: print "processed", cnt, "rows"

    store.close()
    if verbose: print "Finished."


def hdf2csv(in_path, out_path, feature_types, verbose=True):
    '''HDF5 to CSV feature file converter

    in_path: input path + filename stub (without extension) to read HDF5 files from (expected: .h5 extension)
    out_path: input path + filename stub (without extension) to write CSV files to (.type.csv will be added)
    feature_types: string or list of strings with feature types such as 'rp', 'rh', 'ssd', etc.'''

    # read HDF5 files into dict of feature_types
    ids, feat = load_multiple_hdf5_feature_files(in_path, feature_types)

    # write in a batch manner to CSV files
    write_features_csv_batch(ids, feat, out_path, verbose)



# == FEATURE MANIPULATION ==

def concatenate_features(feat, feature_types = ('ssd', 'rh')):
    ''' concatenate features vectors of various feature types together

    All entries in feature dictionary MUST have the same number of instances, but may have different dimensions.

    feat: feature dictionary, containing np.arrays for various feature types (named 'ssd', 'rh', etc.)
    feature_types: tuple or list of strings with the names of feature types to concatenate. also allowed:
        - 1 entry tuple or single feature type as string: will return this feature type only, without concatenation
        - string with +, e.g. "rp+ssd" to define which features shall be combined
    '''
    import numpy as np

    if isinstance(feature_types, str):
        if '+' in feature_types:  # we allow something like "rp+ssd"
            feature_types = feature_types.split('+')
        else: # in case only 1 string was passed instead of tuple or list we return directly
            return feat[feature_types]

    # stack the features:
    # take the first feature type
    new_feat = feat[feature_types[0]]
    # and iteratively horizontally stack the remaining feature types
    for e in feature_types[1:]:
        new_feat = np.hstack((new_feat, feat[e]))
    return new_feat


def sorted_feature_subset(features, ids_orig, ids_select):
    '''
    selects a (sorted) subset of the original features array
    features: a feature dictionary, containing multiple Numpy arrays or Pandas dataframes, one for each feature type ('rh', 'ssd', etc.)
    ids_orig: original labels/ids for the features, MUST be same length AND order as feature arrays (if Numpy arrays are used)
    ids_select: set of ids in different order or subset of ids to be selected from original feature arrays
    returns: sorted subset of the original features array
    '''
    new_feat = {}
    for e in features.keys():
        if isinstance(features[e], pd.core.frame.DataFrame):
            # if features are already in a dataframe we just subindex
            new_feat[e] = features[e].ix[ids_select]
        else:
            # otherwise we create dataframe, subindex and return just values
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




if __name__ == '__main__':

    import argparse
    argparser = argparse.ArgumentParser() #formatter_class=argparse.ArgumentDefaultsHelpFormatter) # formatter_class adds the default values to print output

    # TODO add and test nargs="+" option to allow -concat option below
    argparser.add_argument('input_path', help='input feature file or file path to search for wav/mp3 files to analyze')
    argparser.add_argument('-out', '--output_filestub', nargs='?', help='output path + filename stub for output feature file (without extension)', default=None) # nargs='?' to make it optional

    # test loading
    argparser.add_argument('-arff',   action='store_true',help='test loading of ARFF file',default=False) # boolean opt
    argparser.add_argument('-csv',   action='store_true',help='test loading of CSV file',default=False) # boolean opt
    argparser.add_argument('-h5','--hdf5', action='store_true',help='test loading of HDF5 file',default=False) # boolean opt
    argparser.add_argument('-test',   action='store_true',help='test some custom stuff',default=False) # boolean opt

    # converters
    #argparser.add_argument('-concat', action='store_true',help='concatenate multiple CSV-based feature files to 1 (for each feature type) - specify multiple input_paths',default=False) # boolean opt
    argparser.add_argument('-csv2arff', action='store_true',help='convert CSV file to ARFF file',default=False) # boolean opt
    argparser.add_argument('-hdf2csv', action='store_true',help='convert HDF5 files to CSV files',default=False) # boolean opt


    args = argparser.parse_args()

    feature_types = ['rp', 'ssd', 'rh'] #, 'mvd']

    if args.csv2arff: # CSV to ARFF converter

        if args.output_filestub is None:
            raise ValueError("Need output_filestub defined on command line for target file.")

        csv2arff(args.input_path,args.output_filestub,feature_types)

    elif args.hdf2csv:  # HDF5 to CSV converter

        if args.output_filestub is None:
            raise ValueError("Need output_filestub defined on command line for target file.")

        hdf2csv(args.input_path, args.output_filestub, feature_types)

    else:
        print "Reading", args.input_path

        if args.arff: # try to load ARFF
            features, classes = load_arff(args.input_path)
            print "Classes:" , classes.shape
        elif args.hdf5: # try to load HDF5
            ids, features = load_hdf5_features(args.input_path)
            print "Number of file ids:", len(ids)
        elif args.test: # testing some stuff
            #ids, feat = load_hdf5_features(args.input_path, verbose=True, ids_only=False, return_id2=False)
            ids = load_multiple_hdf5_feature_files(args.input_path, feature_types, verbose=True, ids_only=True)
            print ids
            sys.exit()
        else: # if args.csv: # try to load CSV
            ids, features = read_csv_features1(args.input_path)
            print "Number of file ids:", len(ids)

        print "Feature dimensions:", features.shape
        #print features
        #print ids
