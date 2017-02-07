

''' Classes_Io

    functions to read and write comma or tab separated class list files, i.e.

    <audiofilename>\t<class label>

    additional functions to create, maintain, edit and query a <filename>: <classlabel> dictionary

    __author__ = 'Thomas Lidy'
'''


import os
import sys
import pandas as pd


# --- READ AND WRITE ---

def read_class_file(filename, delimiter='\t', as_dict=True, cut_path=False, cut_ext=False):
    ''' Read Class File

    read a comma or tab separated file providing class labels to analyzed audio files, typically in the format:

    <audio file name or id>\TAB<class_label>

    No CSV header allowed.

    :param filename: input filename to read class labels from
    :param delimiter: separator in the input file: \t by default, can be set to ',', ';' or anything else needed
    :param as_dict: True by default, will return a dict with file ids as key and class label as value
            if False, it will return a list of lists, each list entry containing a "tuple" of file id and label
    :param cut_path: will cut off relative path of file ids in class_file (only possible when as_dict is True)
    :param cut_ext: will cut off file extension of file ids in class_file (only possible when as_dict is True)
    :return:
    '''

    import csv
    fi = open(filename, 'r')
    reader = csv.reader(fi, delimiter=delimiter)
    result = dict(reader) if as_dict else list(reader)
    fi.close()

    if as_dict and (cut_path or cut_ext):
        result = strip_filenames_in_dict(result, cut_path, cut_ext)

    return result


def read_class_file_as_dataframe(filename, delimiter='\t', cut_path=False, cut_ext=False):
    dataframe = pd.read_csv(filename, sep=delimiter, index_col=0, header=None)
    dataframe.columns = ['class']
    if cut_path or cut_ext:
        dataframe.index = strip_filenames(dataframe.index, cut_path, cut_ext)
    return dataframe


def read_multi_class_file(filename, delimiter='\t', stripfilenames=False, replace_labels=True, pos_labels='x', neg_labels='', verbose=True):
    '''read multi label class assignment files in the format (with CSV header):

    filename    genre1  genre2  genre3
    file1       x       x
    file2               x       x
    etc.

    (TAB separated, but can be changed with delimiter parameter)

    :param filename:
    :param delimiter:
    :param stripfilenames:
    :param pos_labels:
    :param neg_labels:
    :param verbose:
    :return:
    '''

    # we use pandas to import CSV as pandas dataframe, because it handles quoted filenames (containing ,) well (by contrast to other CSV readers)
    import numpy as np

    dataframe = pd.read_csv(filename, sep=delimiter, index_col=0)

    # CSV file is supposed to have file names without extension. otherwise do:
    if stripfilenames:
        dataframe.index = strip_filenames(dataframe.index) # in class data

    # get categories from the header
    categories = dataframe.columns.values.tolist()
    if verbose:
        print 'Categories in CSV file:', ", ".join(categories)

    if replace_labels:
        # replace positive labels as 1 and negative or empty as 0
        dataframe.replace(pos_labels, 1, inplace=True)
        dataframe.replace(neg_labels, 0, inplace=True)
        dataframe.fillna(0, inplace=True) # treat empty cells as negative

        # sanity check before we convert to integers
        wrong_entries = np.where(np.logical_and(dataframe.values != 1, dataframe.values != 0))
        if len(wrong_entries[0]) > 0:
            for i,j in zip(wrong_entries[0], wrong_entries[1]):
                print "Unrecognized entry in row", i+1, ", column", j+1, ":"
                print dataframe.index[i] + ": '" + dataframe.iloc[i,j] + "'"
            raise ValueError("CSV contains unrecognized entries. Please correct CSV file or define pos_labels when calling read_multi_class_file function.")

    # make an in-place conversion to integer (if not possible, will throw error)
    dataframe = dataframe.astype(int, copy=False)

    return dataframe


def write_class_file(filename, file_ids, class_labels, delimiter='\t'):
    fil = open(filename,'w')
    for f, c in zip(file_ids, class_labels):
        fil.write(f + delimiter + c + '\n') # python will convert \n to os.linesep
    fil.close()


def write_class_dict(filename, class_dict, delimiter='\t'):
    fil = open(filename,'w')
    for f, c in class_dict.iteritems():
        fil.write(f + delimiter + c + '\n') # python will convert \n to os.linesep
    fil.close()


def write_multi_class_table(filename, ids, predictions, class_columns, pos_label='x', neg_label=''):
        pred_df = pd.DataFrame(predictions, index=ids, columns=class_columns)
        pred_df.replace(0, neg_label, inplace=True)
        pred_df.replace(1, pos_label, inplace=True)

        with open(filename, 'w') as file:
            pred_df.to_csv(file, sep='\t')


def multi_class_table_tolist(dataframe, pos_label=1, delimiter=','):
    '''convert a multi-class table (dataframe with 'x' on positive classes) to a comma-separated list of (positive) classes'''

    list_of_lists = []
    for idx, row in dataframe.iterrows():
        row_list = [c for c, r in row.iteritems() if r==pos_label]
        row_string = delimiter.join(row_list)
        list_of_lists.append(row_string)

    ids = dataframe.index.values.tolist()
    return ids, list_of_lists


def multi_class_table_todict(dataframe, pos_label=1):
    '''convert a multi-class table (dataframe with 'x' on positive classes) to a dict
     containing comma-separated lists of (positive) classes for each audio file'''

    class_dict = {}
    for id, row in dataframe.iterrows():
        row_list = [c for c, r in row.iteritems() if r==pos_label]
        class_dict[id] = row_list

    return class_dict


def write_multi_class_list(filename, ids, class_lists, ids2=None, delimiter='\t'):
    fil = open(filename,'w')
    if ids2 is None:
        for id, classes in zip(ids, class_lists):
            fil.write(id + delimiter + classes + '\n')
    else:
        for id, id2, classes in zip(ids, ids2, class_lists):
            fil.write(id + delimiter + id2 + delimiter + classes + '\n')
    fil.close()


# --- HANDLING CLASS DATA ---


def classes_from_filename(filenames,split_char=os.sep):
    '''Classes_From_Filename

    derive class label from filename or relative file path
    this function derives class labels from the document file names (ids) given in the original feature files

    # TODO: adapt to enable splitting by LAST appearance of split_char instead of first

    Examples:
    # split class by first / or \ (os.sep) as e.g. in "pop/file1.wav"
    >>>classes = classes_from_filename(ids[ext])
    # split class by first '.' as used e.g. in GTZAN collection: "pop.00001.wav"
    >>>classes = classes_from_filename(ids[ext],'.')
    '''

    # we try to catch here the case that file was created on Windows but is used on Linux or vice-versa (so we reverse the split_char)
    if split_char==os.sep:
        example_file = filenames[0]
        if os.sep == '/' and example_file.count(os.sep)==0: split_char = '\\'
        if os.sep == '\\' and example_file.count(os.sep)==0: split_char = '/'

    # this example works for GTZAN collection: class is first part of filename before '.'
    classes = [x.split(split_char, 1)[0] for x in filenames]

    # sanity check
    if len(set(classes)) == len(filenames):
        print "Example file name:", filenames[0]
        print "Example class label:", classes[0]
        raise ValueError("Class label could not be derived from filename! Filename must contain a sub-directory path to be used as class label. Otherwise provide a class label file.")

    return classes

def classdict_from_filename(filenames,split_char=os.sep):
    '''derive class labels from filenames or relative paths and create a class dict out of both of them'''
    classes = classes_from_filename(filenames,split_char)  # get classes from relative path
    class_dict = dict(zip(filenames, classes))    # Note: if the lists are big you should use from itertools import izip
    return class_dict


def classes_to_numeric(class_labels, verbose=True, return_encoder = False):
    '''Classes_to_Numeric

    encode string class labels to numeric values

    will return encoded numeric classes

    Note:  to transform (predicted) numeric classes back to strings use as follows:

    > labelenc.transform(class_labels) # to output numeric classes

    > list(labelenc.inverse_transform([2, 2, 1])) # to transform (predicted) numeric classes back to strings
    '''

    from sklearn.preprocessing import LabelEncoder

    labelencoder = LabelEncoder()
    labelencoder.fit(class_labels)
    if (verbose): print len(labelencoder.classes_), "classes:", ", ".join(list(labelencoder.classes_))
    classes_num = labelencoder.transform(class_labels)
    if return_encoder:
        return (classes_num, labelencoder)
    else:
        return classes_num


def classdict_to_numeric(class_dict, return_encoder = False):
    '''ClassDict_to_Numeric

    in a dictionary containing filenames as keys and class labels as values (e.g.: {'pop.00006.wav': 'pop'})
    encode all string class labels to numeric values (this will create a new dictionary)
    '''

    # this will create a new dict with old keys and numeric values
    if return_encoder:
        classes_num, labelencoder = classes_to_numeric(class_dict.values(), return_encoder=return_encoder)
    else:
        classes_num = classes_to_numeric(class_dict.values())

    new_class_dict = (dict(zip(class_dict.keys(),classes_num)))

    if return_encoder:
        return (new_class_dict, labelencoder)
    else:
        return new_class_dict


def classdict_to_numeric_with_encoder(class_dict, labelencoder):
    '''encode a the values in a given class dictionary with a given label encoder to numeric classes'''

    classes_num = labelencoder.transform(class_dict.values())
    return dict(zip(class_dict.keys(), classes_num))



def get_classes_from_dict(class_dict,filenames):
    '''Get_Classes_From_Dict

    get multiple class values at once (as a list) for multiple file ids in a class label dictionary

    :param class_dict: a dictionary containing filenames as keys and class labels as values (e.g.: {'pop.00006.wav': 'pop'})
    :param filenames: a list of filenames to be queried as keys in this dictionary
    :return: list of class values (string or numeric, depending of the composition of the given dictionary)

    also see classdict_to_numeric
    '''
    return([class_dict.get(key) for key in filenames])


def reduce_class_dict(class_dict,new_file_ids):
    '''reduce a {filename: class} dictionary to a subset of 'new_file_ids'
    all new_file_ids must be contained as keys in the given class_dict
    '''
    # check if all new_file_ids are contained in the original class_dict
    #if len(set(new_file_ids) - set(class_dict.keys())) > 0
    # we avoid this check because for key will throw an error anyway
    new_class_dict = { key: class_dict[key] for key in new_file_ids }
    return (new_class_dict)


def match_filenames(file_ids_featurefile, file_ids_classfile, strip_files=False, lower=False, verbose=True, print_nonmatching=True):
    '''Match file ids in audio feature files and class files.

    returns the set of overlapping filenames (file ids) of the two lists of file ids
    (one from the class file, one from the feature file(s))

    :param strip_files:
    :return: file_ids_matched
    '''
    from rp_feature_io import check_duplicates

    if strip_files:
        file_ids_classfile = strip_filenames(file_ids_classfile)
        file_ids_featurefile = strip_filenames(file_ids_featurefile)

    if lower:
        file_ids_classfile = [s.lower() for s in file_ids_classfile]
        file_ids_featurefile = [s.lower() for s in file_ids_featurefile]

    check_duplicates(file_ids_classfile)
    check_duplicates(file_ids_featurefile)

    file_ids_matching = set(file_ids_classfile).intersection(file_ids_featurefile)

    if verbose:
        print len(file_ids_featurefile), "files in feature file(s)"
        print len(file_ids_classfile), "files in class file"
        print len(file_ids_matching), "files matching"

    if print_nonmatching:  # output missing files

        diff = set(file_ids_classfile) - set(file_ids_matching)
        if len(diff) > 0:
            print
            print 'in class definition but not in audio feature files:\n'
            for f in diff: print f

        diff = set(file_ids_featurefile) - set(file_ids_matching)
        if len(diff) > 0:
            print
            print 'in audio feature files but not in class definition:\n'
            for f in diff: print f

    return list(file_ids_matching)


def align_features_and_classes(features, feature_ids, class_data, strip_files=False, lower=False, verbose=True):

    '''match the ids of the features and the class dictionary/dataframe which could be in different order,
    or match only case insensitive or without extension.

    finds the intersecting subset of ids among the two and reduces both the features and the class_data to the
    matching ids, ensuring same order

    features: dictionary with multiple numpy arrays, on per each feature type
    feature_ids: list of strings containing the ids for the features (must be same length as rows in feature arrays)
    class_data: class dict or dataframe containing single or multi-class information (respectively), with file id in first resp. index column
    strip_files: whether or not to remove file extensions before matching
    lower: whether or not to lower-case all characters before matching
    verbose: output statistics how many are being matched and the list of non-matched files
    '''
    from rp_feature_io import sorted_feature_subset

    if isinstance(class_data, dict):
        file_ids_classfile = class_data.keys()
    elif isinstance(class_data, pd.DataFrame):
        file_ids_classfile = list(class_data.index)
    else:
        raise ValueError("Class data must be passed as Python dict or Pandas dataframe!")

    ids_matched = match_filenames(feature_ids, file_ids_classfile, strip_files, lower, verbose=verbose, print_nonmatching=verbose)

    # Note: sorting or not sorting changes the results of cross-validation!
    # ids_matched = sorted(ids_matched)

    if isinstance(class_data, dict):
        class_data = reduce_class_dict(class_data, ids_matched)
        n_class_entries = len(class_data)
    if isinstance(class_data, pd.DataFrame):
        # create a new reduced dataframe that contains only the matched files (in the matched order)
        class_data = class_data.ix[ids_matched]
        n_class_entries = class_data.shape[0]

    if strip_files: # even if don already in match_filenames, we need to strip here again for sorted_feature_subset below
        feature_ids = strip_filenames(feature_ids)

    # cut & resort the features according to matched ids (subset, if files are missing in class file)
    features = sorted_feature_subset(features, feature_ids, ids_matched)

    if verbose: print "\nRetaining", features.values()[0].shape[0], "feature rows,", n_class_entries, "class entries."

    return features, ids_matched, class_data



def align_predictions_and_groundtruth(pred_df, groundtruth_df, strip = True, lower = False, verbose=True):
    # a) check column names
    if pred_df.columns.tolist() != groundtruth_df.columns.tolist():
        raise ValueError('Column names in groundtruth and predictions do not match!')

    # b) align filenames
    filenames1 = list(pred_df.index)
    filenames2 = list(groundtruth_df.index)

    if strip:
        filenames1 = strip_filenames(filenames1)
        filenames2 = strip_filenames(filenames2)

    if lower:
        filenames1 = [s.lower() for s in filenames1]
        filenames2 = [s.lower() for s in filenames2]

    # assign altered index back, otherwise .loc below does not work
    pred_df.index = filenames1
    groundtruth_df.index = filenames2

    files_matched = match_filenames(filenames1, filenames2, verbose=verbose, print_nonmatching=verbose)

    # from the given dataframes, cut & sort only the matched file ids to align them
    pred_df_sorted = pred_df.loc[files_matched]
    groundtruth_df_sorted = groundtruth_df.loc[files_matched]

    return (pred_df_sorted, groundtruth_df_sorted)


# OBSOLETE?
def match_and_reduce_class_dict(class_dict,new_file_ids,strip_files = True):
    '''check for matching file ids in a class dictionary and reduce the class dictionary to the matching ones
    :param class_dict:
    :param new_file_ids:
    :return:
    '''
    if strip_files:
        new_file_ids = strip_filenames(new_file_ids)
    print len(class_dict), "files in class definition file"
    print len(new_file_ids), "files from audio feature analysis"
    matching = set(class_dict.keys()).intersection(new_file_ids)
    print len(matching), "files matching"
    new_class_dict = reduce_class_dict(class_dict,matching)
    return new_class_dict


def reduce_class_dict_to_classes(class_dict,reduced_list_of_classes):
    '''reduce a {filename: class} dictionary to a subset of classes given in list 'reduced_list_of_classes' '''
    new_class_dict = {}
    for key, val in class_dict.iteritems():
        if val in reduced_list_of_classes:
            new_class_dict[key] = val
    return new_class_dict


def reduce_class_dict_min_instances(class_dict, min_instances=2, raiseError=False):
    ''' reduce a {filename: class} dictionary to retain classes only with a minimum number of file instances per class
    :param class_dict: a {filename: class} dictionary
    :param min_instances: minimum file instances per class required (default = 2)
    :param raiseError: will produce error instead of reducing the class_dict, if requirement is not fulfilled
    :return: {filename: class} dictionary with entries removed where class does not fulfil minimum requirement
    '''

    classes = class_dict.values()
    class_stats = {c: classes.count(c) for c in set(classes)}

    retain_classes = []
    for key, val in class_stats.iteritems():
        if val >= min_instances: retain_classes.append(key)
    #retain_classes
    diff = len(set(classes)) - len(retain_classes)
    if diff > 0:
        if raiseError:
            raise ValueError("Class requirement of minimum of", min_instances, "instances per class not fulfilled.")
        else:
            print "Removing", diff, "classes for required minimum of", min_instances, "instances per class."

    new_class_dict = reduce_class_dict_to_classes(class_dict, retain_classes)

    if diff > 0:
        print "Removed", len(class_dict) - len(new_class_dict), "file instances from class dictionary."

    return new_class_dict


def get_class_counts(class_data, printit=False):
    '''print number of instances per class in class assignments (groundtruth)

       class_data must be passed Python dict or Pandas dataframe'''

    if isinstance(class_data, dict):
        classes = class_data.values()
    elif isinstance(class_data, pd.DataFrame):
        classes = class_data.ix[:, 0].tolist()
    else:
        raise ValueError("Class data must be passed as Python dict or Pandas dataframe!")

    class_stats = {c: classes.count(c) for c in set(classes)}
    if (printit):
        for key, val in class_stats.iteritems():
            print key + ":", val
    return class_stats


def get_filenames_for_class(class_dict,classname):
    '''Get_Filenames_For_Class

    return filename ids for a selected class

    classname: e.g. 'Jazz'
    '''
    key_list = []
    for key,val in class_dict.iteritems():
        if val == classname: key_list.append(key)
    return key_list


def get_baseline(class_data, printit=True):
    '''Print classification baseline according to class with maximum instances

       class_data must be passed Python dict or Pandas dataframe'''

    class_counts = get_class_counts(class_data)
    # print "Class counts:", class_counts
    max_class = max(class_counts.values())
    baseline = max_class * 1.0 / len(class_data)
    if printit: print "Baseline: %.2f %% (max class=%d/%d)" % ((baseline * 100), max_class, len(class_data))
    return baseline


# == HELPER FUNCTIONS ==


def read_filenames(filename):
    '''Read_Filenames

    reads a list of audio files to process from a text file (one audio file per line)
    (used instead of find_files)

    :param filename: filename of input text file
    :return: list of audio files to process, read line-wise from filename
    '''

    with open(filename) as f:
        content = [line.rstrip('\r\n') for line in f]
    return(content)


def write_filenames(filename, filelist):
    '''write a list of filenames to a plain text file (one per line)'''
    fil = open(filename,'w')
    for f in filelist:
        fil.write(f + "\n") # python will convert \n to os.linesep
    fil.close()


def strip_filenames(filenames, cut_path=True, cut_ext=True):
    '''Strip_Filenames

    strips off the preceding paths and/or the extensions of all given filenames in an array of filenames
    :param filenames: array of filenames (possibly including absolute or relative path)
    :param cut_path: whether or not to cut away the preceding path (leaving filename only)
    :param cut_ext: whether or not to cut away the extension of the filename
    :return: array of filenames only without path
    '''
    from os.path import basename, splitext
    if (cut_path): filenames = ([basename(f) for f in filenames])
    if (cut_ext): filenames = ([splitext(f)[0] for f in filenames])
    return(filenames)

def strip_filenames_in_dict(class_dict, cut_path=True, cut_ext=True):
    from os.path import basename, splitext
    new_class_dict = {}
    for key, val in class_dict.iteritems():
        if (cut_path): key = basename(key)
        if (cut_ext): key = splitext(key)[0]
        new_class_dict[key] = val
    return new_class_dict