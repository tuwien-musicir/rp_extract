

''' Classes_Io

    functions to read and write comma or tab separated class list files, i.e.

    <audiofilename>\t<class label>

    additional functions to create, maintain, edit and query a <filename>: <classlabel> dictionary

    __author__ = 'Thomas Lidy'
'''


import os


def read_class_file(filename, delimiter='\t',as_dict=True):
    ''' Read_Class_File

    read a comma or tab separated file providing class labels to analyzed audio files, typically in the format:
    <audio file name or id> <class_label>

    :param filename: input filename to read class labels from
    :param delimiter: separator in the input file: \t by default, can be set to ',', ';' or anything else needed
    :param as_dict: True by default, will return a dict with file ids as key and class label as value
            if False, it will return a list of lists, each list entry containing a "tuple" of file id and label
    :return:
    '''

    import csv
    fi = open(filename, 'r')
    reader = csv.reader(fi, delimiter=delimiter)
    result = dict(reader) if as_dict else list(reader)
    fi.close()
    return(result)


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

    # this example works for GTZAN collection: class is first part of filename before '.'
    classes = [x.split(split_char, 1)[0] for x in filenames]
    return classes


def classes_to_numeric(class_labels,verbose=True):
    '''Classes_to_Numeric

    encode string class labels to numeric values

    will return encoded numeric classes

    Note:  to transform (predicted) numeric classes back to strings use as follows:

    > labelenc.transform(class_labels) # to output numeric classes

    > list(labelenc.inverse_transform([2, 2, 1])) # to transform (predicted) numeric classes back to strings
    '''

    from sklearn.preprocessing import LabelEncoder

    labelenc = LabelEncoder()
    labelenc.fit(class_labels)
    if (verbose): print len(labelenc.classes_), "classes:", list(labelenc.classes_)
    return(labelenc.transform(class_labels))


def classdict_to_numeric(class_dict):
    '''ClassDict_to_Numeric

    in a dictionary containing filenames as keys and class labels as values (e.g.: {'pop.00006.wav': 'pop'})
    encode all string class labels to numeric values (this will create a new dictionary)
    '''

    # this will create a new dict with old keys and numeric values
    classes_num = classes_to_numeric(class_dict.values())
    return (dict(zip(class_dict.keys(),classes_num)))


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


def reduce_class_dict_min_instances(class_dict,min_instances=2):
    ''' reduce a {filename: class} dictionary to retain classes only with a minimum number of file instances per class
    :param class_dict: a {filename: class} dictionary
    :param min_instances: minimum file instances per class required (default = 2)
    :return: {filename: class} dictionary with entries removed where class does not fulfil minimum requirement
    '''

    classes = class_dict.values()
    class_stats = {c: classes.count(c) for c in set(classes)}

    retain_classes = []
    for key, val in class_stats.iteritems():
        if val >= min_instances: retain_classes.append(key)
    #retain_classes
    diff = len(set(classes)) - len(retain_classes)
    if diff > 0: print "Removing", diff, "classes for required minimum of", min_instances, "instances per class."

    new_class_dict = {}
    for key, val in class_dict.iteritems():
        if val in retain_classes:
            new_class_dict[key] = val

    if diff > 0: print "Removed", len(class_dict) - len(new_class_dict), "file instances from class dictionary."
    return (new_class_dict)

def get_class_counts(class_dict,printit=False):
    '''print number of instances per class in a class_dict'''
    classes = class_dict.values()
    class_stats = {c: classes.count(c) for c in set(classes)}
    if (printit):
        for key, val in class_stats.iteritems():
            print key+":",val
    return (class_stats)



def get_filenames_for_class(class_dict,classname):
    '''Get_Filenames_For_Class

    return filename ids for a selected class

    classname: e.g. 'Jazz'
    '''
    key_list = []
    for key,val in class_dict.iteritems():
        if val == classname: key_list.append(key)
    return(key_list)



# == HELPER FUNCTIONS ==


def read_filenames(filename):
    '''Read_Filenames

    reads a list of audio files to process from a text file (one audio file per line)
    (used instead of find_files)

    :param filename: filename of input text file
    :return: list of audio files to process, read line-wise from filename
    '''

    with open(filename) as f:
        content = [line.rstrip('\n') for line in f]
    return(content)


def write_filenames(filename, filelist):
    '''write a list of filenames to a plain text file (one per line)'''
    fil = open(filename,'w')
    for f in filelist:
        fil.write(f + "\n") # python will convert \n to os.linesep
    fil.close()


def strip_filenames(filenames,cut_path=True, cut_ext=True):
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

