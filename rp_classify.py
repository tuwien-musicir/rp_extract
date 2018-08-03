#!/usr/bin/env python

'''
Classification of Music into Genres, Moods or other categories
using the Rhythm Pattern audio analyzer (rp_extract.py)

2015-11 - 2016-08 by Thomas Lidy
'''

import os.path
import argparse
import cPickle
import numpy as np

from sklearn import preprocessing, svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
# sklearn <= 0.17
#from sklearn.cross_validation import cross_val_score
# sklearn >= 0.18
from sklearn.model_selection import cross_val_score

from rp_feature_io import load_or_analyze_features, concatenate_features
from classes_io import *


# STANDARDIZE DATA
def standardize(data, return_scaler = True):
    if return_scaler:
        # STANDARDIZATION (0 mean, unit var)
        scaler = preprocessing.StandardScaler()
        # alternative: NORMALIZATION (min - max Normalization to (0,1))
        #scaler = preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)
        return (data, scaler)
    else:
        return preprocessing.scale(data,axis=0)
        # axis=0 means independently standardize each feature, otherwise (if 1) standardize each sample

# how to get scaler parameters
#print scaler.mean_
#print scaler.scale_


# TRAIN
def train_model(train_data, train_classes, print_accuracy = True): # with_probabilities = False,
    '''train a SVM classifier model'''
    #model = svm.SVC() # defaults to kernel='rbf' (usually worse for RP/SSD/RH features)
    #model = svm.SVC(kernel='linear', probability=with_probabilities)

    # we use this syntax as it supports multi-class classification
    model = OneVsRestClassifier(SVC(kernel='linear'))

    model.fit(train_data, train_classes)

    if print_accuracy:
        pred_train = model.predict(train_data) # predictions on train set
        # for multi-class train sets we do the accuracy column-wise and then compute the mean over all accuracies
        acc_per_column = np.sum(pred_train == train_classes, axis=0) * 100.0 / len(train_classes)
        mean_acc = np.mean(acc_per_column)
        print "Accuracy on train set: %2.2f %%" % mean_acc

    return model

# CLASSIFY
def classify(model, features, labelencoder = None):
    pred = model.predict(features)
    if labelencoder:
        return labelencoder.inverse_transform(pred)
    else:
        return pred

# CROSS VALIDATION
def cross_validate(model, features, classes, folds=10, measure='accuracy'):
    return cross_val_score(model, features, classes, scoring=measure, cv=folds)
    # scoring value: Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro',
    # 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error',
    # 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted',
    # 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']


# CROSS VALIDATION (for multi-class predictions)
def cross_validate_multiclass(model, features, classes, categories, folds=10, measure='accuracy', verbose=True):
    acc = [] # empty list
    # we iterate over the categories in class file columns here
    for c in range(len(categories)):
        if verbose:
            print '.',
            sys.stdout.flush()
        cls = classes[:,c]
        a = cross_val_score(model, features, cls, scoring=measure, cv=folds)
        mean_acc = np.mean(a) # a contains the scores per fold, we average over folds
        acc.append(mean_acc)

        # TEST:
        # other way to do it:
        #from sklearn import metrics
        #predicted = cross_val_predict(model, features, cls, cv=folds)
        #mean_acc2 = metrics.accuracy_score(cls, predicted)
        #mean_prec = metrics.precision_score(cls, predicted)
        #print "New Acc/Prec: %s\t%2.2f %%\t%2.2f %%" % (c,mean_acc2*100,mean_prec*100)

    if verbose:
        print
        sys.stdout.flush()
    return zip(categories,acc)


# SAVE MODEL
def save_model(filename,model,scaler=None,labelencoder=None,multi_categories=None):
    basename = os.path.splitext(filename)[0]
    with open(basename + ".model.pkl", 'wb') as f:
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    if scaler:
        with open(basename + ".scaler.pkl", 'wb') as f:
            cPickle.dump(scaler, f, protocol=cPickle.HIGHEST_PROTOCOL)
    if labelencoder:
        with open(basename + ".labelenc.pkl", 'wb') as f:
            cPickle.dump(labelencoder, f, protocol=cPickle.HIGHEST_PROTOCOL)
    if multi_categories:
        with open(basename + ".multilabels.csv", 'w') as f:
            f.write('\n'.join(multi_categories))


# LOAD MODEL
def load_model(filename,scaler=True,labelencoder=True,multilabels=False):
    basename = os.path.splitext(filename)[0]

    # load model
    f = open(basename + ".model.pkl", 'rb')
    model = cPickle.load(f)
    f.close()

    if scaler:
        f = open(basename + ".scaler.pkl", 'rb')
        scaler = cPickle.load(f)
        f.close()
    else: scaler = None

    if labelencoder:
        labelfile = basename + ".labelenc.pkl"
        if os.path.isfile(labelfile): # check if exists
            f = open(labelfile, 'rb')
            labelencoder = cPickle.load(f)
            f.close()
        else: labelencoder = None
    else: labelencoder = None

    if multilabels:
        multilabelfile = basename + ".multilabels.csv"
        if os.path.isfile(multilabelfile): # check if exists
            with open(multilabelfile) as f:
                multi_categories = [line.rstrip('\n') for line in f]
        else: multi_categories = None
        # if multilabels is True we return with multi_categories otherwise we omit 4th return value
        return (model,scaler,labelencoder,multi_categories)

    return (model,scaler,labelencoder)



if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Classification of music files into genre, mood or other categories (with optional training of own models).') #formatter_class=argparse.ArgumentDefaultsHelpFormatter) # formatter_class adds the default values to print output

    argparser.add_argument('input_path', help='input file path to search for wav/mp3/m4a/aif(f) files')
    argparser.add_argument('model_file', nargs='?', help='model file name (input filename for predictions (if omitted, default in models folder is used), or output file to write after training)')
    argparser.add_argument('output_filename', nargs='?', help='filename for predictions to write (if omitted, will print predictions to stdout)') # nargs='?' to make it optional

    argparser.add_argument('-t','--train',action='store_true',help='train a model with the input data',default=False) # boolean opt
    argparser.add_argument('-c', '--classfile', help='single label class file for training and/or cross-validation (format: <filename>TAB<class_string>)',default=None)
    argparser.add_argument('-m', '--multiclassfile', help='multi label class file for training and/or cross-validation (format: <filename>  x  x     x)',default=None)
    argparser.add_argument('-cv','--crossval',action='store_true',help='cross-validate accuracy on the input data',default=False) # boolean opt
    argparser.add_argument('-cvp','--crossvalprec',action='store_true',help='cross-validate precision on the input data',default=False) # boolean opt

    argparser.add_argument('-mot','--multiclasstable',action='store_true',help='write multi-class table instead of list',default=False) # boolean opt

    argparser.add_argument('-strip','--stripfeat',action='store_true',help='strip off file extensions in feature files (to match with classfile)',default=False) # boolean opt

    argparser.add_argument('-rh',   action='store_true',help='use Rhythm Histograms (default)',default=False) # boolean opt
    argparser.add_argument('-ssd',  action='store_true',help='use Statistical Spectrum Descriptors (default)',default=False) # boolean opt
    argparser.add_argument('-rp',   action='store_true',help='use Rhythm Patterns',default=False) # boolean opt
    argparser.add_argument('-trh',  action='store_true',help='use Temporal Rhythm Histograms',default=False) # boolean opt
    argparser.add_argument('-tssd', action='store_true',help='use Temporal Statistical Spectrum Descriptors',default=False) # boolean opt
    argparser.add_argument('-mvd',  action='store_true',help='use Modulation Frequency Variance Descriptors',default=False) # boolean opt
    argparser.add_argument('-3',    action='store_true',help='(shortcut flag:) use RH + SSD + RH',default=False) # boolean opt

    args = argparser.parse_args()
    argsdict = vars(args)  # needed only for numeric flag (-3)

    # select the feature types according to given option(s) or default
    feature_types = []
    if args.rh: feature_types.append('rh')
    if args.ssd: feature_types.append('ssd')
    if args.rp: feature_types.append('rp')
    if argsdict['3']:feature_types = ['ssd','rh','rp']  # keep it at this position to avoid duplicate definition
    if args.trh: feature_types.append('trh')
    if args.tssd: feature_types.append('tssd')
    if args.mvd: feature_types.append('mvd')

    # if none was selected set DEFAULT feature set
    if feature_types == []: feature_types = ['ssd','rh']

    # TRAINING / CROSS-VALIDATION
    do_crossval = args.crossval or args.crossvalprec
    if do_crossval:
        crossval_folds = 10
        crossval_measure = 'precision' if args.crossvalprec else 'accuracy'

    if args.train or do_crossval:

        if args.train and not do_crossval and args.model_file is None:
            raise ValueError("model_file must be provided to store when training a model.")

        # LOAD OR EXTRACT FEATURES
        # TODO: store and load feature extraction parameters with model
        ids, feat = load_or_analyze_features(args.input_path, feature_types)

        # CLASSES: read from file or derive from sub-path
        if args.classfile or args.multiclassfile:
            if args.stripfeat:
                ids = strip_filenames(ids)  # strip filenames of audio feature files

            if args.classfile:
                class_dict = read_class_file(args.classfile)
            elif args.multiclassfile:
                class_dict = read_multi_class_file(args.multiclassfile, replace_labels=True, pos_labels=('x','x ','X/2')) # class_dict here is in fact a dataframe

            feat, ids, class_dict = align_features_and_classes(feat, ids, class_dict, strip_files=True)
            if len(ids) == 0:
                raise ValueError("No features could be matched with class file! Cannot proceed.")
        else:
            # try to derive classes from filename (e.g. sub-directory)
            classes = classes_from_filename(ids)
            class_dict = dict(zip(ids, classes))

        # convert to numeric classes
        if args.multiclassfile:
            # multi-class files are already converted to numeric in read function
            # - just get the numeric classes from the class dataframe's values
            classes_num = class_dict.as_matrix()
            # get the categories from the header
            multi_categories = class_dict.columns.values.tolist()
            labelencoder = None # no label encoder for multi-class files
        else:
            (class_dict_num, labelencoder) = classdict_to_numeric(class_dict, return_encoder = True)
            classes_num = get_classes_from_dict(class_dict_num,ids)
            multi_categories = None

        # CONCATENATE MULTIPLE FEATURES
        # (optional but needs to be done in same way at prediction time)
        features = concatenate_features(feat, feature_types)
        print "Using features:", " + ".join(feature_types)

        # STANDARDIZE
        features, scaler = standardize(features)

        # TRAIN + SAVE MODEL
        if args.train:
            print "Training model:"
            model = train_model(features, classes_num)
            # save model
            save_model(args.model_file, model, scaler, labelencoder, multi_categories)
            print "Saved model to", args.model_file + ".*"

        # CROSS-VALIDATE
        if do_crossval:
            print "CROSS-VALIDATION:"
            if not args.train:
                # if we trained, we have a model already; otherwise we initialize a fresh one
                model = svm.SVC(kernel='linear')

            if not args.multiclassfile:
                acc = cross_validate(model, features, classes_num, crossval_folds, crossval_measure)
                print "Fold " + crossval_measure + ":", acc
                print "Avg " + crossval_measure + " (%d folds): %2.2f %% (std.dev.: %2.2f)" % (len(acc), (np.mean(acc)*100), np.std(acc)*100)
            else:
                acc_zip = cross_validate_multiclass(model, features, classes_num, multi_categories, crossval_folds, crossval_measure, verbose=False)
                for c, a in acc_zip:
                    print "Class: %s\t%2.2f %%" % (c,a*100)
                acc_per_class = zip(*acc_zip)[1] # unzip to 2 lists and take second one
                avg_acc = np.mean(acc_per_class)
                print "Average " + crossval_measure + ":\t%2.2f %%" % (avg_acc*100)

    else: # do CLASSIFICATION only when not training

        # check for inappropriate parameters
        if args.classfile or args.multiclassfile:
            raise SyntaxError("Class file can only be provided when training with -t parameter or cross-validating with -cv.")

        # LOAD MODEL
        if args.model_file is None:
            args.model_file = 'models/GTZAN'   # default model file

        if not args.train: # if we train + classify in one step, we don't need to load the model
            # we always try to get a multi labels file (if we can't find it, multi_categories will be None; vice-versa for labelencoder)
            model, scaler, labelencoder, multi_categories = load_model(args.model_file, multilabels=True)

        # info print
        #if multi_categories:
        #    print "Multiple categories to predict:", ", ".join(multi_categories)

        # EXTRACT FEATURES FROM NEW FILES
        ids, feat = load_or_analyze_features(args.input_path, feature_types)

        if len(feat) == 0:
            raise ValueError("No features were extracted from input files. Check format.")

        # SELECT OR CONCATENATE FEATURES
        features_to_classify = concatenate_features(feat, feature_types)

        # SCALE FEATURES LIKE TRAINING DATA
        if features_to_classify.shape[1] != len(scaler.scale_):
            raise ValueError("Features have "+ str(features_to_classify.shape[1]) + " dimensions, but "
                             "StandardScaler was saved with " + str(len(scaler.scale_)) + " dimensions. Feature mismatch!")

        features_to_classify = scaler.transform(features_to_classify)

        # CLASSIFY
        print "Using features:", " + ".join(feature_types)
        print "Classification:"
        if labelencoder:
            print len(labelencoder.classes_), "possible classes:", ", ".join(list(labelencoder.classes_))

        predictions = classify(model, features_to_classify, labelencoder)

        # OUTPUT
        if not multi_categories:

            # single label classification
            if args.output_filename:
                print "Writing to output file: ", args.output_filename
                write_class_file(args.output_filename, ids, predictions)
            else:
                # just print to stdout
                for (i, label) in zip(ids,predictions):
                    print i + ":\t",label
        else:
            # multi label classification

            if args.output_filename:
                print "Writing to output file: ", args.output_filename

                if args.multiclasstable:
                    # write as table with x entries for positive categories
                    write_multi_class_table(args.output_filename, ids, predictions, class_columns=multi_categories)
                else:
                    # write as comma-separated list of positive classes
                    import pandas as pd
                    pred_df = pd.DataFrame(predictions, index=ids, columns=multi_categories)
                    ids, class_lists = multi_class_table_tolist(pred_df)
                    write_multi_class_list(args.output_filename, ids, class_lists)

            else:
                # just print to stdout
                import pandas as pd
                pred_df = pd.DataFrame(predictions, index=ids, columns=multi_categories)
                pred_df.replace(0, '', inplace=True)
                pred_df.replace(1, 'x', inplace=True)
                print pred_df