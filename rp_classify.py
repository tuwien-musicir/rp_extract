'''
Classification of Music into Genres, Moods or other categories
using the Rhythm Pattern audio analyzer (rp_extract.py)

2015-11 by Thomas Lidy
'''

import argparse
import cPickle
import numpy as np

from sklearn import preprocessing, svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from rp_feature_io import load_or_analyze_features
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
        return(labelencoder.inverse_transform(pred))
    else:
        return(pred)

# CROSS VALIDATION (for experimentation)
def cross_validate(model, features, classes, folds=10):
    from sklearn import cross_validation
    return cross_validation.cross_val_score(model, features, classes, scoring='accuracy', cv=folds)
    # scoring value: Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro',
    # 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error',
    # 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted',
    # 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']


# SAVE MODEL
def save_model(filename,model,scaler=None,labelencoder=None):
    basename = os.path.splitext(filename)[0]
    with open(basename + ".model.pkl", 'wb') as f:
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    if scaler:
        filename = os.path.splitext(filename)[0] + ".scaler.pkl"
        with open(basename + ".scaler.pkl", 'wb') as f:
            cPickle.dump(scaler, f, protocol=cPickle.HIGHEST_PROTOCOL)
    if labelencoder:
        filename = os.path.splitext(filename)[0] + ".labelencoder.pkl"
        with open(basename + ".labelenc.pkl", 'wb') as f:
            cPickle.dump(labelencoder, f, protocol=cPickle.HIGHEST_PROTOCOL)


# LOAD MODEL
def load_model(filename,scaler=True,labelencoder=True):
    basename = os.path.splitext(filename)[0]
    f = open(basename + ".model.pkl", 'rb')
    model = cPickle.load(f)
    f.close()
    if scaler:
        f = open(basename + ".scaler.pkl", 'rb')
        scaler = cPickle.load(f)
        f.close()
    if labelencoder:
        f = open(basename + ".labelenc.pkl", 'rb')
        labelencoder = cPickle.load(f)
        f.close()
    return (model,scaler,labelencoder)



if __name__ == '__main__':

    argparser = argparse.ArgumentParser() #formatter_class=argparse.ArgumentDefaultsHelpFormatter) # formatter_class adds the default values to print output

    argparser.add_argument('input_path', help='input file path to search for wav/mp3 files')
    argparser.add_argument('model_file', nargs='?', help='model file name (input for predictions, or to write after training)')
    argparser.add_argument('output_filename', nargs='?', help='filename for predictions to write (if omitted, will print output') # nargs='?' to make it optional

    argparser.add_argument('-t','--train',action='store_true',help='train a model with the input data',default=False) # boolean opt
    argparser.add_argument('-cv','--crossval',action='store_true',help='cross-validate with the input data',default=False) # boolean opt

    args = argparser.parse_args()


    if args.train or args.crossval:

        if args.train and not args.crossval and args.model_file is None:
            raise ValueError("model_file must be provided to store when training a model.")

        # LOAD OR EXTRACT FEATURES
        # TODO: store and load feature extraction parameters with model
        ids, feat = load_or_analyze_features(args.input_path)

        # CLASSES: derive from sub-path
        # TODO alternatively provide class file
        classes = classes_from_filename(ids)
        class_dict = dict(zip(ids, classes))

        # convert to numeric classes
        (class_dict_num, labelencoder) = classdict_to_numeric(class_dict, return_encoder = True)

        class_num = get_classes_from_dict(class_dict_num,ids)

        # CONCATENATE MULTIPLE FEATURES
        # (optional) here: concatenate ssd + rh
        # TODO don't hardcode this
        features = np.hstack((feat['ssd'],feat['rh']))

        # STANDARDIZE
        features, scaler = standardize(features)

        # TRAIN + SAVE MODEL
        if args.train:

            model = train_model(features, class_num)

            # save model
            save_model(args.model_file, model, scaler, labelencoder)

        # CROSS-VALIDATE
        if args.crossval:

            if not args.train:
                # if we trained, we have a model already; otherwise we initialize a fresh one
                model = svm.SVC(kernel='linear')

            acc = cross_validate(model, features, classes, folds=10)
            print "Fold Accuracy:", acc
            print "Avg Accuracy (%d folds): %2.2f (stddev: %2.2f)" % (len(acc), (np.mean(acc)*100), np.std(acc)*100)

    else: # do classification only when not training

        # LOAD MODEL
        if args.model_file is None:
            args.model_file = 'models/GTZAN'   # default model file

        if not args.train:
            # TODO: store and load feature extraction parameters with model
            model, scaler, labelencoder = load_model(args.model_file)

        # EXTRACT FEATURES FROM NEW FILES
        ids, feat = load_or_analyze_features(args.input_path)

        if len(feat) == 0:
            raise ValueError("No features were extracted from input files. Check format.")

        # SELECT OR CONCATENATE FEATURES
        features_to_classify = np.hstack((feat['ssd'],feat['rh']))

        # SCALE FEATURES LIKE TRAINING DATA
        features_to_classify = scaler.transform(features_to_classify)

        # CLASSIFY
        predictions = classify(model, features_to_classify, labelencoder)

        # OUPUT
        if args.output_filename:
            print "Writing to output file: ", args.output_filename
            write_class_file(args.output_filename, ids, predictions)
        else:
            # just print to stdout
            for (i, label) in zip(ids,predictions):
                print i + ":\t",label
