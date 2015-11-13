
# 2015-11 by Thomas Lidy

import argparse
import cPickle
import numpy as np
from sklearn import preprocessing, svm

from rp_extract_batch import extract_all_files_in_path
from rp_feature_io import read_csv_features
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
def train_model(train_data, train_classes, print_accuracy = True):
    # TRAIN
    #model = svm.SVC() # defaults to kernel='rbf' (usually worse for RP/SSD/RH features)
    model = svm.SVC(kernel='linear')

    # with probabilities?
    #model = svm.SVC(probability=True, random_state=0)
    # cross_validation.cross_val_score(clf, X, y, scoring='log_loss')
    model.fit(Xtrain, Ytrain)

    if print_accuracy:
        pred_train = model.predict(train_data)
        n_correct = sum(np.array(pred_train) == np.array(train_classes))
        print "Accuracy on train set: %2.2f %%" % ( n_correct * 100.0 / len(train_classes) )

    return model

# CLASSIFY
def classify(model, features, labelencoder = None):
    pred = model.predict(features)
    if labelencoder:
        return(labelencoder.inverse_transform(pred))
    else:
        return(pred)

# CROSS VALIDATION (for experimentation)
def cross_validate(model, data, classes, folds=10):
    from sklearn import cross_validation
    return cross_validation.cross_val_score(model, Xfull, Yfull, scoring='accuracy', cv=folds)
    # scoring value: Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro',
    # 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error',
    # 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted',
    # 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

#acc = cross_validate(model, Xfull, Yfull)
#print "Avg Accuracy (%d folds): %2.2f (stddev: %2.2f)" % (len(acc), (np.mean(acc)*100), np.std(acc)*100)
#print acc
# ISMIR 2005: SSD: 72.70


# SAVE MODEL
def save_model(filename,model,scaler=None,labelencoder=None):
    with open(filename, 'wb') as f:
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    if scaler:
        filename = os.path.splitext(filename)[0] + ".scaler.pkl"
        with open(filename, 'wb') as f:
            cPickle.dump(scaler, f, protocol=cPickle.HIGHEST_PROTOCOL)
    if labelencoder:
        filename = os.path.splitext(filename)[0] + ".labelencoder.pkl"
        with open(filename, 'wb') as f:
            cPickle.dump(labelencoder, f, protocol=cPickle.HIGHEST_PROTOCOL)


# LOAD MODEL
def load_model(filename,scaler=True,labelencoder=True):
    f = open(filename, 'rb')
    model = cPickle.load(f)
    f.close()
    if scaler:
        filename = os.path.splitext(filename)[0] + ".scaler.pkl"
        f = open(filename, 'rb')
        scaler = cPickle.load(f)
        f.close()
    if labelencoder:
        filename = os.path.splitext(filename)[0] + ".labelencoder.pkl"
        f = open(filename, 'rb')
        labelencoder = cPickle.load(f)
        f.close()
    return (model,scaler,labelencoder)



if __name__ == '__main__':

    argparser = argparse.ArgumentParser() #formatter_class=argparse.ArgumentDefaultsHelpFormatter) # formatter_class adds the default values to print output

    argparser.add_argument('input_path', help='input file path to search for wav/mp3 files')
    argparser.add_argument('model_file', nargs='?', help='model file name (input for predictions, or to write after training)')
    argparser.add_argument('output_filename', nargs='?', help='filename for predictions to write (if omitted, will print output') # nargs='?' to make it optional

    args = argparser.parse_args()

    # default model file
    if args.model_file is None:
        args.model_file = 'models/model_GTZAN.pkl'


    # TODO future parameter for doing training
    do_training = False
    do_classification = True

    if do_training:
        # TODO alternatively load features
        ids, feat = extract_all_files_in_path(args.input_path)
        # TODO: store and load feature extraction parameters with model

        # check for label consistency TODO: do only for features extracted
        if all(ids['rh'] == ids['rp']) and all(ids['rh'] == ids['ssd']):
            ids2 = ids['rh']
        else:
            raise ValueError("ids not matching across feature files!")

        # TODO alternatively provide class file
        ids2 = ids['rh']
        classes = classes_from_filename(ids2)
        class_dict = dict(zip(ids2, classes))

        # convert to numeric classes
        (class_dict_num, labelencoder) = classdict_to_numeric(class_dict, return_encoder = True)

        class_num = get_classes_from_dict(class_dict_num,ids2)

        # optionally: concatenate rh + ssd
        features = np.hstack((feat['ssd'],feat['rh']))

        # standardize
        features, scaler = standardize(features)

        # TRAIN
        model = train_model(features, class_num)

        # save model
        save_model(args.model_file, model, scaler, labelencoder)


    if do_classification:
        # LOAD MODEL
        if not do_training:
            # TODO: store and load feature extraction parameters with model
            model, scaler, labelencoder = load_model(args.model_file)

        # EXTRACT FEATURES FROM NEW FILES
        ids, feat = extract_all_files_in_path(args.input_path)

        # SELECT OR CONCATENATE FEATURES
        features_to_classify = np.hstack((feat['ssd'],feat['rh']))

        # SCALE FEATURES LIKE TRAINING DATA
        features_to_classify = scaler.fit_transform(features_to_classify)

        # CLASSIFY
        predictions = classify(model, features_to_classify, labelencoder)

        # OUPUT
        for (i, label) in zip(ids,predictions):
            print i + ":\t",label

        # TODO write to file. use write class_dict
        print "output file: ", args.output_filename