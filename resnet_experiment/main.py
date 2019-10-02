#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:19:43 2018

@author: fernandodemeer
"""
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import OneHotEncoder

def fit_classifier(n_epochs,load,predict,epoch_to_load,val_loss_to_load,val_acc_to_load):

    x_train = x_train_data
    y_train = y_train_data

    x_test = x_test_data
    y_test = y_test_data

    nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))

    # save original y because later we will use binary
    y_true = y_test.astype(np.int64) 
    # transform the labels from integers to one hot vectors
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.concatenate((y_train,y_test),axis =0).reshape(-1,1))
    y_train = enc.transform(y_train.reshape(-1,1)).toarray()
    y_test = enc.transform(y_test.reshape(-1,1)).toarray()

    if len(x_train.shape) == 2: # if univariate 
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name,input_shape, nb_classes, output_directory,verbose = 2)
    if load:
        classifier.load(epoch_to_load,val_loss_to_load,val_acc_to_load)
    if predict:
        y_pred,y_class_pred = classifier.predict(x_test)
        classifier.plot_confusion_matrix(y_true,y_class_pred,[0,1,2])
    classifier.fit(x_train,y_train,x_test,y_test, y_true,n_epochs)

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose = 2):
     if classifier_name=='resnet':
         from resnet_experiment import resnet
         return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--load', type=bool, default=False) #Load a saved model or start training from scratch
    parser.add_argument('--predict', type=bool, default=False)  # Predict classes on an out-of-sample dataset (change x_test_data accordingly) and plot the confusion matrix
    parser.add_argument('--epoch_to_load', type=int, default=0) # If you have a saved model,change the default to the epochs of the trained model
    parser.add_argument('--epochs_to_train', type=int, default=2000) #Number of training iterations
    parser.add_argument('--val_loss_to_load', type=float, default=0) # If you have a saved model,change the default to your val_loss
    parser.add_argument('--val_acc_to_load', type=float, default=0) # If you have a saved model,change the default to your val_acc
    parser.add_argument('--improved_training', type=bool, default=False) # Enlarge the training set with synthetic series
    parser.add_argument('--data_type',type=str,default='VIX') #Check with sine data as a sanity check


    args = parser.parse_args()

    cwd = os.path.normpath(os.getcwd() + os.sep)

    if args.data_type == 'VIX':

        my_data = pd.read_csv('resnet_experiment/data/vixcurrent.csv', header=1, index_col=0)
        vix_data = my_data.loc[:, 'VIX Close']
        vix_data_arr = np.array((vix_data))


        X = np.array([vix_data_arr[30 * k:30 + 30 * k] for k in range(0, int((vix_data.size) / 30))])
        t = [vix_data.index[30 * k:30 + 30 * k] for k in range(0, int((vix_data.size) / 30))]

        ## We check now the class of each period, 0 if below 15, 1 if above
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0] - 1):
            hundreddaysprice = vix_data[30 * i + 30 + 20]
            if hundreddaysprice > 15:
                y[i, 0] = 1

        X = X[:-1, :]
        y = y[:-1, :]
        x_train_data = X[:99]
        x_test_data = X[99:]
        y_train_data = y[:99]
        y_test_data = y[99:]

        # We modify the test set, instead of the 28 non-overlapping periods of 30 days, we take rolling periods of 30 days until the last day we can tag (the 20th to last)
        my_data = pd.read_csv('resnet_experiment/data/vixcurrent.csv', header=1, index_col=0)
        vix_test_data = my_data.loc['10/20/2015':, 'VIX Close']
        vix_test_data_arr = np.array((vix_test_data))

        X_test = np.array([vix_test_data_arr[k:30 + k] for k in range(0, vix_test_data_arr.size-50)])
        t_test = [vix_test_data.index[30 * k:30 + 30 * k] for k in range(0, vix_test_data_arr.size-50)]
        y_test = np.zeros((X_test.shape[0], 1))
        for i in range(X_test.shape[0]):
            hundreddaysprice = vix_test_data[i + 30 + 20]
            if hundreddaysprice > 15:
                y_test[i, 0] = 1
        x_test_data = X_test
        y_test_data = y_test

    if args.improved_training and args.data_type == 'VIX':
        # Add new data to x_train, y_train
        data = np.load('resnet_experiment/data/Synthetic_VIX_series.npy')
        init_data = np.load('resnet_experiment/data/Synthetic_VIX_init.npy')

        synvix = np.ones(data.shape)
        synvix[:, 0] = init_data
        # We calculate prices from returns
        for i in range(0, data.shape[1] - 1):
            synvix[:, i + 1] = np.multiply(synvix[:, i], (1 + data[:, i]))

        for i in list(range(synvix.shape[0])):
            new_vix = synvix[i]

            X_new = np.array([new_vix[30 * k:30 + 30 * k] for k in range(0, int((new_vix.size) / 30))])


            ## We check now the class of each period
            y_new = np.zeros((X_new.shape[0], 1))
            for j in range(X_new.shape[0] - 1):
                hundreddaysprice = new_vix[30*j +30 + 20]
                if hundreddaysprice > 15:
                    y_new[j, 0] = 1
                # if hundreddaysprice < 12.5:
                #     y_new[j, 0] = 0
            X_new = X_new[:-1, :]
            y_new = y_new[:-1, :]
            x_train_data = np.concatenate((x_train_data, X_new), axis=0)
            y_train_data = np.concatenate((y_train_data, y_new), axis=0)


    root_dir = 'resnet_experiment/Saved models/'


    classifier_name='resnet'

    output_directory = root_dir


    print('Method: ',classifier_name)

    fit_classifier(args.epochs_to_train, args.load, args.predict,args.epoch_to_load,args.val_loss_to_load,args.val_acc_to_load)

    print('DONE')
