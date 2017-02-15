from __future__ import division
import warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import logging,sys

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


FORMAT = '%(asctime)-15s [%(levelname)-8s] %(message)s'
logging.basicConfig(stream=sys.stdout,format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%I')
console = logging.StreamHandler()
console.setLevel(logging.INFO)


#data frame
logging.info("*** CLEANING DATAFRAME ***")
data_frame = pd.read_csv("dataset.csv",header=1)
data_frame.drop(data_frame.columns[[0]], axis=1, inplace=True)
dataset = shuffle(np.array(data_frame))

extracted_dataset= []
target = []

#extract target column
for row in dataset:
    extracted_dataset.append(row[1:])
    if row[0] == 'B':
        target.append(0)
    else:
        target.append(1)



X_train, X_test, Y_train, Y_test= train_test_split(extracted_dataset,target,test_size=0.3)
logging.info("*** DATASET PARTITIONED IN TRAIN: "+str(len(X_train))+ " TEST: "+str(len(X_test)))

Y_train = np.array(Y_train).astype(np.uint8)
Y_test = np.array(Y_test).astype(np.uint8)
X_train = np.array(X_train).reshape((-1, 1, 6, 5)).astype(np.uint8)
X_test = np.array(X_test).reshape((-1, 1, 6, 5)).astype(np.uint8)



"""
epoch: one forward pass and one backward pass of all the training examples --> 15

"""



net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
    # layer parameters:
    input_shape=(None, 1, 6, 5),
    hidden_num_units=1000,  # number of units in 'hidden' layer
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=2,  # target values for the digits 0, 1

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.0001,
    update_momentum=0.9,

    max_epochs=30,
    verbose=0,
)

net1.fit(X_train, Y_train)



def CNN(n_epochs):
    net1 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),  # Convolutional layer.  Params defined below
            ('pool1', layers.MaxPool2DLayer),  # Like downsampling, for execution speed
            ('conv2', layers.Conv2DLayer),
            ('hidden3', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],

        input_shape=(None, 1, 6, 5),
        conv1_num_filters=8,
        conv1_filter_size=(3, 3),
        conv1_nonlinearity=lasagne.nonlinearities.rectify,

        pool1_pool_size=(2, 2),

        conv2_num_filters=12,
        conv2_filter_size=(1, 1),
        conv2_nonlinearity=lasagne.nonlinearities.rectify,

        hidden3_num_units=1000,
        output_num_units=2,
        output_nonlinearity=lasagne.nonlinearities.softmax,

        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=n_epochs,
        verbose=0,
    )
    return net1

cnn = CNN(40).fit(X_train, Y_train)  # train the CNN model for 15 epochs
logging.info("*** TRAINING END ***")


predicted = cnn.predict(X_test)

idx = 0
true = 0
false = 0
for i in X_test:
    #logging.info("*** Pred:"+str(predicted[idx])+" real: "+str(Y_test[idx])+" res "+str(predicted[idx]==Y_test[idx])+" ***")

    if predicted[idx]==Y_test[idx]:
        true +=1
    else:
        false +=1
    idx +=1

accuracy =  (true/(true+false))*100
logging.info("Positive Class: "+str(true))
logging.info("Negative Class: "+str(false))
logging.info("Accuracy: "+str(accuracy))