from __future__ import division
import numpy as np # linear algebra
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import logging,sys

FORMAT = '%(asctime)-15s [%(levelname)-8s] %(message)s'
logging.basicConfig(stream=sys.stdout,format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%I')
console = logging.StreamHandler()
console.setLevel(logging.INFO)



#read data from dataset
logging.info("*** LOAD DATASET ***")
dataset = shuffle(np.array(pd.read_csv("dataset.csv",header=1)))

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

clf_regressor = DecisionTreeRegressor(max_depth = 50)
clf_regressor .fit(X_train,Y_train)

logging.info("*** TRAINING END ***")
predicted_regressor = clf_regressor .predict(X_test)

idx = 0
true = 0
false = 0
for i in X_test:
    #logging.info("*** Pred:"+str(predicted[idx])+" real: "+str(Y_test[idx])+" res "+str(predicted[idx]==Y_test[idx])+" ***")

    if predicted_regressor[idx]==Y_test[idx]:
        true +=1
    else:
        false +=1
    idx +=1

accuracy =  (true/(true+false))*100
logging.info("Positive Class: "+str(true))
logging.info("Negative Class: "+str(false))
logging.info("Accuracy Regressor: "+str(accuracy))