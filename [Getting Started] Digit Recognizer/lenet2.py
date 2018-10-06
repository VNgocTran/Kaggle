import numpy as np
import tensorflow as tf
import pickle

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import flatten


# read data
# old data
# open existed training set
with open("./data/X_train.p", "rb") as f:
    Xtrain = pickle.load(f)

# open existed training set
with open("./data/y_train.p", "rb") as f:
    ytrain = pickle.load(f)

X_train = Xtrain
y_train = ytrain

X_train = np.reshape(X_train, (-1, 28,28,1))

# new data
# open existed training set
with open("./data/X_train_new.p", "rb") as f:
    X_train_new = pickle.load(f)

# open existed training set
with open("./data/y_train_new.p", "rb") as f:
    y_train_new = pickle.load(f)




	
	
	
	
	
	
	
	
	
	
	
	





