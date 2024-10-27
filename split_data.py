import os
import csv
from config import *
import pickle
import parseutils as p
from parseutils import sd
import time
import numpy as np
from collections import deque
from parameters import *
import sys

with open('./data/processed.pkl', 'rb') as f:
    processed = pickle.load(f)

print(len(processed))
print(processed[0])


#Split into training, validation, and test sets, we'll do 80/10/10
train_fraction = .8
validation_fraction = .1

np.random.shuffle(processed)
numtrain = int(len(processed) * train_fraction)
numvalidation = int(len(processed) * validation_fraction)

trainset = processed[:numtrain]
validationset = processed[numtrain : numtrain + numvalidation]
testset = processed[numtrain + numvalidation:]

with open('./data/trainset.pkl', 'wb') as f:
    pickle.dump(trainset, f)

with open('./data/validationset.pkl', 'wb') as f:
    pickle.dump(validationset,f)

with open('./data/testset.pkl', 'wb') as f:
    pickle.dump(testset,f)
