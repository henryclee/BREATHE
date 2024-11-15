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

# cols = ['gender', 'age', 'height', 'weight', 
#         'hr1', 'mbp1', 'spo21', 'gcs1', 'sofa1', 'invasive1','death1',
#         'hr2', 'mbp2', 'spo22', 'gcs2', 'sofa2', 'invasive2','death2',
#         'hr3', 'mbp3', 'spo23', 'gcs3', 'sofa3', 'invasive3','death3',
#         'hr4', 'mbp4', 'spo24', 'gcs4', 'sofa4', 'invasive4','death4',
#         'action', 'reward',
#         'hrn', 'mbpn', 'spo2n', 'gcsn', 'sofan', 'invasiven','deathn']

# statecols = ['gender', 'age', 'height', 'weight', 
#         'hr1', 'mbp1', 'spo21', 'gcs1', 'sofa1', 'invasive1','death1',
#         'hr2', 'mbp2', 'spo22', 'gcs2', 'sofa2', 'invasive2','death2',
#         'hr3', 'mbp3', 'spo23', 'gcs3', 'sofa3', 'invasive3','death3',
#         'hr4', 'mbp4', 'spo24', 'gcs4', 'sofa4', 'invasive4','death4']

stateidx = [cols.index(v) for v in statecols]

actioncols = ['action']
actionidx = [cols.index(v) for v in actioncols]

rewardcols = ['reward']
rewardidx = [cols.index(v) for v in rewardcols]

nstatecols = ['hrn', 'mbpn', 'spo2n', 'gcsn', 'sofan', 'invasiven','deathn']
nstateidx = [cols.index(v) for v in nstatecols]

def makeGeneratorData(dataset, featurename, labelname):
    #Generator - state -> action
    generator_features = dataset[:, stateidx]
    generator_labels = dataset[:, actionidx]
    with open(featurename, 'wb') as f:
        pickle.dump(generator_features, f)
    with open(labelname, 'wb') as f:
        pickle.dump(generator_labels, f)


def makePredictorData(dataset, featurename, labelname):
    #Predictor - state, action -> reward, next state
    predictor_features = dataset[:, stateidx + actionidx]
    predictor_labels = dataset[:, rewardidx + nstateidx]
    with open(featurename, 'wb') as f:
        pickle.dump(predictor_features, f)
    with open(labelname, 'wb') as f:
        pickle.dump(predictor_labels, f)

def makeReplayBuffer(dataset, transitionname):
    with open (transitionname, 'wb') as f:
        pickle.dump(dataset, f)
    return

path = './data/'
datasets = ['train', 'validation', 'test']

for datatype in datasets:
    with open(f'{path}{datatype}set.pkl', 'rb') as f:
        data = pickle.load(f)
    makeGeneratorData(data, f'{path}generator_{datatype}_features.pkl', f'{path}generator_{datatype}_labels.pkl')
    makePredictorData(data, f'{path}predictor_{datatype}_features.pkl', f'{path}predictor_{datatype}_labels.pkl')

makeReplayBuffer(data, f'{path}replaybuffer_train.pkl') # Only the training buffer will be used

print("finished")
