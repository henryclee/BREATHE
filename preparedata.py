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

with open ('./data_small.pkl', 'rb') as pkl:
    dataset = pickle.load(pkl)

#Static State
# static_state = ['gender', 'age', 'height', 'weight', 'race']
#Dynamic State
# dynamic_state = ['heart_rate', 'mbp', 'spo2', 'gcs', 'sofa', 'invasive','death']

# Index into states
s = {}
d = {}

for i,v in enumerate(['gender', 'age', 'height', 'weight', 'race']):
    s[v] = i

for i,v in enumerate(['hr', 'mbp', 'spo2', 'gcs', 'sofa', 'invasive','death']):
    d[v] = i


def calcReward(dstate):
    reward = 0
    if MINMBP <= dstate[d['mbp']] <= MAXMBP:
        reward += GOODBPREWARD
    else:
        reward += BADBPREWARD
    if MINSPO2 <= dstate[d['spo2']] <= MAXSPO2:
        reward += GOODO2REWARD
    else:
        reward += BADO2REWARD
    if dstate[d['invasive']] == 0:
        if dstate[d['death']] == 0:
            reward += SURVIVE
        else:
            reward += DEATH
    return reward

def generateState(sstate, dstateq):
    state = []
    state += sstate
    for dstate in dstateq:
        state += dstate
    return state

