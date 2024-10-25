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

# Calculate reward from state
def getReward(dstate):
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


# concatenate static state and window_size states of dynamic state queue
def generateState(sstate, dstateq):
    state = []
    state += sstate
    for dstate in dstateq:
        state += dstate
    return state

# Are all the states in the window identical to each other, and next state?
def identical(dstateq, nstate):
    for i in range(WINDOW_SIZE):
        if dstateq[i] != nstate:
            return False
    return True

# Get action, as 1 of 162 possible states
def getAction(actionlist):
    action = 0
    peep, rr, tv, fio2, vaso = actionlist
    # Encode peep
    if peep > HIPEEP:
        action =  2
    elif peep >= LOPEEP:
        action = 1
    action *= 3
    
    # Encode rr
    if rr > HIRR:
        action += 2
    elif rr >= LORR:
        action += 1
    action *= 3

    # Encode tv
    if tv > HITV:
        action += 2
    elif tv >= LOTV:
        action += 1
    action *= 3

    # Encode fio2
    if fio2 > HIO2:
        action += 2
    elif fio2 >= LOO2:
        action += 1
    action *= 2

    # Encode vaso (0 or 1)
    action += vaso
    
    return action

# Processed state will be currentstate [static + 4 dynamic_states] = 4 + 7*4, action = 1, next_state 7
def process_data(fp):

    with open (fp, 'rb') as pkl:
        dataset = pickle.load(pkl)

    processed = []

    for patient_ep in dataset:

        if len(patient_ep) < 12:
            continue

        dstateq = deque(maxlen = WINDOW_SIZE) # Dynamic state queue
        reward = float('-inf')
        discount = DF

        for step in patient_ep:
            reward = max(reward, float('-inf'))
            sstate, dstate, action, nstate = step
            dstateq.append(dstate)

            if len(dstateq) < WINDOW_SIZE:
                continue
            if identical(dstateq, nstate):
                reward = getReward(nstate) * discount if reward ==float('-inf') else reward + getReward(nstate) * discount
                discount *= DF
                continue

            reward = getReward(nstate) if reward == float('-inf') else reward + getReward(nstate)
            actionval = getAction(action)

            state = generateState(sstate[:-1], dstateq)
            processed.append(state + [actionval] + [reward] + nstate)

            reward = float('-inf')
            discount = DF

    processed = np.array(processed)

    with open('./processed.pkl', 'wb') as pf:
        pickle.dump(processed, pf)    


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Need filepath")
        sys.exit(1)

    fp = sys.argv[1]

    process_data(fp)