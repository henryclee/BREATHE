# Imports
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
import gymnasium as gym
import random
import math
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
from neuralnetworks import *
import pickle
import time
from parameters import *


boundaries = torch.tensor([
    [float('-inf'), 1000.0, 2000.0, float('inf')], # Heart rate is not factored
    [float('-inf'), MINMBP -.1, MAXMBP, float('inf') ], # MBP buckets
    [float('-inf'), MINSPO2 -.1, MAXSPO2, float('inf') ], # SPO2
    # [0.0, 1000.0, 2000.0, 3000.0], # not considering vaso
    [float('-inf'), 1000.0, 2000.0, float('inf')], # not considering gcs
    [float('-inf'), 1000.0, 2000.0, float('inf')], # not considering sofa
    [float('-inf'), .5, 1000.0, float('inf')], # invasive
    [float('-inf'), .5, 1000.0, float('inf')] # death
], dtype = torch.float32)

# discretize dstate
def bucketize(dstate):
    batch_size, num_features = dstate.shape
    dstate_buckets = torch.zeros_like(dstate, dtype=torch.int8)
    for i in range(num_features):
        # Bucketize each value with its respective boundaries
        dstate_buckets[:, i] = torch.bucketize(dstate[:, i], boundaries[i], right=False) - 1
    return dstate_buckets

# Uses a trained predictNet ddqn and generator to output a batch constrained action, given an input state  
def validateDDQN(k = 4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    with open('./data/predictor_validation_features.pkl', 'rb') as f:
        featuredata = pickle.load(f)

    with open('./data/predictor_validation_labels.pkl', 'rb') as f:
        labeldata = pickle.load(f)

    ddqn = DDQN(len(STATEIDX), ACTIONSPACE)
    generator = Generator(len(STATEIDX), ACTIONSPACE)
    predictor = Predictor(features = len(featuredata[0]), label_dim = len(labeldata[0]))
    
    # ddqn takes a state (sstate + 4 dstate), and generates an action
    ddqn.load_state_dict(torch.load('./models/ddqn.pth'))
    # generator takes state (sstate + 4 dstate), and generates plausible actions
    generator.load_state_dict(torch.load('./models/generator.pth'))
    # predictor takes a state (state + 4 dstate), and predicts the next dstate
    predictor.load_state_dict(torch.load('./models/predictor.pth'))

    ddqn.to(device)
    generator.to(device)
    predictor.to(device)
    
    ddqn.eval()
    generator.eval()
    predictor.eval()

    dataset = ICUDataset(featuredata, labeldata, labelint = True)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)

    start = time.time()

    # We'll store a list of low(0), optimal(1), high(2) mbps and o2s
    true_MBP = []
    predicted_MBP = []
    true_SPO2 = []
    predicted_SPO2 = []
    # 0 survive, 1 death
    true_Invasive = []
    true_Outcome = []
    predicted_Invasive = []
    predicted_Outcome = []

    for i, (features, labels) in enumerate(dataloader):

        # features are sstate + 4dstates + action
        # labels are reward + nstate (single dstate)
        features, labels = features.to(device), labels.to(device)
        
        # Gather true data - skip reward
        label_buckets = bucketize(labels[:, 1:])
        true_MBP.extend(label_buckets[:, 1].tolist())
        true_SPO2.extend(label_buckets[:, 2].tolist())
        true_Invasive.extend(label_buckets[:, 5].tolist())
        true_Outcome.extend(label_buckets[:, 6].tolist())
        
        # Gather predicted data
        # Batch constraint
        batch_constraints = generator(features[:, :-1])
        # Get the top k action indices from batch_constraints
        _, topkidx = torch.topk(batch_constraints, k, dim=1)
        # Get the q-values from predictNet, constrained to topk actions
        ddqn_q_values = ddqn(features[:, :-1]).gather(1, topkidx)
        # Get the topkidx of the argmax from predictnet
        ddqn_topk_argmax = torch.argmax(ddqn_q_values, dim = 1)
        # Get the batch constrained actions from action space
        ddqn_actions = topkidx.gather(1, ddqn_topk_argmax.unsqueeze(1))

        # add actions to features for predictor
        ddqn_features = torch.cat((features[:, :-1], ddqn_actions), dim = 1)
        
        predicted_actions, predicted_states = predictor(ddqn_features)

        predicted_buckets = bucketize(predicted_states)
        predicted_MBP.extend(predicted_buckets[:, 1].tolist())
        predicted_SPO2.extend(predicted_buckets[:, 2].tolist())
        predicted_Invasive.extend(predicted_buckets[:, 5].tolist())
        predicted_Outcome.extend(predicted_buckets[:, 6].tolist())

    true_MBP_buckets = [0,0,0]
    true_SPO2_buckets = [0,0,0]

    pred_MBP_buckets = [0,0,0]
    pred_SPO2_buckets = [0,0,0]

    for tbp, tsp, pbp, psp in zip(true_MBP, true_SPO2, predicted_MBP, predicted_SPO2):

        if pbp > 1000:
            print(pbp)


        true_MBP_buckets[tbp] += 1
        true_SPO2_buckets[tsp] += 1
        pred_MBP_buckets[pbp] += 1
        pred_SPO2_buckets[psp] += 1

    print("true mbp buckets", true_MBP_buckets)
    print("pred mbp buckets", pred_MBP_buckets)
        
    print("true o2 buckets", true_SPO2_buckets)
    print("pred o2 buckets", pred_SPO2_buckets)

    true_Outcome_buckets = [0,0]
    pred_Outcome_buckets = [0,0]

    for tinv, td, pinv, pd in zip(true_Invasive, true_Outcome, predicted_Invasive, predicted_Outcome):
        if tinv == 0:
            true_Outcome_buckets[td] += 1
        if pinv == 0:
            pred_Outcome_buckets[pd] += 1

    print ("true outcome", true_Outcome_buckets)
    print ("predicted outcome", pred_Outcome_buckets)
    


    
    









    end = time.time()
    print(f"Finished in {end - start} seconds")








validateDDQN()