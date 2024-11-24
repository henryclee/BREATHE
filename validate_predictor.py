# Imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from neuralnetworks import Predictor, ICUDataset
import pickle
import time
from parameters import *

# Predictor takes a state, action pair (static + 4 dstates), and outputs a next state (1 dstate) 
# we will validate after discretizing the next state, based on the reward structure

# 'hrn', 'mbpn', 'spo2n', 'gcsn', 'sofan', 'invasiven','deathn'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

boundaries = torch.tensor([
    [0.0, 1000.0, 2000.0, 3000.0], # Heart rate is not factored
    [0.0, MINMBP, MAXMBP, 1000.0 ], # MBP buckets
    [0.0, MINSPO2, MAXSPO2, 1000.0 ], # SPO2
    # [0.0, 1000.0, 2000.0, 3000.0], # not considering vaso
    [0.0, 1000.0, 2000.0, 3000.0], # not considering gcs
    [0.0, 1000.0, 2000.0, 3000.0], # not considering sofa
    [0.0, .5, 1000.0, 2000.0], # invasive
    [0.0, .5, 1000.0, 2000.0] # death
], dtype = torch.float32).to(device)

# discretize dstate
def bucketize(dstate):
    batch_size, num_features = dstate.shape
    dstate_buckets = torch.zeros_like(dstate, dtype=torch.float32)
    for i in range(num_features):
        # Bucketize each value with its respective boundaries
        dstate_buckets[:, i] = torch.bucketize(dstate[:, i], boundaries[i], right=False) - 1
    return dstate_buckets


with open('./data/predictor_validation_features.pkl', 'rb') as f:
    featuredata = pickle.load(f)

with open('./data/predictor_validation_labels.pkl', 'rb') as f:
    labeldata = pickle.load(f)

net = Predictor(features = len(featuredata[0]), label_dim = len(labeldata[0]))

net.load_state_dict(torch.load('./models/predictor.pth'))

net.to(device)
net.eval()

dataset = ICUDataset(featuredata, labeldata, labelint = True)
dataloader = DataLoader(dataset, batch_size = 4096, shuffle = True)

start = time.time()

correct = 0
total = 0
with torch.no_grad():
    for i, (features, labels) in enumerate(dataloader):

        features, labels = features.to(device), labels.to(device)

        label_buckets = bucketize(labels[:,1:])
        rewards, next_states = net(features)
        output_buckets = bucketize(next_states)

        correct += (output_buckets == label_buckets).sum().item()

        total += labels.size(0) * 7


end = time.time()
print(correct, total)
print(correct / total)

print(f'Finished Validation in {end - start} seconds')
