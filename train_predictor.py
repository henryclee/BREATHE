# Imports
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import random
import optuna
from optuna.trial import TrialState
from neuralnetworks import Predictor, ICUDataset
import pickle
import time

with open('./data/predictor_validation_features.pkl', 'rb') as f:
    featuredata = pickle.load(f)

with open('./data/predictor_validation_labels.pkl', 'rb') as f:
    labeldata = pickle.load(f)

print(len(featuredata))
print(labeldata[0])

dataset = ICUDataset(featuredata, labeldata, labelint = False)
dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)

net = Predictor(features = len(featuredata[0]), label_dim = len(labeldata[0]))
criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr = 0.001)
optimizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.99, eps=1e-08)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
net.to(device)

start = time.time()

for epoch in range(5):

    running_loss = 0.0
    for i, (features, labels) in enumerate(dataloader):

        features, labels = features.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        rewards, nextstates = net(features)

        rewardloss = criterion(rewards, labels[:,[0]])
        nstateloss = criterion(nextstates, labels[:,1:])

        loss = rewardloss + nstateloss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 1000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

end = time.time()

print(f'Finished Training in {end - start} seconds')
