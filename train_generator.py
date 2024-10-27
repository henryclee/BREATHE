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
from neuralnetworks import Generator, ICUDataset
import pickle
import time

with open('./data/generator_validation_features.pkl', 'rb') as f:
    featuredata = pickle.load(f)

with open('./data/generator_validation_labels.pkl', 'rb') as f:
    labeldata = pickle.load(f)

print(len(featuredata))

dataset = ICUDataset(featuredata, labeldata, labelint = True)
dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)

net = Generator(features = len(featuredata[0]), output_dim=162)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr = 0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
net.to(device)

start = time.time()

for epoch in range(5):

    running_loss = 0.0
    for i, (features, labels) in enumerate(dataloader):

        features, labels = features.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(features)
        loss = criterion(outputs, labels)
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