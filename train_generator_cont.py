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
from parameters import *


with open('./data/generator_train_features.pkl', 'rb') as f:
    featuredata = pickle.load(f)

with open('./data/generator_train_labels.pkl', 'rb') as f:
    labeldata = pickle.load(f)

print(len(featuredata))

dataset = ICUDataset(featuredata, labeldata, labelint = True)
dataloader = DataLoader(dataset, batch_size = 4096, shuffle = True)

net = Generator(features = len(featuredata[0]), output_dim=ACTIONSPACE)
net.load_state_dict(torch.load('./models/generator.pth'))
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr = 0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
net.to(device)

start = time.time()

for epoch in range(10_000):

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

    if epoch % 100 == 99:
        print(f'[{epoch + 1}] loss: {running_loss / 100:.3f}')
        running_loss = 0.0
        torch.save(net.state_dict(), './models/generator.pth')

end = time.time()

print(f'Finished Training in {end - start} seconds')

torch.save(net.state_dict(), './models/generator.pth')