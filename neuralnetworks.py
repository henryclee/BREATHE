# Imports
import numpy as np
# Neural Networks
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import random
import optuna
from optuna.trial import TrialState


# Dataset class
class ICUDataset(Dataset):
    def __init__(self, features, labels, labelint = False, squeeze = True):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long if labelint else torch.float32)
        if squeeze:
            self.labels = self.labels.squeeze()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
# Dataset class for DDQN
class DDQNDataset(Dataset):
    def __init__(self, replaybuffer):
        self.transitions = torch.tensor(replaybuffer, dtype=torch.float32)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]

 

# Train to generate batch constrained actions S -> A
class Generator(nn.Module):

    # Input will be the state as a one hot encoded vector
    def __init__(self, features : int, output_dim : int):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, input):
        input = input.float()
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)
        return output
    
    def save(self, fname):
        torch.save(self.state_dict(),fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))

# Train to predict next state S,A -> reward, S'
class Predictor(nn.Module):

    def __init__(self, features : int, label_dim : int):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(features, 256)
        self.fc2 = nn.Linear(256, 256)

        self.reward = nn.Linear(256, 1)
        self.nstate = nn.Linear(256, label_dim-1)
        
    def forward(self, input):
        input = input.float()
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))

        reward = self.reward(f2)
        nstate = self.nstate(f2)
        return reward, nstate
    
    def save(self, fname):
        torch.save(self.state_dict(),fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))


# DDQN to for batch learning
class DDQN(nn.Module):

    # Input will be the state as a one hot encoded vector
    def __init__(self, features : int, outputs : int):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, outputs)

    def forward(self, input):
        input = input.float()
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)
        return output
    
    def save(self, fname):
        torch.save(self.state_dict(),fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
    