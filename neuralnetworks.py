# Imports
import numpy as np
# Neural Networks
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random
import optuna
from optuna.trial import TrialState

# Train to generate batch constrained actions S -> A
class Generator(nn.Module):

    # Input will be the state as a one hot encoded vector
    def __init__(self, features : int):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 162)

    def forward(self, input):
        input = input.float()
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        f3 = F.relu(self.fc3(f2))
        output = self.fc4(f3)
        return output
    
    def save(self, fname):
        torch.save(self.state_dict(),fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))

# Train to predict next state S,A -> S'
class Predictor(nn.Module):

    # Input will be the state as a one hot encoded vector
    def __init__(self, features : int):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 162)

    def forward(self, input):
        input = input.float()
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        f3 = F.relu(self.fc3(f2))
        output = self.fc4(f3)
        return output
    
    def save(self, fname):
        torch.save(self.state_dict(),fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))


# DDQN to for batch learning
class DDQN(nn.Module):

    # Input will be the state as a one hot encoded vector
    def __init__(self, features : int):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 162)

    def forward(self, input):
        input = input.float()
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        f3 = F.relu(self.fc3(f2))
        output = self.fc4(f3)
        return output
    
    def save(self, fname):
        torch.save(self.state_dict(),fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
    