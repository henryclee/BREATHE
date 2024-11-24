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
from neuralnetworks import DDQN, DDQNDataset, Generator
import pickle
import time
from parameters import *


# Train batch constrained DDQN on replay buffer data
# c is how often we update targetnet, and k is the batch constraint parameter
def trainDDQN(epochs = 10, c = 10, k = 4):

    with open('./data/replaybuffer_train.pkl', 'rb') as f:
        replaybuffer = pickle.load(f)

    dataset = DDQNDataset(replaybuffer)
    dataloader = DataLoader(dataset, batch_size = 512, shuffle = True)

    predictNet = DDQN(len(STATEIDX), ACTIONSPACE)
    targetNet = DDQN(len(STATEIDX), ACTIONSPACE)
    generator = Generator(len(STATEIDX), ACTIONSPACE)

    # Generator takes state (sstatic + 4 dstate), and generates plausible actions
    generator.load_state_dict(torch.load('./models/generator.pth'))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(predictNet.parameters(), lr = 0.001)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    predictNet.to(device)
    targetNet.to(device)

    # How often do we update targetnet
    # c = 10
    # k is how limiting the batch constraint clipping will be
    # k = 4

    # Generator

    start = time.time()

    discount_factor = .99
    losses = []

    for epoch in range(epochs):

        running_loss = 0.0
        for i, transitions in enumerate(dataloader):

            transitions = transitions.to(device)

            td_rewards = transitions[:, REWARDIDX].clone() # Because we are changing this value
            states = transitions[:,STATEIDX]
            nstates = transitions[:, SSTATEIDX + DSTATEIDX2 + DSTATEIDX3 + DSTATEIDX4 + NSTATEIDX]
            actions = transitions[:, ACTIONIDX].to(torch.int64)
            dones = transitions[:, DONEIDX] # 0 if done, 1 if not done

            # Batch constraint
            batch_constraints = generator(nstates)
            # Get the top k action indices from batch_constraints
            _, topkidx = torch.topk(batch_constraints, k, dim=1)
            # Get the q-values from predictNet, constrained to topk actions
            predict_q_values = predictNet(nstates).gather(1, topkidx)
            # Get the topkidx of the argmax from predictnet
            predict_topk_argmax = torch.argmax(predict_q_values, dim = 1)
            # Get the batch constrained actions from action space
            bc_predict_argmax = topkidx.gather(1, predict_topk_argmax.unsqueeze(1))     
            # Use the Q-value from targetNet from the argmax of the batch constrained actions
            ddqn_maxQ = targetNet(nstates).gather(1, bc_predict_argmax)

            td_rewards = td_rewards + discount_factor * ddqn_maxQ * (dones)

            selected_outputs = predictNet(states).gather(1, actions)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            loss = criterion(selected_outputs, td_rewards)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictNet.parameters(), max_norm=.5)
            optimizer.step()

            # Update targetNet every c steps
            if i % c == c-1:
                targetNet.load_state_dict(predictNet.state_dict())

            # print statistics
            running_loss += loss.item()

        # print(f'[{epoch + 1}] loss: {running_loss / 1000:.3f}')
        losses.append(running_loss)
        running_loss = 0.0

    end = time.time()
    print(f'Finished Training in {end - start} seconds')

    return predictNet, losses






# trainDDQN()