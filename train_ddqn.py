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
from neuralnetworks import DDQN, DDQNDataset
import pickle
import time

with open('./data/replaybuffer_train.pkl', 'rb') as f:
    replaybuffer = pickle.load(f)

dataset = DDQNDataset(replaybuffer)
dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)

predictNet = DDQN(5,5)
targetNet = DDQN(5,5)

criterion = nn.MSELoss()
optimizer = optim.Adam(predictNet.parameters(), lr = 0.001)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
predictNet.to(device)
targetNet.to(device)

rewardcols = [5] ########
statecols = [5]
nstatecols = [5]
actioncols = [5]
donecols = [5]

start = time.time()

discount_factor = .99

for epoch in range(5):

    running_loss = 0.0
    for i, transitions in enumerate(dataloader):

        transitions = transitions.to(device)

        td_rewards = transitions[:, [3]].clone()
        states = transitions[:,statecols]
        nstates = transitions[:, nstatecols]
        actions = transitions[:, [5]]
        dones = transitions[:, [6]] # 0 if done, 1 if not done

        predict_argmax = torch.argmax(predictNet(nstates), dim = 1)
        ddqn_maxQ = targetNet(nstates).gather(1, predict_argmax.unsqueeze(1))

        td_rewards = td_rewards + discount_factor * ddqn_maxQ * (dones)

        selected_outputs = predictNet(states).gather(1, actions)
        # for ib in range(len(transitions)):
        #     if dones[ib] == 1:
        #         # Have to do a forward pass on predict network on next state, get the argmax
        #         predict_argmax = torch.argmax(predictNet(nstate[ib].unsqueeze(0)))
        #         # Then do a forward pass on the target network, and the get the Q(s+1, a') value using the a from above
        #         ddqn_maxQ = targetNet(nstate[ib].unsqueeze(0))[predict_argmax]
        #         td_rewards[ib] += discount_factor * ddqn_maxQ.item() # This is the "label"

        # outputs = predictNet(states)
        # selected_outputs = outputs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        loss = criterion(selected_outputs, td_rewards)
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


### OLD STUFF


def DDQNtrain(
        netClass: nn.Module, 
        env: gym.Env,
        device = "cpu",
        replay_size = 5000, 
        batch_size = 128,
        n_episodes = 10_000,
        train_steps = 10,
        start_epsilon = 1,
        final_epsilon = 0.01,
        discount_factor = .99,
        learning_rate = .001,
        visualize = False,
        trial = None
        ):
    
    # Exponential decay
    epsilon_decay = (final_epsilon/start_epsilon) ** (1/n_episodes) 

    replayMem = deque(maxlen = replay_size) # Will hold (state, action, reward, state', done) tuples

    observation, info = env.reset()

    predictNet = netClass(len(observation))
    targetNet = netClass(len(observation))

    action_size = env.action_space.n

    # cuda?
    # predictNet.to(device)
    # targetNet.to(device)

    targetNet.load_state_dict(predictNet.state_dict())

    # First, fill the replay buffer. Moves will be entirely random
    state, info = env.reset()
    # state = torch.tensor(state, dtype = torch.float32)
    state = torch.from_numpy(norm(state))
    
    for _ in range(replay_size):

        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        # Custom reward
        reward = 100*((math.sin(3*next_state[0]) * 0.0025 + 0.5 * next_state[1] * next_state[1]) - (math.sin(3*state[0]) * 0.0025 + 0.5 * state[1] * state[1]))
        reward = reward.item()

        # next_state = torch.tensor(next_state, dtype = torch.float32)
        next_state = torch.from_numpy(norm(next_state))
        transition = (state, action, reward, next_state, terminated)
        replayMem.append(transition)
        if terminated or truncated:
            state, info = env.reset()
            # state = torch.tensor(state, dtype = torch.float32)
            state = torch.from_numpy(norm(state))
        else:
            state = next_state

    # DQN Algorithm
    epsilon = start_epsilon
    # Reset the environment
    state, info = env.reset()
    # state = torch.tensor(state, dtype = torch.float32)
    state = torch.from_numpy(norm(state))

    criterion = nn.MSELoss()
    # optimizer = optim.SGD(predictNet.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(predictNet.parameters())


    running_loss = 0.0
    cumulative_rewards = []
    single_run_reward = 0
    single_discount = 1
    epsilons = []
    
    for episode in range(n_episodes):

        # We want to constantly add to the replay buffer, which we'll do using an epsilon greedy strategy

        for _ in range (train_steps):
            if np.random.rand() < epsilon:
                # action = np.random.randint(action_size)
                action = env.action_space.sample()
            else:
                predicted_Q = predictNet.forward(state)
                action = torch.argmax(predicted_Q).item()
                        
            next_state, reward, terminated, truncated, info = env.step(action)

            # Custom reward
            reward = 100*((math.sin(3*next_state[0]) * 0.0025 + 0.5 * next_state[1] * next_state[1]) - (math.sin(3*state[0]) * 0.0025 + 0.5 * state[1] * state[1]))
            reward = reward.item()
            
            # Count single run reward
            single_run_reward += single_discount * reward
            # single_discount *= discount_factor
            
            # next_state = torch.tensor(next_state, dtype = torch.float32)
            next_state = torch.from_numpy(norm(next_state))
            transition = (state, action, reward, next_state, terminated)
            replayMem.append(transition) # This will also remove the first entry
            
            if terminated or truncated:
                state, info = env.reset()
                # state = torch.tensor(state, dtype = torch.float32)
                state = torch.from_numpy(norm(state))
                cumulative_rewards.append(single_run_reward)
                # if terminated:
                    # print("episode ended in success, reward: ", single_run_reward)
                single_run_reward = 0
                single_discount = 1
            else:
                state = next_state

        # Next, we want to get a sample of batches from the replay memory
        minibatch = random.sample(replayMem, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*minibatch) # as Lists
        
        # These are the sampled states
        batch_state = torch.stack(states)
        
        ground_truths = [] 

        for action, reward, nstate, done in zip(actions, rewards, next_states, dones):
            # scaled_reward = min(1, max(-1, reward/10))
            scaled_reward = reward
            td_reward = scaled_reward   # Temporal difference reward
            if not done:
                # Have to do a forward pass on predict network on next state, get the argmax
                predict_argmax = torch.argmax(predictNet.forward(nstate))
                # Then do a forward pass on the target network, and the get the Q(s+1, a') value using the a from above
                ddqn_maxQ = targetNet.forward(nstate)[predict_argmax]

                td_reward += discount_factor * ddqn_maxQ
            ground_truths.append(td_reward)

        # These are the target values for the Q(s,a) = imm reward + discount_factor * max over a of Q(s'a)
        batch_labels = torch.tensor(ground_truths, dtype=torch.float32)

        # convert actual actions taken into a tensor of shape [batch_size]
        actions = torch.tensor(actions, dtype=torch.int64)
        # outputs size will be [batch_size, action_space(3)], holding the predicted Q(s,a) values
        outputs = predictNet(batch_state)
        # Select from outputs only the Q(s',a) of the actions taken (shape [batch_size, 1]) then squeeze to ([batch_size])
        selected_outputs = outputs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Train the network

        # Zero the optimizer gradients
        optimizer.zero_grad()   
        loss = criterion(selected_outputs, batch_labels)
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(predictNet.parameters(), max_norm=1.0)
        optimizer.step()

        epsilons.append(epsilon)
      
        epsilon *= epsilon_decay

        # print statistics every x episodes
        running_loss += loss.item()
        if episode % 1000 == 999:    # print every 100 mini-batches
            print(f'[episode: {episode + 1:4d}] loss: {running_loss / 1000:.4f}')
            running_loss = 0.0
            if trial:
                trial.report(sum(cumulative_rewards[-1000:])/1000, episode+1)

        # Update targetNetwork
        if episode % 20 == 19:    
            targetNet.load_state_dict(predictNet.state_dict())

    print("finished training")


