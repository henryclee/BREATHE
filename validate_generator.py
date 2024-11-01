# Imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from neuralnetworks import Generator, ICUDataset
import pickle
import time

with open('./data/generator_validation_features.pkl', 'rb') as f:
    featuredata = pickle.load(f)

with open('./data/generator_validation_labels.pkl', 'rb') as f:
    labeldata = pickle.load(f)

net = Generator(features = len(featuredata[0]), output_dim=162)
net.load_state_dict(torch.load('./models/generator.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        outputs = net(features)

        _, predicted = torch.max(outputs, 1)
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

end = time.time()

print(correct / total)

print(f'Finished Training in {end - start} seconds')
