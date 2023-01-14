import numpy as np
import gym
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import time
from IPython import display
import torch
import torch.nn as nn
from operator import itemgetter
import copy
%matplotlib inline

class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        reward = 2.5
        reward -= 10 * abs(next_state[2])
        if(done): reward -= 15
        return next_state, reward, done, info

env = BasicWrapper(gym.make('CartPole-v1'))
env.reset()

numActions = env.action_space.n

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.Sigmoid(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64,2),
            nn.Softmax()
        )
        
    def forward(self, x):
        return self.layers(x)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.Sigmoid(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64,1)
        )
        
    def forward(self, x):
        return self.layers(x)

#HYPERPARAMETERS
gamma = 0.995
LR = 1e-3
eps = 2000
maxEpSteps = 750

#OPTIMIZERS
actor = Actor()
actorOptim = torch.optim.Adam(actor.parameters(), lr=LR)
critic = Critic()
criticOptim = torch.optim.Adam(critic.parameters(), lr=LR)

def toTensor(array) : return torch.from_numpy(array).float()

#TRAINING
countList = []
for i in tqdm(range(eps)) :
    
    state = env.reset()
    count = 0
    
    for j in range(maxEpSteps) :
        
        probs = actor(toTensor(state))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
            
        newState, reward, done, info = env.step(action.detach().item())
        
        adv = reward + gamma * critic(toTensor(newState)) - critic(toTensor(state))
        
        actorLoss = -dist.log_prob(action) * adv.detach()
        
        actorOptim.zero_grad()
        actorLoss.backward()
        actorOptim.step()
        
        criticLoss = adv.pow(2)
        
        criticOptim.zero_grad()
        criticLoss.backward()
        criticOptim.step()
        
        state = newState
        
        count += 1
        
        if(done) : break
            
    countList.append(count)
        
    if i%(25) == 0:
        plt.plot(countList, color="blue")
        display.clear_output(wait=True)
        display.display(plt.gcf())
