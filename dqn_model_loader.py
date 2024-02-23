import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self, n_observ,n_output) :
        super(DQN,self).__init__()
        self.fc1=nn.Linear(n_observ,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,n_output)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=(self.fc3(x))
        return x


def select_action(env,policy_nn,state):
    action_q=policy_nn(torch.tensor(state,device=device))
    return action_q.max(-1).indices.item()

env = gym.make('LunarLander-v2' , render_mode='human')
policy_nn=torch.load('lunarlander-lowgamma.pt')

for epoch in range(10):

    state , info = env.reset()
    
    for x in count():
        action=select_action(env,policy_nn,state)
        new_state,reward,terminated,truncated,info=env.step(action)
        
        
        
        
        if terminated or truncated:
            
            break
        state=new_state