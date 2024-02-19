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

import keyboard



Transition = namedtuple('transition',('state','action','reward','next_state'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):
    def __init__(self,len) -> None:
        self.memory=deque([],maxlen=len)
    
    def push(self,x):
        self.memory.append(Transition(*x))
    
    def sample(self,count):
        return random.sample(self.memory,count)
    
    def print(self):
        print(self.sample(1))

    def __len__(self):
        return len(self.memory)

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



# Hyperparams

REPLAY_BUFFER_SIZE=100000
EPS_START=0.9
EPS_DECAY=1000
EPS_END=0.05
LR=1e-4
BATCH_SIZE=128
GAMMA=0.99
TAU=0.005


env = gym.make('LunarLander-v2')
policy_nn=DQN((env.observation_space.shape[0]),env.action_space.n)
target_nn=DQN((env.observation_space.shape[0]),env.action_space.n)
replayBuffer=ReplayMemory(REPLAY_BUFFER_SIZE)
optimizer=torch.optim.AdamW(policy_nn.parameters(),lr=LR,amsgrad=True)
target_nn.load_state_dict(policy_nn.state_dict())


  



steps=0
def select_action(env,nn,state,has_eps=1.0):
    global steps
    
    eps=EPS_END +(EPS_START-EPS_END)*math.exp(-1*(steps/EPS_DECAY))
    eps*=has_eps
    steps+=1

    
    if torch.rand(1).item() <= eps:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            action_q=nn(torch.tensor(state,device=device))
            
            return action_q.max(-1).indices.item()

def optimize(replayBuffer,BATCH_SIZE):




    if(len(replayBuffer) < BATCH_SIZE):
        return 0
    else:
        minibatch=Transition( *zip(*replayBuffer.sample(BATCH_SIZE)))

        state=(torch.tensor(minibatch.state,device=device))
        action=(torch.tensor(minibatch.action,device=device))
        reward=(torch.tensor(minibatch.reward,device=device))
        next_state=(torch.tensor(minibatch.next_state,device=device))

        Q_s=policy_nn(state)
        Q_output = Q_s.gather(1,action.unsqueeze(1))

        with torch.no_grad():
            Q_target_s=target_nn(next_state)
            Q_target_op=Q_target_s.max(1).values

        
            

        expectation=reward + Q_target_op*GAMMA  
        


        criterion=nn.SmoothL1Loss()
        loss = criterion(Q_output,expectation.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(policy_nn.parameters(), 100)
        optimizer.step()
        return loss.item()

        



rewardlist=[]
for epoch in range(700):

    state , info = env.reset()
    
    
    for x in count():
        action=select_action(env,policy_nn,state)
        new_state,reward,terminated,truncated,info=env.step(action)
        
        if terminated or truncated:
            rewardlist.append(x)
            
            optimize(replayBuffer,BATCH_SIZE)

            target_nn_state_dict=target_nn.state_dict()
            policy_nn_state_dict=policy_nn.state_dict()

            for k in policy_nn_state_dict:
                target_nn_state_dict[k]=policy_nn_state_dict[k]*TAU + (1-TAU)*target_nn_state_dict[k]
            
            target_nn.load_state_dict(target_nn_state_dict)
            
            break
        
        
        replayBuffer.push((state,action,reward,new_state))
        
        state=new_state
        optimize(replayBuffer,BATCH_SIZE)

        target_nn_state_dict=target_nn.state_dict()
        policy_nn_state_dict=policy_nn.state_dict()

        for k in policy_nn_state_dict:
            target_nn_state_dict[k]=policy_nn_state_dict[k]*TAU + (1-TAU)*target_nn_state_dict[k]
        
        target_nn.load_state_dict(target_nn_state_dict)
        
        if keyboard.is_pressed('space'):
            break
        
    if keyboard.is_pressed('space'):
            break
    
    plt.plot(range(len(rewardlist)),rewardlist,color='r')  
    plt.draw()
    plt.pause(0.01)
        

    

torch.save(policy_nn,'lunarlander.pt')
env = gym.make('LunarLander-v2' , render_mode='human')

for epoch in range(10):

    state , info = env.reset()
    
    for x in count():
        action=select_action(env,policy_nn,state,has_eps=0.0)
        new_state,reward,terminated,truncated,info=env.step(action)
        
        if terminated or truncated:
            break
        state=new_state

        














        

        

        


        
        
            

            
            
    
    
        
        

    

plt.show()

    
    
    
