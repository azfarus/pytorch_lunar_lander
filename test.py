import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy


class DQNAgentGPT:
    def __init__(self, state_size, action_size,model):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.5  # Discount factor
        self.epsilon = 1  # Exploration rate
        self.epsilon_decay = 0.999999
        self.epsilon_min = 0.01
        self.model = model

    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        
        endm=len(self.memory)
        startm=endm-batch_size

        minibatch = random.sample(([self.memory[i] for i in range(startm , endm)]), batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict_next_action(self, state):
        
        state = np.reshape(state, [1, self.state_size])
        action_values = self.model.predict(state)
        next_action = np.argmax(action_values[0])
        return next_action



def build_agent(model,states,actions):
    
    # memory=SequentialMemory(limit=20000,window_length=1)
    # policy=BoltzmannQPolicy()
    # agent = DQNAgent(model=model,memory=memory,policy=policy,nb_actions=actions , nb_steps_warmup=10 , target_model_update=1e-2)

    agent = DQNAgentGPT(state_size=states , action_size=actions , model=model)
    return agent
    


def build_model(actions , states):
    model=Sequential()
    model.add(Flatten(input_shape=(1 , states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions , activation='linear'))
    return model


env=gym.make('CartPole-v1' )
states = env.observation_space.shape[0]
actions=env.action_space.n

q_model=build_model(states=states , actions=actions)
dqn=build_agent(states=states , actions=actions , model=q_model)
q_model.compile(adam_v2.Adam(learning_rate=0.5) ,loss='mean_absolute_error')



avg_score=0
rounds=0
prev_time=0
EPISODES=500000
for episodes in range(EPISODES):
    state=env.reset()
    state=state[0]

    state=np.reshape(state , [1,1,states])

    for time in range (500):
        cur_action = dqn.act(state)
        next_state, reward, done,trunc , info = env.step(cur_action)
        
        next_state = np.reshape(next_state, [1,1, states])
        
        dqn.remember(state, cur_action, reward, next_state, done)
        state = next_state
        if done:
            rounds+=1
            avg_score+=time
            print(f"Episode: {episodes + 1}/{EPISODES}, Score: {time + 1}, Avg: {avg_score/rounds}")
            break

    if  len(dqn.memory)>32 :
                
        dqn.replay(32)

env=gym.make('CartPole-v1',render_mode="human")
EPISODES=100
for episodes in range(EPISODES):
    state=env.reset()
    state=state[0]
    state=np.reshape(state , [1,1,states])

    for time in range (1000):
        env.render()
        cur_action = dqn.act(state)
        next_state, reward, done,trunc , info = env.step(cur_action)
        
        next_state = np.reshape(next_state, [1,1, states])
        
        state = next_state
        if done:
            break


