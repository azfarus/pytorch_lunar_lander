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


ip=[
    [1],
    [2]]

ip=torch.tensor(ip)

op=[11,12,13,14]
op=torch.tensor(op)

print(op.max(-1).indices.view(1,1))
