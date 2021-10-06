# Pylon: A PyTorch Framework for Learning with Constraints

![Python package](https://github.com/ucinlp/pytorch-constraints/workflows/Python%20package/badge.svg)

## Dependencies
- Python >= 3.6
- torch>=1.9.0
- astor

## Installation

Optional, set up virtualenv:
```
python3 -m venv /path/to/env
source /path/to/env/bin/activate
```
Install using pip:
```
pip install pylon-lib
```

Alternatively, compile from source:
```
git clone https://github.com/pylon-lib/pylon.git
cd pylon
python3 -m pip install --upgrade pip
pip install flake8 pytest
pip install -r requirements.txt
```
Make sure to install PyTorch: https://pytorch.org

## Basic Example
Our goal is to enforce the XOR constraint on the output of a simple classifier:
only one of the outputs can be "on" i.e. set to 1

```
import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, w=None):
        super().__init__()
        if w is not None:
            self.w = torch.nn.Parameter(torch.tensor(w).float().view(6, 1))
        else:
            self.w = torch.nn.Parameter(torch.rand(6, 1))

    def forward(self, x):
        return torch.matmul(self.w, x).view(3, 2)
```
We define our constraint funciton
```
from pylon.constraint import constraint
from pylon.brute_force_solver import SatisfactionBruteForceSolver

# Our constraint function accepts a decoding tensor of
# shape (batch_size, ...) and is expected to return
# a tensor fo shape (batch_size, )
def xor(y):
    return y[:, 0] != y[:, 1] and y[:, 1] != y[:, 2]
    
xor_cons = constraint(xor, SatisfactionBruteForceSolver())
```
And proceed to our training loop
```
# Create network and optimizer
net = Net()
opt = torch.optim.SGD(net.parameters(), lr=0.1)

# Input and label
x = torch.tensor([1.])
y = torch.tensor([0, 0, 1])

# training loop
y0, y1, y2 = [], [], []
for i in range(500):
    opt.zero_grad()
    y_logit = net(x)
    loss = F.cross_entropy(y_logit[2:], y[2:])
    loss += xor_cons(y_logit.unsqueeze(0)) #Pylon expect tensors of shape (batch_size, ...)
    loss.backward()
    y_prob = torch.softmax(y_logit, dim=-1)
    y0.append(y_prob[0,1].data); y1.append(y_prob[1,1].data); y2.append(y_prob[2,1].data)
    opt.step()

import matplotlib.pyplot as plt
plt.plot(y0, label='y0')
plt.plot(y1, label='y1')
plt.plot(y2, label='y2')
plt.legend()
```
![Image](https://user-images.githubusercontent.com/2989475/135705681-ce62667f-cdf1-4b8a-9efc-db0fc9cefb2e.png)

