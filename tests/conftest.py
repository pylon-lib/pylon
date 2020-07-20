import pytest
import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    '''Neural network with a single input (fixed) and two binary outputs.'''

    def __init__(self, w=None):
        super().__init__()
        if w is not None:
            self.w = torch.nn.Parameter(torch.tensor(w).float().view(4, 1))
        else:
            self.w = torch.nn.Parameter(torch.rand(4, 1))

    def forward(self, x):
        return torch.matmul(self.w, x).view(2, 2)


def xor(y):
    return (y[0] and not y[1]) or (not y[0] and y[1])


def train(net, data, constraint=None, epoch=100):
    x, y = data
    y0 = F.softmax(net(x), dim=-1)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)

    for _ in range(100):
        opt.zero_grad()
        y_logit = net(x)
        loss = F.cross_entropy(y_logit[1:], y[1:])
        if constraint is not None:
            loss += constraint(y_logit)
        loss.backward()
        opt.step()

    return net, y0


@pytest.fixture
def net():
    net = Net()
    return net


@pytest.fixture
def data():
    '''Set input and one of the outputs to be 1.'''
    x = torch.tensor([1.0])
    y = torch.tensor([0, 1])
    return x, y
