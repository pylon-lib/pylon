import torch.nn.functional as F

from pytorch_constraints.brute_force_solver import *
from pytorch_constraints.constraint import constraint

from .conftest import train, xor


def test_default(net, data):
    cons = constraint(xor)

    net, _ = train(net, data, cons)
    x, _ = data
    y = F.softmax(net(x), dim=-1)

    assert y[0, 0] > 0.5 and y[0, 1] < 0.5
    assert y[1, 0] < 0.5 and y[1, 1] > 0.5


def test_satisfy(net, data):
    cons = constraint(xor, SatisfactionBruteForceSolver())

    net, _ = train(net, data, cons)
    x, _ = data
    y = F.softmax(net(x), dim=-1)

    assert y[0, 0] > 0.5 and y[0, 1] < 0.5
    assert y[1, 0] < 0.5 and y[1, 1] > 0.5


def test_violate(net, data):
    cons = constraint(xor, ViolationBruteForceSolver())

    net, _ = train(net, data, cons)
    x, _ = data
    y = F.softmax(net(x), dim=-1)

    assert y[0, 0] > 0.5 and y[0, 1] < 0.5
    assert y[1, 0] < 0.5 and y[1, 1] > 0.5
