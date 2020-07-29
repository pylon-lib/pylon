import torch.nn.functional as F

from pytorch_constraints.weighted_sampling_solver import *
from pytorch_constraints.constraint import constraint

from .conftest import train, xor


def test_weighted_sampling(net, data):
    num_samples = 100
    num_tries = 10  # since it's random
    success = 0
    for i in range(num_tries):
        cons = constraint(xor, WeightedSamplingSolver(num_samples))

        net, y0 = train(net, data, cons)
        x, _ = data
        y = F.softmax(net(x), dim=-1)

        if y[0, 0] > 0.5 and y[0, 1] < 0.5:
            success += 1
        assert y[1, 0] < 0.5 and y[1, 1] > 0.5

    assert success == num_tries
