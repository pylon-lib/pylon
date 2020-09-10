import pytest
import torch
import sys
sys.path.append('..')

from pytorch_constraints.ilp_solver import ILPSolver
from pytorch_constraints.constraint import constraint

from .basic_model import net_binary, train


def xor(y):
    return (y[0] and not y[1]) or (not y[0] and y[1])


def test_xor_binary(net_binary):
    solver = ILPSolver()
    num_tries = 5  # since it's random
    success = 0
    for i in range(num_tries):
        cons = constraint(xor, solver)

        net, y0 = train(net_binary, cons)
        x = torch.tensor([1.0])
        y = torch.softmax(net(x), dim=-1)
        y = solver.inference(y)
        y = torch.tensor(y)

        if y[0] < 0.25:
            success += 1
        assert y[1] > 0.75

    assert success == num_tries


if __name__ == '__main__':
    pytest.main([__file__])
