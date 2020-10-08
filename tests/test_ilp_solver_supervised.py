from .basic_model import net_binary
from pytorch_constraints.constraint import constraint
from pytorch_constraints.ilp_solver import ILPSolver
import pytest
import torch
import sys
sys.path.append('..')


def train(net, constraint=None, epoch=100):
    x = torch.tensor([1.0])
    y = torch.tensor([0, 1])
    y0 = torch.softmax(net(x), dim=-1)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)

    for _ in range(100):
        opt.zero_grad()
        y_logit = net(x)
        loss = torch.nn.functional.cross_entropy(y_logit[1:], y[1:])
        if constraint is not None:
            loss += constraint(y_logit, targets=y)
        loss.backward()
        opt.step()

    return net, y0


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
        y_ = solver.inference(y)

        if y_[0][0][1] < 0.25:
            success += 1
        assert y_[0][1][1] > 0.75

    assert success == num_tries


if __name__ == '__main__':
    pytest.main([__file__])
