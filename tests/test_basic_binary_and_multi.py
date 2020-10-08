from .basic_model import net_binary, net_multi, train
from pytorch_constraints.circuit_solver import SemanticLossCircuitSolver
from pytorch_constraints.tnorm_solver import *
from pytorch_constraints.sampling_solver import *
from pytorch_constraints.constraint import constraint
from pytorch_constraints.brute_force_solver import *
from pytorch_constraints.ilp_solver import ILPSolver
import torch.nn.functional as F
import torch
import sys
sys.path.append('../')

import pytest
import torch
import torch.nn.functional as F

def get_tnorm_solvers():
    return [
        ProductTNormLogicSolver(), LukasiewiczTNormLogicSolver(), GodelTNormLogicSolver()
    ]


def get_sampling_solvers(num_samples):
    return [
        #SatisfactionBruteForceSolver(), ViolationBruteForceSolver(),
        SamplingSolver(num_samples), WeightedSamplingSolver(num_samples)
    ]

def get_solvers(num_samples):
        return get_sampling_solvers(num_samples) + get_tnorm_solvers() + [ILPSolver()]

class Net(torch.nn.Module):
    '''Neural network with a single input (fixed) and two categorical outputs.'''

    def __init__(self, num_labels, w=None):
        super().__init__()
        self.num_labels = num_labels
        if w is not None:
            self.w = torch.nn.Parameter(
                torch.tensor(w).float().view(
                    self.num_labels*2, 1))
        else:
            self.w = torch.nn.Parameter(
                torch.rand(self.num_labels*2, 1))

    def forward(self, x):
        return torch.matmul(self.w, x).view(2, self.num_labels)

def train(net, constraint=None, epoch=100):
    x = torch.tensor([1.0])
    y = torch.tensor([0, 1])
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

def my_or(x,y):
    return  x[0] or y[0]

def test_or_binary_and_multi():

    net_binary = Net(2)
    net_multi = Net(3)

    x = torch.tensor([1.0])
    y = torch.tensor([0, 1])

    solvers = get_solvers(num_samples=20)
    for solver in solvers:

        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):

            cons = constraint(my_or, solver)
            opt = torch.optim.SGD(list(net_binary.parameters()) + list(net_multi.parameters()), lr=0.1)
            
            for _ in range(100):
                opt.zero_grad()
                y0_logit = net_binary(x)
                y1_logit = net_multi(x)

                loss = F.cross_entropy(y0_logit[1:], y[1:])
                loss += F.cross_entropy(y1_logit[1:], y[1:])
                loss += cons(y0_logit, y1_logit)

                loss.backward()
                opt.step()

            x = torch.tensor([1.0])
            y0 = F.softmax(net_binary(x), dim=-1)
            y1 = F.softmax(net_multi(x), dim=-1)

            if not (y0[0, 0] > 0.6 and y0[1, 0] > 0.6 and y1[0,0] > 0.6 and y1[1,0] > 0.6):
                success += 1

        assert success == num_tries


if __name__ == "__main__":
    pytest.main([__file__])
