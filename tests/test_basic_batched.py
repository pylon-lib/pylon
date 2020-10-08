from .basic_model import net_binary, net_multi, train
from pytorch_constraints.circuit_solver import SemanticLossCircuitSolver
from pytorch_constraints.tnorm_solver import *
from pytorch_constraints.sampling_solver import *
from pytorch_constraints.constraint import constraint
from pytorch_constraints.brute_force_solver import *
import torch.nn.functional as F
import torch
import sys
sys.path.append('../')


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
        return torch.matmul(self.w, x).view(-1, 2, self.num_labels)


def train(net, constraint=None, epoch=100):
    x = torch.cat([torch.tensor([1.0])]*2).reshape(2, 1, 1)
    y = torch.cat([torch.tensor([0, 1])]*2)

    y0 = F.softmax(net(x), dim=-1)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)

    for _ in range(100):
        opt.zero_grad()
        y_logit = net(x)
        loss = F.cross_entropy(y_logit[:, 1:].reshape(2, 2), torch.tensor([1, 1]))
        if constraint is not None:
            loss += constraint(y_logit)
        loss.backward()
        opt.step()

    return net, y0


def xor(y):
    return torch.logical_or(
        y[:, 0].logical_and(y[:, 1].logical_not()),
        y[:, 0].logical_not().logical_and(y[:, 1]))


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
    return get_sampling_solvers(num_samples)  # + [SemanticLossCircuitSolver()] + get_tnorm_solvers()


def test_xor_batch():
    net_binary = Net(2)
    solvers = get_solvers(num_samples=10)
    for solver in solvers:
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(xor, solver)

            net, y0 = train(net_binary, cons)
            x = torch.cat([torch.tensor([1.0])]*2).reshape(2, 1, 1)
            y = F.softmax(net(x), dim=-1)

            if all(y[:, 0, 0] > 0.75) and all(y[:, 0, 1] < 0.25):
                success += 1

            assert all(y[:, 1, 0] < 0.25) and all(y[:, 1, 1] > 0.75)

        assert success == num_tries
