import sys
sys.path.append("/space/ahmedk/wtf")

from pylon.circuit_solver import SemanticLossCircuitSolver
from pylon.tnorm_solver import *
from pylon.sampling_solver import *
from pylon.constraint import constraint
from pylon.brute_force_solver import *
from pylon.ilp_solver import ILPSolver
import torch.nn.functional as F
import torch
import pytest


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
    opt = torch.optim.SGD(net.parameters(), lr=0.1)

    for _ in range(100):
        opt.zero_grad()
        y_logit = net(x)
        loss = F.cross_entropy(y_logit[1:], y[1:])
        if constraint is not None:
            loss += constraint(y_logit.unsqueeze(0))
        loss.backward()
        opt.step()

    y0 = F.softmax(net(x), dim=-1)
    return net, y0


@pytest.fixture
def net_binary():
    net = Net(2)
    return net


@pytest.fixture
def net_multi():
    net = Net(3)
    return net


a = {"key": 1}


def xor(y):
    return (y[:, 0] and y[:, a['key']].logical_not()) or (y[:, 0].logical_not() and y[:, a['key']])


def test_basic_binary(net_binary):
    net, y0 = train(net_binary)
    x = torch.tensor([1.0])
    y = F.softmax(net(x), dim=-1)
    assert (y[0] == y0[0]).all(), "y[0] should remain unchanged"
    assert y[1, 0] < 0.5 and y[1, 1] > 0.5


def test_basic_multi(net_multi):
    net, y0 = train(net_multi)
    x = torch.tensor([1.0])
    y = F.softmax(net(x), dim=-1)
    assert (y[0] == y0[0]).all(), "y[0] should remain unchanged"
    assert y[1, 0] < 0.2 and y[1, 1] > 0.6 and y[1, 2] < 0.2


def get_tnorm_solvers():
    return [
        ProductTNormLogicSolver(), LukasiewiczTNormLogicSolver(), GodelTNormLogicSolver()
    ]


def get_sampling_solvers(num_samples):
    return [
        #SatisfactionBruteForceSolver(), 
        SamplingSolver(num_samples), WeightedSamplingSolver(num_samples)
    ]


def get_solvers(num_samples):
    return get_sampling_solvers(num_samples) #+ [SemanticLossCircuitSolver()] #+ get_tnorm_solvers()


def test_xor_binary(net_binary):
    solvers = get_solvers(num_samples=10)
    for solver in solvers:
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(xor, solver)

            net, y0 = train(net_binary, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if y[0, 0] > 0.75 and y[0, 1] < 0.25:
                success += 1
            assert y[1, 0] < 0.25 and y[1, 1] > 0.75

        assert success == num_tries


def test_xor_multi(net_multi):
    solvers = get_solvers(num_samples=20)
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(xor, solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if y[0, 0] > 0.6 and y[0, 1] < 0.2 and y[0, 2] < 0.2:
                success += 1
            assert y[1, 0] < 0.2 and y[1, 1] > 0.6 and y[1, 2] < 0.2

        assert success == num_tries


def test_or_binary(net_binary):
    solvers = get_solvers(num_samples=10)
    for solver in solvers:
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[:, 0] or y[:, 1], solver)

            net, y0 = train(net_binary, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if not (y[0, 0] > 0.6 and y[1, 0] > 0.6):
                success += 1

        assert success == num_tries


def test_or_multi(net_multi):
    solvers = get_solvers(num_samples=20)
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[:, 0] or y[:, 1], solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if not (y[0, 0] > 0.6 and y[1, 0] > 0.6):
                success += 1

        assert success == num_tries


def test_eq_multi(net_multi):
    solvers = get_solvers(num_samples=20)
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[:, 0] == y[:, 1], solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)
            if y[0, 1] > 0.8:
                success += 1
            assert y[1, 1] > 0.9

        assert success == num_tries


def test_neq_multi(net_multi):
    solvers = get_solvers(num_samples=20)
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[:, 0] != y[:, 1], solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)
            if y[0, 1] < 0.2:
                success += 1
            assert y[1, 1] > 0.8

        assert success == num_tries


def test_logical_and_multi(net_multi):
    solvers = get_solvers(num_samples=20)
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[:, 0] and y[:, 1], solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if y[0, 0] < 0.2:
                success += 1
            assert y[1, 0] < 0.6

        assert success == num_tries


def test_implication_multi(net_multi):
    # TODO, currently only implemented in tnorms solvers
    solvers = get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[:, 0] <= y[:, 1], solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if y[0, 1:].max() < y[1, 1:].max():
                success += 1

        assert success == num_tries


def test_quant_forall_list(net_binary):
    solvers = get_sampling_solvers(num_samples=20) + get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: all([y[:, 0], y[:, 1]]), solver)

            net, y0 = train(net_binary, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)
            if y[0, 1] > 0.75:
                success += 1
            assert y[1, 1] > 0.8

        assert success == num_tries


def test_quant_forall_list_wnegs(net_binary):
    solvers = get_sampling_solvers(num_samples=20) + get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: all([y[:, 0].logical_not(), y[:, 1]]), solver)

            net, y0 = train(net_binary, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)
            if y[0, 1] < 0.25:
                success += 1
            assert y[1, 1] > 0.8

        assert success == num_tries


def test_quant_exists_list(net_binary):
    solvers = get_sampling_solvers(num_samples=20) + get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: any([y[:, 0], y[:, 1].logical_not()]), solver)

            net, y0 = train(net_binary, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)
            if y[0, 1] > 0.75:
                success += 1
            assert y[1, 1] > 0.8

        assert success == num_tries

def test_quant_forall_var(net_binary):
    solvers = get_sampling_solvers(num_samples=20) #+ get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: torch.all(y.bool()), solver)

            net, y0 = train(net_binary, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)
            if y[0, 1] > 0.75:
                success += 1
            assert y[1, 1] > 0.8

        assert success == num_tries


def test_quant_forall_cond(net_multi):
    solvers = get_sampling_solvers(num_samples=20) #+ get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: torch.all(y == 1, dim=1), solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)
            if y[0, 1] > 0.75:
                success += 1
            assert y[1, 1] > 0.8

        assert success == num_tries


def test_quant_exists_cond(net_multi):
    solvers = get_sampling_solvers(num_samples=20) #+ get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: torch.any(y == 2, dim=1), solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)
            if y[0, 2] > 0.75:
                success += 1
            assert y[1, 1] > 0.8

        assert success == num_tries


def test_quant_exists_var(net_binary):
    solvers = get_sampling_solvers(num_samples=20) #+ get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: torch.any(y.logical_not()), solver)

            net, y0 = train(net_binary, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)
            if y[0, 1] < 0.25:
                success += 1
            assert y[1, 1] > 0.8

        assert success == num_tries


if __name__ == "__main__":
    pytest.main([__file__])
