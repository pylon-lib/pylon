import sys
sys.path.append('../')
import torch
import torch.nn.functional as F

from pytorch_constraints.brute_force_solver import *
from pytorch_constraints.constraint import constraint
from pytorch_constraints.sampling_solver import *
from pytorch_constraints.tnorm_solver import *
from pytorch_constraints.circuit_solver import SemanticLossCircuitSolver

from .basic_model import net_binary, net_multi, train


def xor(y):
    return (y[0] and not y[1]) or (not y[0] and y[1])


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


def get_solvers(num_samples):
    return [
        SatisfactionBruteForceSolver(), ViolationBruteForceSolver(),
        SamplingSolver(num_samples), WeightedSamplingSolver(num_samples),
        SemanticLossCircuitSolver(),
        ProductTNormLogicSolver(), LukasiewiczTNormLogicSolver(), GodelTNormLogicSolver()
    ]

def get_tnorm_solvers():
    return [
        ProductTNormLogicSolver(), LukasiewiczTNormLogicSolver(), GodelTNormLogicSolver()
    ]


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


def test_eq_multi(net_multi):
    solvers = get_solvers(num_samples=20)
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[0] == y[1], solver)

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
            cons = constraint(lambda y: y[0] != y[1], solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)
            if y[0, 1] < 0.2:
                success += 1
            assert y[1, 1] > 0.8

        assert success == num_tries

def test_logical_and_binary(net_binary):
    solvers = get_solvers(num_samples=10)
    for solver in solvers:
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[0] and y[1], solver)

            net, y0 = train(net_binary, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if y[0, 0] < 0.25:
                success += 1
            assert y[1, 0] < 0.25

        assert success == num_tries


def test_logical_and_multi(net_multi):
    solvers = get_solvers(num_samples=20)
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[0] and y[1], solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if y[0, 0] < 0.2:
                success += 1
            assert y[1, 0] < 0.6

        assert success == num_tries

# TODO, currently only implemented in tnorms solvers
def test_implication_binary(net_multi):
    solvers = get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[0] <= y[1], solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if y[0, 1:].max() < y[1, 1:].max():
                success += 1

        assert success == num_tries

# TODO, currently only implemented in tnorms solvers
def test_implication_multi(net_multi):
    solvers = get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[0] <= y[1], solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if y[0, 1:].max() < y[1, 1:].max():
                success += 1

        assert success == num_tries


# TODO, sigmoidal implication are currently only implemented in tnorm solvers
def test_s_implication_binary(net_multi):
    solvers = get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[0].s_implies(y[1]), solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if y[0, 1:].max() < y[1, 1:].max():
                success += 1

        assert success == num_tries

# TODO, sigmoidal implication are currently only implemented in tnorm solvers
def test_s_implication_multi(net_multi):
    solvers = get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: y[0].s_implies(y[1]), solver)

            net, y0 = train(net_multi, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)

            if y[0, 0] > y[1, 0]:
                success += 1

        assert success == num_tries
