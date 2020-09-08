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


def get_tnorm_solvers():
    return [
        ProductTNormLogicSolver(), LukasiewiczTNormLogicSolver(), GodelTNormLogicSolver()
    ]


def get_sampling_solvers(num_samples):
    return [
        SatisfactionBruteForceSolver(), ViolationBruteForceSolver(),
        SamplingSolver(num_samples), WeightedSamplingSolver(num_samples)
    ]


def get_solvers(num_samples):
    return get_sampling_solvers(num_samples) + [SemanticLossCircuitSolver()] + get_tnorm_solvers()


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
            cons = constraint(lambda y: y[0] or y[1], solver)

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
            cons = constraint(lambda y: y[0] or y[1], solver)

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


def test_implication_multi(net_multi):
    # TODO, currently only implemented in tnorms solvers
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


def test_quant_forall_list(net_binary):
    solvers = get_sampling_solvers(num_samples=20) + get_tnorm_solvers()
    for solver in solvers:
        print("Testing", type(solver).__name__)
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(lambda y: all([y[0], y[1]]), solver)

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
            cons = constraint(lambda y: all([not y[0], y[1]]), solver)

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
            cons = constraint(lambda y: any([y[0], not y[1]]), solver)

            net, y0 = train(net_binary, cons)
            x = torch.tensor([1.0])
            y = F.softmax(net(x), dim=-1)
            if y[0, 1] > 0.75:
                success += 1
            assert y[1, 1] > 0.8

        assert success == num_tries
