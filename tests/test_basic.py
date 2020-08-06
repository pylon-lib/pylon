import torch.nn.functional as F

from .basic_model import train, net, data

from pytorch_constraints.constraint import constraint
from pytorch_constraints.brute_force_solver import *
from pytorch_constraints.semantic_solver import SemanticSolver
from pytorch_constraints.sampling_solver import *
from pytorch_constraints.tnorm_solver import TNormLogicSolver


def xor(y):
    return (y[0] and not y[1]) or (not y[0] and y[1])


def test_basic_model(net, data):
    net, y0 = train(net, data)
    x, _ = data
    y = F.softmax(net(x), dim=-1)
    assert (y[0] == y0[0]).all(), "y[0] should remain unchanged"
    assert y[1, 0] < 0.5 and y[1, 1] > 0.5


def test_solvers(net, data):
    num_samples = 50
    solvers = [
        SatisfactionBruteForceSolver(), ViolationBruteForceSolver(),
        SamplingSolver(num_samples), WeightedSamplingSolver(num_samples),
        SemanticSolver(), TNormLogicSolver()
    ]
    for solver in solvers:
        num_tries = 5  # since it's random
        success = 0
        for i in range(num_tries):
            cons = constraint(xor, solver)

            net, y0 = train(net, data, cons)
            x, _ = data
            y = F.softmax(net(x), dim=-1)

            if y[0, 0] > 0.5 and y[0, 1] < 0.5:
                success += 1
            assert y[1, 0] < 0.5 and y[1, 1] > 0.5

        assert success == num_tries
