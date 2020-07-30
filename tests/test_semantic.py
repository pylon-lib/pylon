import torch.nn.functional as F

from pytorch_constraints.constraint import constraint
from pytorch_constraints.semantic_solver import SemanticSolver

from .conftest import train, xor

def test_semantic_solver(net, data):
    num_tries = 10  # since it's random
    success = 0
    for i in range(num_tries):
        cons = constraint(xor, SemanticSolver())

        net, y0 = train(net, data, cons)
        x, _ = data
        y = F.softmax(net(x), dim=-1)

        if y[0, 0] > 0.5 and y[0, 1] < 0.5:
            success += 1
        assert y[1, 0] < 0.5 and y[1, 1] > 0.5

    assert success == num_tries
