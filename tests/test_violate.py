import torch
import torch.nn.functional as F

from pytorch_constraints.constraint.violation_penalty import ViolationLambda

from .test_basic import net, data, train, xor


def test_violate(net, data):
    constraint = ViolationLambda(xor)

    net, _ = train(net, data, constraint)
    x, _ = data
    y = F.softmax(net(x), dim=-1)

    assert y[0,0] > 0.5 and y[0,1] < 0.5
    assert y[1,0] < 0.5 and y[1,1] > 0.5
