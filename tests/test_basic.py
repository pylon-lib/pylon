import torch.nn.functional as F

from .conftest import train


def test_basic(net, data):
    net, y0 = train(net, data)
    x, _ = data
    y = F.softmax(net(x), dim=-1)
    assert (y[0] == y0[0]).all(), "y[0] should remain unchanged"
    assert y[1, 0] < 0.5 and y[1, 1] > 0.5
