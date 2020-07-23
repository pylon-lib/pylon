import pytest
import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    '''Neural network with a single input (fixed) and two binary outputs.'''

    def __init__(self, w=None):
        super().__init__()
        if w is not None:
            self.w = torch.nn.Parameter(torch.tensor(w).float().view(4, 1))
        else:
            self.w = torch.nn.Parameter(torch.rand(4, 1))

    def forward(self, x):
        return torch.matmul(self.w, x).view(2, 2)


class NER_Net(torch.nn.Module):
    '''Simple Named Entity Recognition model'''

    def __init__(self, vocab_size, num_classes, hidden_dim=50, embedding_dim=300):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # layers
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim, num_classes)

        # Initialize fully connected layer
        self.fc.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=1)

    def forward(self, s):
        s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim
        s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim
        s = self.fc(s)          # dim: batch_size*batch_max_len x num_tags

        return s


class RE_Net(NER_Net):
    '''Simple Relation extraction model which extends the NER model'''

    def __init__(self, vocab_size, num_classes, hidden_dim=50, embedding_dim=300):
        super().__init__(vocab_size, num_classes, hidden_dim, embedding_dim)

        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=2,
                                  batch_first=True, dropout=0.7, bidirectional=True)


def xor(y):
    return (y[0] and not y[1]) or (not y[0] and y[1])


def train(net, data, constraint=None, epoch=100):
    x, y = data
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


@pytest.fixture
def net():
    net = Net()
    return net


@pytest.fixture
def data():
    '''Set input and one of the outputs to be 1.'''
    x = torch.tensor([1.0])
    y = torch.tensor([0, 1])
    return x, y
