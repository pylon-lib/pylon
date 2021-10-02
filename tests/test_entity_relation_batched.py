import torch
import torch.nn.functional as F

from pylon.constraint import constraint
from pylon.sampling_solver import *

ENTITY_TO_ID = {"O": 0, "Loc": 1, "Org": 2, "Peop": 3, "Other": 4}
REL_TO_ID = {"*": 0, "Work_For_arg1": 1, "Kill_arg1": 2, "OrgBased_In_arg1": 3, "Live_In_arg1": 4,
             "Located_In_arg1": 5, "Work_For_arg2": 6, "Kill_arg2": 7, "OrgBased_In_arg2": 8,
             "Live_In_arg2": 9, "Located_In_arg2": 10}


def get_solvers(num_samples):
    return [WeightedSamplingSolver(num_samples)]


class NER_Net(torch.nn.Module):
    '''Simple Named Entity Recognition model'''

    def __init__(self, vocab_size, num_classes, hidden_dim=50, embedding_dim=100):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # layers
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        #self.embedding.weight = torch.nn.Parameter(vocab.vectors)
        self.embedding.weight.data.uniform_(-1.0, 1.0)

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


class RE_Net(torch.nn.Module):
    '''Simple Relation extraction model'''

    def __init__(self, vocab_size, num_classes, hidden_dim=50, embedding_dim=100):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # layers
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        #self.embedding.weight = torch.nn.Parameter(vocab.vectors)
        self.embedding.weight.data.uniform_(-1.0, 1.0)

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


def OrgBasedIn_Org_Loc(ne_batch, re_batch):

    out = []
    for ne, re in zip(ne_batch, re_batch):
        arg1 = (re == 3).nonzero(as_tuple=False)
        arg2 = (re == 8).nonzero(as_tuple=False)

        out += [all(ne[arg1] == 2) and all(ne[arg2] == 1)]

    return torch.tensor(out)


def train(constraint):

    ner = NER_Net(vocab_size=3027, num_classes=len(ENTITY_TO_ID))
    re = RE_Net(vocab_size=3027, num_classes=len(REL_TO_ID))

    opt = torch.optim.SGD(list(ner.parameters()) + list(re.parameters()), lr=1.0)

    tokens, entities, relations = get_data()

    for i in range(100):
        opt.zero_grad()

        ner_logits = ner(tokens)

        re_logits = re(tokens)

        re_loss = F.cross_entropy(re_logits.view(-1, re_logits.shape[2]), relations.view(-1))
        closs = constraint(ner_logits, re_logits)
        loss = 0.05 * closs + 10 * re_loss

        loss.backward()
        opt.step()

    return ner, re


def test_entity_relation():

    tokens, entities, relations = get_data()
    tokens = torch.cat((tokens, tokens, tokens))
    entities = torch.cat((entities, entities, entities))
    relations = torch.cat((relations, relations, relations))

    for solver in get_solvers(num_samples=200):

        cons = constraint(OrgBasedIn_Org_Loc, solver)
        ner, re = train(cons)

        re = torch.argmax(torch.softmax(re(tokens).view(-1, 11), dim=-1), dim=-1)
        ner = torch.argmax(torch.softmax(ner(tokens).view(-1, 5), dim=-1), dim=-1)

        assert (ner[re == 3] == 2).all() and (ner[re == 8] == 1).all()


def get_data():

    tokens = torch.tensor([[32, 1973, 2272,   15,    3,    0,    0,    5,    0,  389,    0,   12,
                            7,  823,    4, 2636,    4,    0,  114,    5,    3, 2701,    6]])
    entities = torch.LongTensor([0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0])
    relations = torch.LongTensor([0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    return tokens, entities, relations
