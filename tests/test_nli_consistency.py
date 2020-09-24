import torch
import torch.nn.functional as F

from pytorch_constraints.constraint import constraint
from pytorch_constraints.tnorm_solver import *

LABEL_TO_ID = {'Entailment': 0, 'Contradiction': 1, 'Neutral': 2}
ENT = LABEL_TO_ID['Entailment']
CON = LABEL_TO_ID['Contradiction']
NEU = LABEL_TO_ID['Neutral']

def get_solvers():
    return [ProductTNormLogicSolver()]

class NLI_Net(torch.nn.Module):
    '''Simple NLI model'''

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
        p, h = s
        p = self.embedding(p)   # dim: batch_size x batch_max_len x embedding_dim
        h = self.embedding(h)   # dim: batch_size x batch_max_len x embedding_dim
        s = torch.cat([p, h], dim=1)
        s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim
        s = s.max(dim=1)[0]     # dim: batch_size x lstm_hidden_dim
        s = self.fc(s)          # dim: batch_size x num_tags

        return s


def transitivity(ph_batch, hz_batch, pz_batch):
    # TODO, can't do this:
    # ENT = LABEL_TO_ID['Entailment']
    # (ph_batch[ENT] and hz_batch[ENT]) <= pz_batch[ENT]
    ee_e = (ph_batch == 0 and hz_batch == 0) <= (pz_batch == 0)
    ec_c = (ph_batch == 0 and hz_batch == 1) <= (pz_batch == 1)
    ne_notc = (ph_batch == 2 and hz_batch == 0) <= (not (pz_batch == 1))
    nc_note = (ph_batch == 2 and hz_batch == 1) <= (not (pz_batch == 0))
    block_safezone = (ph_batch == 2 and hz_batch == 2) <= (not (pz_batch == 2)) # just block Neu and Neu -> Neu and force it to change
    return ee_e and ec_c and ne_notc and nc_note and block_safezone

# inputs are binary masks where 1 is the spiked label and 0's are the rest
def transitivity_check(ph_y_mask, hz_y_mask, pz_y_mask):
    ee_e = ph_y_mask[:, ENT].logical_and(hz_y_mask[:, ENT]).logical_not().logical_or(pz_y_mask[:, ENT]).all()
    ec_c = ph_y_mask[:, ENT].logical_and(hz_y_mask[:, CON]).logical_not().logical_or(pz_y_mask[:, CON]).all()
    ne_notc = ph_y_mask[:, NEU].logical_and(hz_y_mask[:, ENT]).logical_not().logical_or(pz_y_mask[:, CON].logical_not()).all()
    nc_note = ph_y_mask[:, NEU].logical_and(hz_y_mask[:, CON]).logical_not().logical_or(pz_y_mask[:, ENT].logical_not()).all()
    block_safezone = ph_y_mask[:, NEU].logical_and(hz_y_mask[:, NEU]).logical_not().logical_or(pz_y_mask[:, NEU].logical_not()).all()
    return ee_e and ec_c and ne_notc and nc_note and block_safezone


def train(data, constraint):
    
    nli = NLI_Net(vocab_size=3027, num_classes=len(LABEL_TO_ID))

    opt = torch.optim.SGD(list(nli.parameters()), lr=1.0)
    
    ph_tokens, hz_tokens, pz_tokens, ph_y = data

    for i in range(100):
        opt.zero_grad()

        ph_logits = nli(ph_tokens)
        hz_logits = nli(hz_tokens)
        pz_logits = nli(pz_tokens)

        yloss = F.cross_entropy(ph_logits, ph_y.view(-1))
        closs = constraint(ph_logits, hz_logits, pz_logits)
        loss = 0.5 * closs + 0.95 * yloss 

        loss.backward()
        opt.step()
        
    return nli

    
def test_nli():
    ph_tokens, hz_tokens, pz_tokens, ph_y = get_data()

    for solver in get_solvers():
        
        cons = constraint(transitivity, solver)
        nli = train([ph_tokens, hz_tokens, pz_tokens, ph_y], cons)

        ph_y_ = torch.softmax(nli(ph_tokens).view(-1, len(LABEL_TO_ID)), dim=-1)
        hz_y_ = torch.softmax(nli(hz_tokens).view(-1, len(LABEL_TO_ID)), dim=-1)
        pz_y_ = torch.softmax(nli(pz_tokens).view(-1, len(LABEL_TO_ID)), dim=-1)

        ph_y_mask = (ph_y_ == ph_y_.max(-1)[0].view(-1, 1))
        hz_y_mask = (hz_y_ == hz_y_.max(-1)[0].view(-1, 1))
        pz_y_mask = (pz_y_ == pz_y_.max(-1)[0].view(-1, 1))

        sup_rs = ph_y_mask[:, NEU].all()
        transitivity_rs = transitivity_check(ph_y_mask, hz_y_mask, pz_y_mask)
        
        assert sup_rs and transitivity_rs
    
def get_data():
    
    p = torch.tensor([[  32, 1973, 2272,   15,    3,    0,    0,    5,    0,  389,    0,   12,
            7,  823,    4, 2636,    4,    0,  114,    5,    3, 2701,    6]])
    h = p + 100
    z = h + 100
    y = torch.tensor([NEU])
    
    return [p, h], [h, z], [p, z], y
