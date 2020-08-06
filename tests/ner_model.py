import pytest
import torch
import torch.nn.Module


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
