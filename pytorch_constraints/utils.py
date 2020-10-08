import torch


def decoding_loss(values, log_probs):
    '''Loss for a single decoded value of the variables.'''
    # sum of log prob
    # prob -> logprob = (0,1] -> (-inf, 0]

    loss = 0
    for i in range(len(log_probs)):

        # Sizeof log_probs[i]: batch_size x N x classes
        # Sizeof values[i]: batch_size x N
        loss += log_probs[i].gather(-1, values[i].unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    return loss
