import torch


def decoding_loss(values, log_probs):
    '''Loss for a single decoded value of the variables.'''
    # sum of log prob
    # prob -> logprob = (0,1] -> (-inf, 0]
    #
    loss = 0
    for log_prob, value in zip(log_probs, values):
        loss += log_prob[value]
    return loss
