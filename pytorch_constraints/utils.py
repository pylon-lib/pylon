import torch


def decoding_loss(values, log_probs, reduce='sum'):
    '''Loss for a single decoded value of the variables.'''
    # sum of log prob
    # prob -> logprob = (0,1] -> (-inf, 0]

    loss = 0
    for i in range(len(log_probs)):
        for log_prob, value in zip(log_probs[i], values[i]):
            loss += log_prob.gather(1, value.view(-1,1))

    if reduce == 'sum':
        return loss.sum()
    elif reduce == 'mean':
        return loss.mean()
    else:
        return loss
