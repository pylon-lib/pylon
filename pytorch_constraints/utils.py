import torch
import warnings

def decoding_loss(values, log_probs):
    '''Loss for a single decoded value of the variables.'''

    assert(len(value.shape) == 1 for value in values)
    assert(len(log_probs.shape) == 2 for value in values)

    loss = 0
    bsz = log_probs[0].size(0)

    # Shape of log_prob: batch_size x classes
    # Shape of values: batch_size
    for log_prob, value in zip(log_probs, values):

        # the log probability of a decoding is the
        # sum of log probabilities of the value
        # assumed by each random variable in the
        # decoding.
        loss += log_prob[range(bsz), value]

    return loss
