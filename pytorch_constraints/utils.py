import torch
import warnings

def decoding_loss(values, log_probs):
    '''Loss for a single decoded value of the variables.'''

    loss = 0
    bsz = log_probs[0].size(0)

    # Shape of log_prob: batch_size x ... x classes
    # Shape of values: batch_size x ...
    for log_prob, value in zip(log_probs, values):

        # the log probability of a decoding is the
        # sum of log probabilities of the value
        # assumed by each random variable in the
        # decoding.

        # Shape of loss: batch_size x ...
        loss += log_prob.gather(-1, value.unsqueeze(-1))\
                .squeeze(-1).view(bsz, -1).sum(-1)

    # Reduce the tensor across the extraneous dimensions;
    # this corresponds to calculating the log probability
    # of that decoding of the combinatorial object
    return loss
