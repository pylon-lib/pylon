import torch


def decoding_loss(values, log_probs):
    '''Loss for a single decoded value of the variables.'''
    # sum of log prob
    # prob -> logprob = (0,1] -> (-inf, 0]
    loss = 0
    for i in range(len(log_probs)):
        # The input log_probs and values could come from different use cases:
        #   1. Sequence of labels
        #       sizeof log_probs[i]: (batch_size, N, classes)
        #       sizeof values[i]: (batch_size, N)
        #       and the result loss should have shape (batch_size,)
        #   2. Instance-wise labels
        #       sizeof log_probs[i]: (batch_size, classes)
        #       sizeof values[i]: (batch_size)
        #       and the result loss should have shape (batch_size,)
        #   3. unbatched model
        #       sizeof log_probs[i]: (N,..., classes)
        #       sizeof values[i]:   (N,...,)
        #       and the result loss should have shape (N,...,)
        #   In general, within this function, there is no good clue which use cases the inputs come from.
        #       So, here it only gathers (and thus squeezes the last dim). And the rest is to be handled outside of this func
        loss += log_probs[i].gather(-1, values[i].unsqueeze(-1)).squeeze(-1)

    return loss
