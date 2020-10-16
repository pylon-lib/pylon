import torch


def decoding_loss(values, log_probs):
    '''Loss for a single decoded value of the variables.'''
    # sum of log prob
    # prob -> logprob = (0,1] -> (-inf, 0]

    loss = 0
    for i in range(len(log_probs)):

        loss_i = log_probs[i].gather(-1, values[i].unsqueeze(-1)).squeeze(-1)

        if len(log_probs[i].shape) == 3:
            # for sequence of labels where
            #     Sizeof log_probs[i]: batch_size x N x classes
            #     Sizeof values[i]: batch_size x N
            loss += loss_i.sum(dim=-1)    # sum over sequence and result in shape (batch_size,)

        elif len(log_probs[i].shape) == 2:
            # for instance-wise labels where
            #     Sizeof log_probs[i]: batch_size x classes
            #     Sizeof values[i]: batch_size
            loss += loss_i	# shape (batch_size,)

        else:
            raise Exception('failed to generalize for log_prob shape {0} and value shape {1}'.format(
                log_probs[i].shape, values[i].shape))


    return loss
