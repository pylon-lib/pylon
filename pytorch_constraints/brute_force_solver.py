import torch
import itertools

from .solver import Solver
from .utils import decoding_loss


class BruteForceSolver(Solver):
    '''Instantiates all possible decodings and checks whether the constraint is satisfied or not. Has two variations, depending on whether the loss is specified on satisfied decodings or unsatisfied decodings.'''

    def all_samples(self, log_probs):
        '''All possible decodings from the `log_probs` Tensor.'''

        llgprbs = len(log_probs)
        shapes = [log_probs[i].shape for i in range(llgprbs)]

        batch_size = shapes[0][0]

        # values: tuple of 1D tensors; values[i].shape: number of classes for
        # the classifier corresponding to log_probs[i]
        decodings =  tuple(torch.tensor(range(shapes[i][-1])) for i in range(llgprbs))

        # Shape: \prod_i num_class_i
        decodings = torch.cartesian_prod(*decodings)

        # Reshaping
        decodings = decodings.unsqueeze(-1).expand(-1, -1, batch_size)
        decodings = [decoding.unbind(0) for decoding in decodings]

        return decodings

    def filter(self, value):
        '''Which variable values should we compute the loss over.'''
        raise NotImplementedError

    def reduce(self, losses):
        '''How do we aggregate the losses for each decoding.'''
        return sum(losses)

    def loss(self, *logits, **kwargs):

        log_probs = [torch.log_softmax(logits[i], dim=-1) for i in range(len(logits))]
        batch_size = log_probs[0].shape[0] 

        # Get all possible decodings
        samples = self.all_samples(log_probs)

        indices = torch.stack([self.cond(*sample, kwargs) for sample in samples])

        losses = torch.stack([decoding_loss(sample, log_probs) for sample in samples])

        sat_losses = losses.clone()
        sat_losses[~indices] = -float('inf')

        loss = sat_losses.logsumexp(dim=0) - losses.logsumexp(dim=0)
        return -loss.sum()

class SatisfactionBruteForceSolver(BruteForceSolver):
    '''Add a loss that encourages the total probability over all possible decodings that satisfy the constraint.'''

    def filter(self, value):
        return self.cond(*value)

    def reduce(self, losses):
        # we want to maximize this target
        return -torch.stack(tuple(losses)).logsumexp(dim=0)


class ViolationBruteForceSolver(BruteForceSolver):
    '''Add a loss that discourages the total probability over all possible decodings that violate the constraint.'''

    def filter(self, value):
        return not self.cond(*value)

    def reduce(self, losses):
        return torch.stack(tuple(losses)).logsumexp(dim=0)
