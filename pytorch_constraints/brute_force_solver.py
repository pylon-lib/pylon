import torch
import itertools

from .solver import Solver
from .utils import decoding_loss


class BruteForceSolver(Solver):
    '''Instantiates all possible decodings and checks whether the constraint is satisfied or not. Has two variations, depending on whether the loss is specified on satisfied decodings or unsatisfied decodings.'''

    def all_samples(self, log_probs):
        '''All possible decodings from the `log_probs` Tensor.'''

        vals = tuple(
                torch.cartesian_prod(
                    *[torch.tensor(range(log_probs[i].shape[-1]))] * log_probs[i].shape[-2]
                ).unsqueeze(1).expand(-1, log_probs[i].shape[0], -1) for i in range(len(log_probs)))

        return list(itertools.product(*vals))

    def filter(self, value):
        '''Which variable values should we compute the loss over.'''
        raise NotImplementedError

    def reduce(self, losses):
        '''How do we aggregate the losses for each decoding.'''
        return sum(losses)

    def loss(self, *logits, **kwargs):

        log_probs = [torch.log_softmax(logits[i], dim=-1) for i in range(len(logits))]
        samples = self.all_samples(log_probs)

        indices = torch.stack([torch.tensor(data=self.cond(*sample), dtype=torch.bool) for sample in samples])
        losses = torch.stack([decoding_loss(sample, log_probs) for sample in samples])

        return -(losses[indices].logsumexp(dim=0) - losses.logsumexp(dim=(0,1)))


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

        ##TODO: The below is a hack to get the xor examples to work
        ## What should be the expectation in this case?
        #if len(log_probs[0].shape) == 2:
        #    log_probs = [log_probs[i].reshape(1, *log_probs[i].shape) for i in range(len(logits))]
