import itertools


class Solver:
    '''Method to compute the loss for the given constraint.'''

    def set_cond(self, cond):
        '''Sets the boolean condition that defines the constraint, performs any fixed computations, as needed.'''
        self.cond = cond

    def loss(self, logits):
        '''Return the loss for the constraint for the given logits of the variables involved in the constraint.'''
        raise NotImplementedError
