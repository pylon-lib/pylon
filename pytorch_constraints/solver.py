import itertools

import ast
import inspect

from .ast_visitor import FunDefFindingVisitor
from .ast_visitor import LogicExpressionASTVisitor


class Solver:
    '''Computes the loss for the given constraint.'''

    def set_cond(self, cond):
        '''Sets the boolean condition that defines the constraint, performs any fixed computations, as needed.'''
        self.cond = cond

    def loss(self, *logits, **kwargs):
        '''Return the loss for the constraint for the given logits of the variables involved in the constraint.'''
        raise NotImplementedError


class ASTLogicSolver(Solver):
    '''Computes the loss by looking at code structure.'''

    def get_bool_tree(self):
        # TODO: need to do this only once (problem is eval in ASTVisitor)
        source = inspect.getsource(self.cond).strip()
        astree = ast.parse(source)
        fundef = FunDefFindingVisitor().visit(astree)
        self.visitor = LogicExpressionASTVisitor(self.cond.__globals__)
        bool_tree = self.visitor.visit(fundef)
        print(bool_tree)
        return bool_tree
