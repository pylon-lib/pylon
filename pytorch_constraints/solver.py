import itertools

import ast
import inspect

from .ast_visitor import FunDefFindingVisitor
from .ast_visitor import LogicExpressionVisitor


class Solver:
    '''Computes the loss for the given constraint.'''

    def set_cond(self, cond):
        '''Sets the boolean condition that defines the constraint, performs any fixed computations, as needed.'''
        self.cond = cond

    def loss(self, logits):
        '''Return the loss for the constraint for the given logits of the variables involved in the constraint.'''
        raise NotImplementedError


class ASTSolver(Solver):
    '''Computes the loss by looking at code structure.'''

    def set_cond(self, cond):
        super().set_cond(cond)
        astree = ast.parse(inspect.getsource(cond))
        self.visit_ast(
            self.find_function_def(astree))

    def find_function_def(self, astree):
        '''Find the appropriate node in the AST.'''
        visitor = FunDefFindingVisitor()
        return visitor.visit(astree)

    def visit_ast(self, astree):
        '''Visit the AST once, to perform any computations.'''
        raise NotImplementedError


class ASTLogicSolver(ASTSolver):

    def __init__(self):
        self.bool_tree = None
        self.visitor = LogicExpressionVisitor()

    def visit_ast(self, astree):
        self.bool_tree = self.visitor.visit(astree)
        print(self.bool_tree)
