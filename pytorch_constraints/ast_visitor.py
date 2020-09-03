import ast
from functools import reduce
import torch
from .tree_node import *


class FunDefFindingVisitor(ast.NodeVisitor):
    def __init__(self):
        super(ast.NodeVisitor).__init__()

    def visit_Module(self, node):
        return self.visit(node.body[0])

    def visit_Assign(self, node):
        return self.visit(node.value)

    def visit_Call(self, node):
        assert(node.func.id == 'constraint')
        return self.visit(node.args[0])

    def visit_Lambda(self, node):
        return node

    def visit_FunctionDef(self, node):
        return node


class LogicExpressionASTVisitor(ast.NodeVisitor):

    def __init__(self):
        self.arg_pos = {}
        self.iddefs = {}
        super(ast.NodeVisitor).__init__()

    def generic_visit(self, node):
        print(ast.dump(node))
        raise NotImplementedError

    def get_arg_pos(self, node):
        arg_pos = {}
        for i, arg in enumerate(node.args.args):
            arg_pos[arg.arg] = i
        return arg_pos

    def visit_FunctionDef(self, node):
        self.arg_pos = self.get_arg_pos(node)
        for b in node.body:
            body_tree = self.visit(b)
        return FunDef(self.arg_pos, self.iddefs, body_tree.as_bool())

    def visit_Lambda(self, node):
        self.arg_pos = self.get_arg_pos(node)
        body_tree = self.visit(node.body)
        return FunDef(self.arg_pos, self.iddefs, body_tree.as_bool())

    def visit_Return(self, node):
        return self.visit(node.value).as_bool()

    def visit_UnaryOp(self, node):
        supported = {
            ast.Not: (lambda opr: Not(opr.as_bool()))
        }
        op_func = supported[type(node.op)]
        opr = self.visit(node.operand)
        return op_func(opr)

    def visit_Call(self, node):
        def attribute_calls(n):
            if n.func.attr == 'implies':
                assert(len(n.args) == 1)
                op_func = lambda x, y: Implication(x.as_bool(), y.as_bool())
                return op_func(self.visit(n.func.value), self.visit(n.args[0]))
            elif n.func.attr == 's_implies':
                assert(len(n.args) <= 2)
                op_func = lambda x, y, s: SigmoidalImplication(x.as_bool(), y.as_bool(), s)
                s = ast.Constant(1.0)

                if len(n.keywords) != 0: # to support func call like x.s_implies(y, s=1.0)
                    s = next(filter(lambda x: x.arg == 's', n.keywords)).value
                elif len(n.args) == 2:   # to support func call like x.s_implies(y, 1.0)
                    s = n.args[1]

                assert(isinstance(s, ast.Constant))
                return op_func(self.visit(n.func.value), self.visit(n.args[0]), s)
            else:
                raise Exception('unsupported function call', n.func.attr)


            supported_attr = {
                'implies': lambda x, y: Implication(x.as_bool(), y.as_bool()), # this is the same as x <= y, i.e. the default implication rule
                's_implies': lambda x, y, s: SigmoidalImplication(x.as_bool(), y.as_bool(), s)
            }

        supported_func = {
            ast.Attribute: attribute_calls
        }

        func = supported_func[type(node.func)]
        return func(node)

    def visit_Subscript(self, node):
        # TODO check node.value is the variable?
        varidx = self.arg_pos[node.value.id]
        return VarUse(varidx, node.value.id, node.slice.value.n)

    def visit_Assign(self, node):
        assert len(node.targets) == 1
        id = node.targets[0].id
        definition = self.visit(node.value)
        iddef = IdentifierDef(id, definition)
        assert id not in self.iddefs
        self.iddefs[id] = iddef
        return iddef

    def visit_Name(self, node):
        # assumes unscripted reference is to local variable
        iddef = self.iddefs[node.id]
        return IdentifierRef(iddef)

    def visit_NameConstant(self, node):
        #deprecated in 3.8
        return Const(node.value)

    def visit_Num(self, node):
        #deprecated in 3.8
        return Const(node.n)

    def visit_Constant(self, node):
        return Const(node.value)

    def visit_BoolOp(self, node):
        supported = {
            ast.And: (lambda left, right: And(left.as_bool(), right.as_bool())),
            ast.Or: (lambda left, right: Or(left.as_bool(), right.as_bool()))
        }
        op_func = supported[type(node.op)]
        trees = map(self.visit, node.values)
        return reduce(op_func, trees)

    def visit_Compare(self, node):
        supported = {
            ast.Eq: (lambda left, right: IsEq(left, right)),
            ast.NotEq: (lambda left, right: Not(IsEq(left, right))),
            ast.LtE:  (lambda left, right: Implication(left.as_bool(), right.as_bool()))   # implication/residuum
        }
        assert(len(node.ops))
        op_func = supported[type(node.ops[0])]
        ltree = self.visit(node.left)
        assert(len(node.comparators))
        rtree = self.visit(node.comparators[0])
        return op_func(ltree, rtree)
