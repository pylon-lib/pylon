import ast
from functools import reduce
import torch


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


class TreeNode:
    def __init__(self, name, children):
        self.name = name
        self.children = children

    def __str__(self):
        arg_str = "(" + ",".join([str(c) for c in self.children]) + ")" if len(self.children) > 0 else ""
        return self.name + arg_str

    def as_bool(self):
        return self

    def prod_tnorm(self, probs):
        print(self)
        raise NotImplementedError

    def lukasiewicz_tnorm(self, probs):
        print(self)
        raise NotImplementedError

    def godel_tnorm(self, probs):
        print(self)
        raise NotImplementedError

    def sdd(self, mgr):
        print(self)
        raise NotImplementedError

    def __eq__(self, obj):
        return (type(obj) == type(self) and
                self.name == obj.name and self.children == obj.children)


class BinaryOp(TreeNode):
    def __init__(self, name, left, right):
        self.left = left
        self.right = right
        super().__init__(name, [left, right])


class And(BinaryOp):
    def __init__(self, left, right):
        super().__init__("And", left, right)

    def prod_tnorm(self, probs):
        lv = self.left.prod_tnorm(probs)
        rv = self.right.prod_tnorm(probs)
        return lv * rv

    def lukasiewicz_tnorm(self, probs):
        lv = self.left.lukasiewicz_tnorm(probs)
        rv = self.right.lukasiewicz_tnorm(probs)
        return torch.relu(lv + rv - 1)

    def godel_tnorm(self, probs):
        lv = self.left.lukasiewicz_tnorm(probs)
        rv = self.right.lukasiewicz_tnorm(probs)
        return torch.min(lv, rv)

    def sdd(self, mgr):
        l = self.left.sdd()
        r = self.right.sdd()
        return l & r


class Or(BinaryOp):
    def __init__(self, left, right):
        super().__init__("Or", left, right)

    def prod_tnorm(self, probs):
        lv = self.left.prod_tnorm(probs)
        rv = self.right.prod_tnorm(probs)
        return lv + rv - lv * rv

    def lukasiewicz_tnorm(self, probs):
        lv = self.left.lukasiewicz_tnorm(probs)
        rv = self.right.lukasiewicz_tnorm(probs)
        # min(1,x) = 1-max(0,1-x)
        return 1 - torch.relu(1 - lv - rv)

    def godel_tnorm(self, probs):
        lv = self.left.lukasiewicz_tnorm(probs)
        rv = self.right.lukasiewicz_tnorm(probs)
        return torch.max(lv, rv)

    def sdd(self, mgr):
        l = self.left.sdd()
        r = self.right.sdd()
        return l | r


class UnaryOp(TreeNode):
    def __init__(self, name, operand):
        self.operand = operand
        super().__init__(name, [operand])


class Not(UnaryOp):
    def __init__(self, operand):
        super().__init__("Not", operand)

    def prod_tnorm(self, probs):
        return 1.0 - self.operand.prod_tnorm(probs)

    def lukasiewicz_tnorm(self, probs):
        return 1.0 - self.operand.lukasiewicz_tnorm(probs)

    def godel_tnorm(self, probs):
        return 1.0 - self.operand.godel_tnorm(probs)

    def sdd(self, mgr):
        return ~self.operand.sdd()


class IsEq(BinaryOp):
    def __init__(self, left, right):
        super().__init__('Eq', left, right)

    def prod_tnorm(self, probs):
        if isinstance(self.left, VarUse) and isinstance(self.right, Const):
            return self.left.probs(probs)[self.right.value]
        elif isinstance(self.left, Const) and isinstance(self.right, VarUse):
            return self.right.probs(probs)[self.left.value]
        elif isinstance(self.left, Const) and isinstance(self.right, Const):
            return 1.0 if self.left.value == self.right.value else 0.0
        elif isinstance(self.left, VarUse) and isinstance(self.right, VarUse):
            return (self.left.probs(probs)*self.right.probs(probs)).sum()
        else:
            raise NotImplementedError

    def lukasiewicz_tnorm(self, probs):
        return self.prod_tnorm(probs)

    def godel_tnorm(self, probs):
        return self.prod_tnorm(probs)

    def sdd(self, mgr):
        return self.left.sdd(mgr).equiv(self.right.sdd(mgr))


class Const(TreeNode):
    def __init__(self, value):
        self.value = value
        self.is_bool = isinstance(value, bool)
        super().__init__(str(value), [])

    def as_bool(self):
        return self if self.is_bool else Const(bool(self.value))

    def prod_tnorm(self, probs):
        return 1.0 if self.value else 0.0

    def lukasiewicz_tnorm(self, probs):
        return 1.0 if self.value else 0.0

    def godel_tnorm(self, probs):
        return 1.0 if self.value else 0.0

    def sdd(self, mgr):
        if self.value == True or self.value == 1:
            return mgr.true()
        elif self.value == False or self.value == 0:
            return mgr.false()
        else:
            raise NotImplementedError


class VarUse(TreeNode):
    def __init__(self, varidx, varname, index):
        self.varidx = varidx
        self.varname = varname
        self.index = index
        super().__init__(self.varname + '[' + str(self.index) + "]", [])

    def probs(self, probs):
        return probs[self.varidx][self.index]

    def as_bool(self):
        return Not(IsEq(self, Const(0)))

    def sdd(self, mgr):
        if self.varidx != 0:
            raise NotImplementedError
        else:
            while mgr.var_count() < self.index+1:
                mgr.add_var_after_last()
            return mgr.literal(self.index+1)


class IdentifierDef(TreeNode):
    def __init__(self, id, definition):
        self.id = id
        self.definition = definition
        super().__init__('{def ' + self.id + '=' + str(self.definition) + "}", [])


class IdentifierRef(TreeNode):
    def __init__(self, iddef):
        self.iddef = iddef
        super().__init__('{'+self.iddef.id+'}', [])

    def prod_tnorm(self, probs):
        return self.iddef.definition.prod_tnorm(probs)

    def lukasiewicz_tnorm(self, probs):
        return self.iddef.definition.lukasiewicz_tnorm(probs)

    def godel_tnorm(self, probs):
        return self.iddef.definition.godel_tnorm(probs)


class FunDef(TreeNode):
    def __init__(self, arg_pos, iddefs, return_node):
        self.arg_pos = arg_pos
        self.iddefs = iddefs
        self.return_node = return_node
        super().__init__(
            'lambda {0}: {1}; return {2}'.format(
                ','.join(self.arg_pos.keys()),
                ';'.join([str(v) for v in self.iddefs.values()]),
                self.return_node), [])

    def prod_tnorm(self, probs):
        return self.return_node.prod_tnorm(probs)

    def lukasiewicz_tnorm(self, probs):
        return self.return_node.lukasiewicz_tnorm(probs)

    def godel_tnorm(self, probs):
        return self.return_node.godel_tnorm(probs)

    def sdd(self, mgr):
        return self.return_node.sdd(mgr)


class LogicExpressionVisitor(ast.NodeVisitor):

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
            ast.NotEq: (lambda left, right: Not(IsEq(left, right)))
        }
        assert(len(node.ops))
        op_func = supported[type(node.ops[0])]
        ltree = self.visit(node.left)
        assert(len(node.comparators))
        rtree = self.visit(node.comparators[0])
        return op_func(ltree, rtree)
