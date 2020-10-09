import torch


class TreeNode:
    def __init__(self, name, children):
        self.name = name
        self.children = children

    def __str__(self):
        arg_str = "(" + ",".join([str(c) for c in self.children]) + ")" if len(self.children) > 0 else ""
        return self.name + arg_str

    def as_bool(self):
        return self

    def __eq__(self, obj):
        return (type(obj) == type(self) and
                self.name == obj.name and self.children == obj.children)


class TreeNodeVisitor:

    def visit(self, node, args):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, args)

    def generic_visit(self, node, args):
        """Called if no explicit visitor function exists for a node."""
        print(self)
        raise NotImplementedError


class BinaryOp(TreeNode):
    def __init__(self, name, left, right):
        self.left = left
        self.right = right
        super().__init__(name, [left, right])


class And(BinaryOp):
    def __init__(self, left, right):
        super().__init__("And", left, right)


class Or(BinaryOp):
    def __init__(self, left, right):
        super().__init__("Or", left, right)


class Implication(BinaryOp):
    def __init__(self, left, right):
        super().__init__("Implication", left, right)


class UnaryOp(TreeNode):
    def __init__(self, name, operand):
        self.operand = operand
        super().__init__(name, [operand])


class Not(UnaryOp):
    def __init__(self, operand):
        super().__init__("Not", operand)


class IsEq(BinaryOp):
    def __init__(self, left, right):
        super().__init__('Eq', left, right)


class Const(TreeNode):
    def __init__(self, value):
        self.value = value
        self.is_bool = isinstance(value, bool)
        super().__init__(str(value), [])

    def as_bool(self):
        return self if self.is_bool else Const(bool(self.value))


class Forall(TreeNode):
    def __init__(self, expr):
        self.expr = expr
        super().__init__("ForAll", [expr])


class Exists(TreeNode):
    def __init__(self, expr):
        self.expr = expr
        super().__init__("Exists", [expr])


class List(TreeNode):
    def __init__(self, elts):
        self.elts = elts
        super().__init__("List", elts)


class Arg(TreeNode):
    '''Use one of the arguments of the constraint, such as "x".'''

    def __init__(self, name, pos):
        self.arg_pos = pos
        self.arg_name = name
        super().__init__(self.arg_name, [])

    def probs(self, probs):
        return probs[self.arg_pos]


class VarList(TreeNode):
    '''Use a subscript to select elements, such as "x[0]".'''

    def __init__(self, arg, indices):
        self.arg = arg
        self.indices = indices
        # super().__init__(str(arg) + '[' + ",".join([str(i) for i in self.indices]) + "]", [])
        super().__init__(str(arg) + '[' + str(self.indices) + "]", [])

    def probs(self, probs):
        return probs[self.arg.arg_pos][self.indices, :]


class Slice(TreeNode):
    def __init__(self, lower, step, upper):
        self.lower = lower
        self.step = step
        self.upper = upper
        super().__init__(f"{lower if lower is not None else ''}:"
                         f"{upper if upper is not None else ''}:"
                         f"{step if step is not None else ''}", [])


class ExtSlice(TreeNode):
    def __init__(self, slices):
        self.slices = slices
        super().__init__(", ".join(map(str, slices)), [])


class VarCond(TreeNode):
    '''Uses a boolean expression to select elements, such as "y[x==2]"'''

    def __init__(self, arg, expr):
        self.arg = arg
        self.expr = expr
        super().__init__(str(arg) + '[' + str(self.expr) + "]", [])


class IdentifierDef(TreeNode):
    def __init__(self, id, definition):
        self.id = id
        self.definition = definition
        super().__init__('{def ' + self.id + '=' + str(self.definition) + "}", [])


class IdentifierRef(TreeNode):
    def __init__(self, iddef):
        self.iddef = iddef
        super().__init__('{'+self.iddef.id+'}', [])


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

    def sdd(self, mgr):
        return self.return_node.sdd(mgr)
