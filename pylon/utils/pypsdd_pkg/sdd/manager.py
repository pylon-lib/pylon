#!/usr/bin/env python

from .sdd import SddNode
from .psdd import PSddNode
import functools

def cmp(x, y):
    return (x > y) - (x < y)

class SddManager:
    """SDD Manager"""

    Node = SddNode # native node class

    def __init__(self,vtree):
        """Constructor

        Initialize with vtree"""

        self.vtree = vtree
        self.var_count = vtree.var_count
        self.unique = {}
        self.id_counter = 0

        self._setup_var_to_vtree(vtree)
        self._setup_terminal_sdds()

    def new_id(self):
        index = self.id_counter
        self.id_counter += 1
        return index

    def _setup_var_to_vtree(self,vtree):
        self.var_to_vtree = [None]*(self.var_count+1)
        for node in vtree:
            if node.is_leaf():
                self.var_to_vtree[node.var] = node

    def _setup_terminal_sdds(self):
        """Create FALSE, TRUE and LITERAL SDDs"""
        self.false = self.Node(SddNode.FALSE,None,None,self)
        self.true = self.Node(SddNode.TRUE,None,None,self)

        lit_type = SddNode.LITERAL
        self.literals = [None]*(2*self.var_count+1)
        for var in range(1,self.var_count+1):
            vtree = self.var_to_vtree[var]
            self.literals[var] = self.Node(lit_type,var,vtree,self)
            self.literals[-var] = self.Node(lit_type,-var,vtree,self)

    def _canonical_elements(self,elements):
        """Given a list of elements, canonicalize them"""
        cmpf = lambda x,y: cmp(x[0].id,y[0].id)
        elf = lambda x: tuple(sorted(x,key=functools.cmp_to_key(cmpf)))
        return elf(elements)

    def lookup_node(self,elements,vtree_node):
        """Unique table lookup for DECOMPOSITION nodes.

        Elements is a list of prime,sub pairs:
            [ (p1,s1),(p2,s2),...,(pn,sn) ]"""
        elements = self._canonical_elements(elements)
        if elements not in self.unique:
            node_type = SddNode.DECOMPOSITION
            node = self.Node(node_type,elements,vtree_node,self)
            self.unique[elements] = node
        return self.unique[elements]

class PSddManager(SddManager):
    Node = PSddNode # native node class

    def __init__(self,vtree):
        SddManager.__init__(self,vtree)
        self._setup_true_false_sdds()

    def _setup_true_false_sdds(self):
        """PSDDs are normalized, so we create a unique true/false SDD/PSDD
        node for each vtree node."""
        node_count = 2*self.var_count - 1
        self.true_sdds = [None]*node_count
        self.false_sdds = [None]*node_count

        for vtree_node in self.vtree.post_order():
            # setup true SDDs
            if vtree_node.is_leaf():
                node_type = SddNode.TRUE
                true_node = self.Node(node_type,None,vtree_node,self)
            else:
                left_true  = self.true_sdds[vtree_node.left.id]
                right_true = self.true_sdds[vtree_node.right.id]
                elements = [(left_true,right_true)]
                true_node = self.lookup_node(elements,vtree_node)
            self.true_sdds[vtree_node.id] = true_node

            # setup false SDDs
            if vtree_node.is_leaf():
                node_type = SddNode.FALSE
                false_node = self.Node(node_type,None,vtree_node,self)
            else:
                left_true   = self.true_sdds[vtree_node.left.id]
                right_false = self.false_sdds[vtree_node.right.id]
                elements = [(left_true,right_false)]
                false_node = self.lookup_node(elements,vtree_node)
            self.false_sdds[vtree_node.id] = false_node
            false_node.is_false_sdd = True

            true_node.negation = false_node
            false_node.negation = true_node

    def negate(self,node,vtree):
        """Negate a normalized SDD node"""

        if   node.is_false(): return self.true_sdds[vtree.id]
        elif node.is_true(): return self.false_sdds[vtree.id]
        elif node.is_literal(): return self.literals[-node.literal]
        elif node.negation is not None: return node.negation
        else: # node.is_decomposition()
            right = node.vtree.right
            elements = [ (p,self.negate(s,right)) for p,s in node.elements ]
            neg = self.lookup_node(elements,vtree)
            neg.negation = node
            node.negation = neg
            return neg

    def copy_and_normalize_sdd(self,alpha,vtree):
        """Copy an SDD alpha from another manager to the self manager, and
        normalize it with respect to the given vtree."""

        for node in alpha.post_order(clear_data=True):
            if   node.is_false():   copy_node = self.false
            elif node.is_true():    copy_node = self.true
            elif node.is_literal(): copy_node = self.literals[node.literal]
            else: # node.is_decomposition()
                elements = []
                left,right = node.vtree.left,node.vtree.right
                for prime,sub in node.elements:
                    copy_prime = self._normalize_sdd(prime.data,left)
                    copy_sub   = self._normalize_sdd(sub.data,right)
                    elements.append( (copy_prime,copy_sub) )
                copy_node = self.lookup_node(elements,node.vtree)
            node.data = copy_node
        root_sdd = self._normalize_sdd(copy_node,vtree)
        return root_sdd

    def _normalize_sdd(self,alpha,vtree):
        """Normalize a given sdd for a given vtree"""
        if alpha.is_false(): return self.false_sdds[vtree.id]
        elif alpha.is_true(): return self.true_sdds[vtree.id]
        elif alpha.vtree is vtree: return alpha

        if alpha.vtree.id < vtree.id:
            left = self._normalize_sdd(alpha,vtree.left)
            right = self.true_sdds[vtree.right.id]
            neg_left = self.negate(left,vtree.left)
            false_right = self.false_sdds[vtree.right.id]
            elements = [ (left,right),(neg_left,false_right) ]
        elif alpha.vtree.id > vtree.id:
            left = self.true_sdds[vtree.left.id]
            right = self._normalize_sdd(alpha,vtree.right)
            elements = [ (left,right) ]
        return self.lookup_node(elements,vtree)
