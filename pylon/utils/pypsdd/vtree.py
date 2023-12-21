class Vtree:
    """Vtrees (variable trees)"""

    def __init__(self,left,right,var):
        """Use Vtree.leaf_node or Vtree.internal_node to create a new vtree node"""
        self.parent = None
        self.left = left
        self.right = right
        self.var = var
        self.id = None
        if self.var: # leaf
            self.var_count = 1
        else:
            self.var_count = left.var_count + right.var_count

    def __eq__(self, other): 
        try:
            for i, j in zip(list(self), list(other)):
                assert(i.is_leaf() == j.is_leaf())
                if i.is_leaf():
                    assert(i.id == j.id and i.var == j.var)
                else:
                    assert(i.id == j.id and i.left.id == j.left.id and i.right.id == j.right.id)
        except:
            return False
        return True

    @classmethod
    def leaf_node(cls,var):
        """Creates new leaf Vtree node with variable var"""
        return Vtree(None,None,var)

    @classmethod
    def internal_node(cls,left,right):
        """Creates new internal Vtree node with children left and right"""
        node = Vtree(left,right,None)
        left.parent = right.parent = node 
        return node

    def __iter__(self):
        """Generator over vtree nodes using in-order traversal"""
        if self.is_leaf():
            yield self
        else:
            for node in self.left:  yield node
            yield self
            for node in self.right: yield node

    def pre_order(self):
        """Generator over vtree nodes using pre-order traversal"""
        if self.is_leaf():
            yield self
        else:
            yield self
            for node in self.left.pre_order():  yield node
            for node in self.right.pre_order(): yield node

    def post_order(self):
        """Generator over vtree nodes using post-order traversal"""
        if self.is_leaf():
            yield self
        else:
            for node in self.left.post_order():  yield node
            for node in self.right.post_order(): yield node
            yield self

    def to_list(self):
        """Returns a list of vtree nodes, indexed by id"""
        node_count = 2*self.var_count - 1
        nodes = [None]*node_count
        for node in self:
            nodes[node.id] = node
        return nodes

    def is_leaf(self):
        """Returns true if the node is a leaf node, and false otherwise"""
        return self.left is None # and self.right is None

    def variables(self):
        variables = set()
        for vtree in self:
            if vtree.is_leaf():
                variables.add(vtree.var)
        return variables

    def var_to_vtree(self):
        """Returns dictionary mapping variable to its leaf vtree node"""
        v2v = dict()
        for node in self:
            if node.is_leaf():
                v2v[node.var] = node
        return v2v

    def last_node(self):
        """Returns the right-most leaf node in the vtree"""
        if self.is_leaf():
            return self
        else:
            return self.right.last_node()

    def height(self):
        """Returns the height of the vtree"""
        if self.is_leaf():
            return 0
        else:
            return 1 + max(self.left.height(),self.right.height())

    def is_sub_vtree_of(self,other):
        """Returns true if vtree is sub-vtree of other (inclusive), and false
        otherwise"""
        node = self
        while node is not None:
            if node == other: return True
            node = node.parent
        return False

    _vtree_file_format = \
        ("c ids of vtree nodes start at 0\n"
         "c ids of variables start at 1\n"
         "c vtree nodes appear bottom-up, children before parents\n"
         "c\n"
         "c file syntax:\n"
         "c vtree number-of-nodes-in-vtree\n"
         "c L id-of-leaf-vtree-node id-of-variable\n"
         "c I id-of-internal-vtree-node id-of-left-child id-of-right-child\n"
         "c\n")


    @staticmethod
    def read(filename):
        """Read vtree from file"""
        with open(filename,'r') as f:
            for line in f:
                node = None
                if line.startswith('c'): continue
                elif line.startswith('vtree'):
                    node_count = int(line[5:]) # skip "vtree"
                    nodes = [None]*node_count
                elif line.startswith('L'):
                    line = line[2:].split() # skip "L" and split
                    node_id,var = [int(x) for x in line]
                    node = Vtree.leaf_node(var)
                elif line.startswith('I'):
                    line = line[2:].split() # skip "I" and split
                    node_id,left_id,right_id = [int(x) for x in line]
                    left,right = nodes[left_id],nodes[right_id]
                    node = Vtree.internal_node(left,right)
                    left.parent = right.parent = node
                if node is not None:
                    node.id = node_id
                    nodes[node_id] = node

        root = nodes[node_id]
        # make sure id's are based on inorder traversal
        for node_id,node in enumerate(root):
            node.id = node_id

        return root

    def vtree_from_pysdd(vtree):
        """Copy a pysdd vtree"""
        nodes = [None] * (2 * vtree.var_count() - 1)
        for elem in post_order(vtree):
            node = None
            if elem.is_leaf():
                node_id = elem.position()
                node = Vtree.leaf_node(elem.var())
            else:
                node_id = elem.position()
                left_id = elem.left.position()
                right_id = elem.right.position()
                left, right = nodes[left_id], node[right_id]
                node = Vtree.internal_node(left, right)
                left.parent = right.parent = node
            if node is not None:
                node.id = node_id
                nodes[node_id] = node

        root = nodes[node_id]
        # make sure id's are based on inorder traversal
        for node_id,node in enumerate(root):
            node.id = node_id

        return root

    #def sdd_from_pysdd(alpha, manager):
    #    """Read an SDD from file"""
    #    vtree_nodes = manager.vtree.to_list()
    #    for elem in post_order(alpha):
    #        node = None
    #        node_count = int(line[3:])
    #        nodes = [None]*node_count
    #        if elem.is_false():
    #            node_id = alpha.id()
    #            node = manager.false
    #        elif elem.is_true():
    #            node_id = alpha.id()
    #            node = manager.true
    #        elif elem.is_literal():
    #            node_id = alpha.id
    #            vtree_id = alpha.vtree().position #TODO: Change
    #            lit = alpha.literal
    #            node = manager.literals[lit]
    #        elif elem.is_decision(): 
    #            node_id = elem.id
    #            vtree_id = elem.vtree().position
    #            size = len(elem.elements())
    #            elements = [nodes]
    #        elif line.startswith('D'):
    #            elements = [ nodes[my_id] for my_id in line[3:] ]
    #            elements = [ element for element in pairs(elements) ]
    #            vtree_node = vtree_nodes[vtree_id]
    #            node = manager.lookup_node(elements,vtree_node)
    #        if node:
    #            nodes[node_id] = node
    #    return node


    def save(self,filename):
        """Save vtree to file"""
        with open(filename,'w') as f:
            # write vtree file format
            f.write(Vtree._vtree_file_format)
            # write vtree header
            node_count = 2*self.var_count - 1
            f.write('vtree %d\n' % node_count)
            # write definition of each node, children before parents
            for n in self.post_order():
                f.write('%s\n' % n)

    def __repr__(self):
        if self.is_leaf():
            st = 'L %d %d' % (self.id,self.var)
        else:
            st = 'I %d %d %d' % (self.id,self.left.id,self.right.id)
        return st
