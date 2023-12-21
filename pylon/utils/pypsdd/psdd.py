import math
import random
import heapq


from .sdd import SddNode,NormalizedSddNode
from .data import DataSet,Inst,InstMap,WeightedInstMap
from .sdd import SddEnumerator,SddTerminalEnumerator
from .prior import Prior

class PSddNode(NormalizedSddNode):
    """Probabilistic Sentential Decision Diagram (PSDD)

    See https://github.com/hahaXD/psdd for PSDD multiply."""

    _brute_force_limit = 10

########################################
# CONSTRUCTOR + BASIC FUNCTIONS
########################################

    def __init__(self,node_type,alpha,vtree,manager):
        """Constructor

        node_type is FALSE, TRUE, LITERAL or DECOMPOSITION
        alpha is a literal if node_type is LITERAL
        alpha is a list of elements if node_type is DECOMPOSITION
        vtree is the vtree that the node is normalized for
        manager is the PSDD manager"""
        NormalizedSddNode.__init__(self,node_type,alpha,vtree,manager)

    def as_table(self,global_var_count=0):
        """Returns a string representing the full-joint probability
        distribution, e.g., a table over two variables would yield:

        AB   Pr
        ---------
        00 0.1000
        01 0.2000
        10 0.3000
        11 0.4000

        Maximum 10 (global) variables.

        If global_var_count is set to the global PSDD var count, then
        print sub-nodes will try to use global variable names.  Otherwise,
        the variable indices start with A.

        If positive is set to True, then only positive rows are printed."""
        var_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        var_count = max(self.vtree.var_count,global_var_count)
        assert var_count <= PSddNode._brute_force_limit

        st = [ var_names[:var_count] + "   Pr  " ]
        st.append( ("-" * var_count) + "+------" )
        for model in self.models(self.vtree,lexical=True):
            pr = self.pr_model(model)
            if global_var_count:
                model[global_var_count] = model[global_var_count] # hack
                st.append( "%s %.4f" % (model,pr) )
            else:
                st.append( "%s %.4f" % (model.shrink(),pr) )
        return "\n".join(st)

########################################
# STATS
########################################

    def theta_count(self):
        """Counts the number of free parameters in a PSDD.

        Only 'live' nodes are considered. i.e., we do not count
        (sub)-nodes of primes with false subs."""
        count = 0
        for node in self.positive_iter():
            if node.is_literal(): # or node.is_false_sdd
                pass
            elif node.is_true():
                count += 1
            else: # node.is_decomposition()
                count += len(node.positive_elements) - 1
        return count

    def zero_count(self):
        """Counts the number of (live) zero parameters in a PSDD"""
        count = 0
        for node in self.positive_iter():
            if node.is_literal(): # or node.is_false_sdd
                val = int(node.literal > 0)
                count += node.theta[val] == 0
            elif node.is_true():
                count += node.theta[0] == 0
                count += node.theta[1] == 0
            else: # node.is_decomposition()
                for element in node.positive_elements:
                    count += node.theta[element] == 0
        return count

    def true_count(self):
        """Counts the number of (live) true nodes in a PSDD"""
        return sum( 1 for n in self.positive_iter() if n.is_true() )

    def vtree_counts(self,manager):
        """Counts the number of nodes for each vtree node, indexed by
        vtree.id"""
        counts = [0]*(2*manager.var_count - 1)
        for node in self.positive_iter():
            counts[node.vtree.id] += 1
        return counts

########################################
# INFERENCE
########################################

    def pr_model(self,inst):
        """Returns Pr(inst) for a complete instantiation inst (where inst is
        an Inst or InstMap).

        Performs recursive test, which can be faster than linear
        traversal as in PSddNode.value."""
        self.is_model_marker(inst,clear_bits=False,clear_data=False)

        if self.data is None:
            pr = 0.0
        else:
            pr = 1.0
            queue = [self] if self.data is not None else []
            while queue:
                node = queue.pop()
                assert node.data is not None
                pr *= node.theta[node.data]/node.theta_sum
                if node.is_decomposition():
                    queue.append(node.data[0]) # prime
                    queue.append(node.data[1]) # sub

        self.clear_bits(clear_data=True)
        return pr

    def value(self,evidence=InstMap(),clear_data=True):
        """Compute the (un-normalized) value of a PSDD given evidence"""
        if self.is_false_sdd: return 0.0
        for node in self.as_positive_list(clear_data=clear_data):
            if node.is_false():
                value = 0.0
            elif node.is_true():
                if node.vtree.var in evidence:
                    val = evidence[node.vtree.var]
                    value = node.theta[val]
                else:
                    value = node.theta_sum
            elif node.is_literal():
                sim = evidence.is_compatible(node.literal)
                value = node.theta_sum if sim else 0.0
            else: # node.is_decomposition()            
                value = 0.0
                for p,s in node.positive_elements:
                    theta = node.theta[(p,s)]
                    value += (p.data/p.theta_sum)*(s.data/s.theta_sum)*theta
            node.data = value

        return value

    def probability(self,evidence=InstMap(),clear_data=True):
        """Compute the probability of evidence in a PSDD"""
        value = self.value(evidence=evidence,clear_data=clear_data)
        return value/self.theta_sum

    def marginals(self,evidence=InstMap(),clear_data=True,do_bottom_up=True):
        """Evaluate a PSDD top-down for its marginals.

        Returns a list var_marginals where:
        = var_marginals[lit] = value(lit,e)
        = var_marginals[0]   = value(e)

        Populates a field on each node:
        = node.pr_context has probability of context
        = node.pr_node has probability of node"""
        var_marginals = [ 0.0 ] * (2*self.vtree.var_count+1)
        if self.is_false_sdd: return var_marginals

        if do_bottom_up: # do not call value if done already
            self.value(evidence=evidence,clear_data=False)
        for node in self.as_positive_list(clear_data=False): # init field
            node.pr_context = 0.0

        value = self.data
        self.pr_context = 1.0
        for node in self.as_positive_list(reverse=True,clear_data=clear_data):
            if node.is_true() or node.is_literal():
                # accumulate variable marginals
                var = node.vtree.var
                pr_pos = node.theta[1]/node.theta_sum
                pr_neg = node.theta[0]/node.theta_sum
                if var in evidence:
                    val = evidence[var]
                    if val: var_marginals[ var] += pr_pos*node.pr_context
                    else:   var_marginals[-var] += pr_neg*node.pr_context
                else:
                    var_marginals[ var] += pr_pos*node.pr_context
                    var_marginals[-var] += pr_neg*node.pr_context
            else: # node.is_decomposition()
                # accumulate node marginals
                for p,s in node.positive_elements:
                    theta = node.theta[(p,s)]/node.theta_sum
                    pr_p = p.data/p.theta_sum
                    pr_s = s.data/s.theta_sum
                    p.pr_context += theta*pr_s*node.pr_context
                    s.pr_context += theta*pr_p*node.pr_context
            node.pr_node = node.pr_context*(node.data/node.theta_sum)

        var_marginals[0] = value
        return var_marginals

    def mpe(self,evidence=InstMap()):
        """Compute the most probable explanation (MPE) given evidence.
        Returns (un-normalized) MPE value and instantiation.

        If evidence is inconsistent with the PSDD, will return arbitrary
        instanatiation consistent with evidence."""
        if self.is_false_sdd:
            inst = InstMap.from_bitset(0,self.vtree.var_count).concat(evidence)
            return 0.0,inst
        for node in self.as_positive_list(clear_data=False):
            if node.is_false():
                var = node.vtree.var
                mpe_val = 0.0
                mpe_ind = evidence[var] if var in evidence else 0 # arbitrary index
            elif node.is_false_sdd:
                mpe_val = 0.0
                mpe_ind = node.elements[0] # arbitrary index
            elif node.is_true() or node.is_literal():
                var,theta = node.vtree.var,list(node.theta)
                if var in evidence:
                    mpe_val = theta[evidence[var]]
                    mpe_ind = evidence[var]
                else:
                    mpe_val = max(theta)
                    mpe_ind = theta.index(mpe_val)
            else: # node.is_decomposition()
                pels = node.positive_elements
                pvals = [ p.data[0]/p.theta_sum for p,s in pels ]
                svals = [ s.data[0]/s.theta_sum for p,s in pels ]
                vals = [ pval*sval*node.theta[el] for pval,sval,el \
                         in zip(pvals,svals,pels) ]
                mpe_val,mpe_ind = max(list(zip(vals,pels)))
            node.data = (mpe_val,mpe_ind)

        mpe_inst = InstMap()
        queue = [self] if self.data is not None else []
        while queue:
            node = queue.pop()
            if node.is_decomposition():
                prime,sub = node.data[1]
                queue.append(prime)
                queue.append(sub)
            else:
                mpe_inst[node.vtree.var] = node.data[1]

        # clear_data
        for node in self._positive_array: node.data = None
        return mpe_val,mpe_inst

    def enumerate_mpe(self,pmanager,evidence=InstMap()):
        """Enumerate the top-k MPE's of a PSDD
        AC: TODO evidence."""

        enum = PSddEnumerator(self.vtree)
        return enum.enumerator(self)

########################################
# KL-DIVERGENCE
########################################

    @staticmethod
    def kl(pr1,pr2):
        """Compute KL-divergence between two (list) distributions pr1 and pr2"""
        kl = 0.0
        for p1,p2 in zip(pr1,pr2):
            if p1 == 0.0: continue
            kl += p1 * (math.log(p1) - math.log(p2))
        if kl < 0.0: kl = 0.0
        return kl

    def kl_psdd_brute_force(self,other):
        """Brute-force (enumerative) computation of KL-divergence between two
        PSDDs"""
        assert self.vtree.var_count <= PSddNode._brute_force_limit
        if self.vtree.var_count != other.vtree.var_count:
            raise ValueError("PSDDs have different # of variables")
        kl = 0.0
        for model in self.models(self.vtree):
            pr1 = self.pr_model(model)
            if pr1 == 0.0: continue
            pr2 = other.pr_model(model)
            kl += pr1 * (math.log(pr1) - math.log(pr2))
        return kl

    def kl_psdd(self,other):
        """Compute KL-divergence between two PSDDs, recursively.  The PSDDs
        must have the same structure, but may have different parameters."""
        if self.is_false_sdd: return 0.0
        for n1,n2 in zip(self.as_positive_list(),other.as_positive_list()):
            assert n1.id == n2.id
            if n1.is_false_sdd:
                kl = 0.0
            elif n1.vtree.is_leaf():
                pr1 = [ p/n1.theta_sum for p in n1.theta ]
                pr2 = [ p/n2.theta_sum for p in n2.theta ]
                kl = PSddNode.kl(pr1,pr2)
            else: # decomposition
                pels1,pels2 = n1.positive_elements,n2.positive_elements
                pr1 = [ n1.theta[el]/n1.theta_sum for el in pels1 ]
                pr2 = [ n2.theta[el]/n2.theta_sum for el in pels2 ]
                kl = sum( p1*(p.data+s.data) for p1,(p,s) in zip(pr1,pels1) )
                kl += PSddNode.kl(pr1,pr2)
            n1.data = kl
        return kl

    def kl_psdd_alt(self,other):
        """Alternative computation of the KL-divergence between two PSDDs.
        The PSDDs must have the same structure, but may have different
        parameters.  This one uses node marginals to compute the KL."""
        self.marginals()
        kl = 0.0
        for n1,n2 in zip(self.as_positive_list(),other.as_positive_list()):
            assert n1.id == n2.id
            if n1.is_false_sdd or n1.pr_node == 0.0:
                continue
            elif n1.vtree.is_leaf():
                pr1 = [ p/n1.theta_sum for p in n1.theta ]
                pr2 = [ p/n2.theta_sum for p in n2.theta ]
                kl += n1.pr_node * PSddNode.kl(pr1,pr2)
            else: # decomposition
                pr1 = [ n1.theta[el]/n1.theta_sum for el in n1.positive_elements ]
                pr2 = [ n2.theta[el]/n2.theta_sum for el in n2.positive_elements ]
                kl += n1.pr_node * PSddNode.kl(pr1,pr2)
        return kl

########################################
# SIMULATE A PSDD
########################################

    @staticmethod
    def sample(pr,z=1.0):
        """If input is a list of tuples (item,probability), randomly return an
        item based according to their probability"""
        q = random.random()
        cur = 0.0
        for item,p in pr:
            cur += p/z
            if q <= cur:
                return item
        return item

    def simulate(self,inst=None,seed=None):
        """Draw a model from the distribution induced by the PSDD"""
        assert not self.is_false()
        if seed is not None: random.seed(seed)
        if inst is None: inst = InstMap()

        if self.is_true():
            p = self.theta[0]/self.theta_sum
            val = 0 if random.random() < p else 1
            inst[self.vtree.var] = val
        elif self.is_literal():
            val = 0 if self.literal < 0 else 1
            inst[self.vtree.var] = val
        else:
            pr = iter(self.theta.items())
            p,s = PSddNode.sample(pr,z=self.theta_sum)
            p.simulate(inst=inst)
            s.simulate(inst=inst)

        return inst

########################################
# LEARN (COMPLETE DATA)
########################################

    def log_likelihood(self,data):
        """Computes the log likelihood

            log Pr(data | theta)
        """
        return sum( cnt*math.log(self.pr_model(inst)) for inst,cnt in data )

    def log_posterior(self,data,prior):
        """Computes the (unnormalized) log posterior:

            log Pr(theta | data) 
              = log Pr(data | theta) + log Pr(theta) - log Pr(data)

        but we leave out the - log Pr(data) term.
        """
        return self.log_likelihood(data) + prior.log_prior(self)

    def learn(self,data,prior,verbose=False):
        """Given a complete dataset and a prior, learn the parameters of a
        PSDD"""
        prior.initialize_psdd(self)
        n = len(data)
        for i,(inst,count) in enumerate(data):
            if verbose and (n-i-1)%max(1,(n/10)) == 0:
                print("%3.0f%% done" % (100.0*(i+1)/n))
            # mark satisfying sub-circuit
            self.is_model_marker(inst,clear_bits=False,clear_data=False)
            self._increment_follow_marker(float(count))
            self.clear_bits(clear_data=True)

    def _increment_follow_marker(self,count):
        """Increment the PSDD parameters by following sub-circuit markers"""
        assert self.data is not None
        queue = [self]
        while queue:
            node = queue.pop()
            node.theta[node.data] += count
            node.theta_sum += count
            if node.is_decomposition():
                queue.append(node.data[0]) # prime
                queue.append(node.data[1]) # sub

########################################
# SUB-CIRCUIT
########################################

class SubCircuit:
    """Sub-Circuit's of PSDD models"""

    def __init__(self,node,element,left,right):
        self.node = node
        self.element = element
        self.left = left
        self.right = right

    def __repr__(self):
        pr = self.node.theta[self.element]/self.node.theta_sum
        vt_id = self.node.vtree.id
        node_id = self.node.id
        if self.node.is_decomposition():
            p_id = self.element[0].id
            s_id = self.element[1].id
            return "vt_%d: %d (%d/%d) %.4f" % (vt_id,node_id,p_id,s_id,pr)
        else:
            return "vt_%d: %d (%d) %.4f" % (vt_id,node_id,self.element,pr)

    def print_subcircuit(self):
        print(self)
        if self.node.is_decomposition():
            self.left.print_subcircuit()
            self.right.print_subcircuit()

    @staticmethod
    def induce_sub_circuit(inst,node):
        node.is_model_marker(inst,clear_bits=False,clear_data=False)
        subcircuit = SubCircuit._induce_sub_circuit(node)
        node.clear_bits(clear_data=True)
        return subcircuit

    @staticmethod
    def _induce_sub_circuit(node):
        left,right = None,None
        element = node.data
        if element is not None:
            if node.is_decomposition():
                prime,sub = element
                left = SubCircuit._induce_sub_circuit(prime)
                right = SubCircuit._induce_sub_circuit(sub)
        return SubCircuit(node,element,left,right)

    def probability(self):
        pr = self.node.theta[self.element]/self.node.theta_sum
        if self.left is not None:
            pr *= self.left.probability()
            pr *= self.right.probability()
        self.pr = pr
        return pr

    def node_of_vtree(self,vtree):
        my_id = self.node.vtree.id
        target_id = vtree.id
        if my_id == target_id:
            return self
        elif target_id < my_id:
            return self.left.node_of_vtree(vtree)
        else: # elif my_id > target_id:
            return self.right.node_of_vtree(vtree)

########################################
# k-BEST ENUMERATION
########################################

class PSddEnumerator(SddEnumerator):
    """Manager for k-best MPE enumeration."""

    @staticmethod
    def _element_update(element_enum,inst):
        """This is invoked after inst.concat(other)"""
        element = (element_enum.prime,element_enum.sub)
        parent = element_enum.parent
        theta = parent.theta[element]/parent.theta_sum
        inst.mult_weight(theta)

    def __init__(self,vtree):
        SddEnumerator.__init__(self,vtree)
        self.terminal_enumerator = PSddTerminalEnumerator

class PSddTerminalEnumerator(SddTerminalEnumerator):
    """Enumerator for terminal PSDD nodes"""

    def __init__(self,node,vtree):
        self.heap = []

        if node.is_false():
            pass
        elif node.is_literal():
            #weight = node.theta[node.literal > 0]/node.theta_sum
            inst = WeightedInstMap.from_literal(node.literal,weight=1.0)
            heapq.heappush(self.heap,inst)
        if node.is_true():
            weight = node.theta[0]/node.theta_sum
            inst = WeightedInstMap.from_literal(-vtree.var,weight=weight)
            heapq.heappush(self.heap,inst)

            weight = node.theta[1]/node.theta_sum
            inst = WeightedInstMap.from_literal(vtree.var,weight=weight)
            heapq.heappush(self.heap,inst)
