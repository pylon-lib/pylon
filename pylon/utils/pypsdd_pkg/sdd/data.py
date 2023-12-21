import math
import random

from collections import defaultdict

# AC: TODO: empty Inst?  Inst.from_list([],var_count)?

class DataSet:
    """Dataset.  Implements a Dict from object to count in dataset"""

    def __init__(self):
        self.data = defaultdict(lambda: 0)
        self.N = 0

    def __len__(self):
        """number of unique instances"""
        return len(self.data)

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if key in self.data:
            self.N -= self.data[key]
        self.data[key] = value
        self.N += value

    def __delitem__(self, key):
        self.N -= self.data[key]
        del self.data[key]

    def __iter__(self):
        return iter(self.data.items())

    def __repr__(self,limit=10):
        cmpf = lambda x,y:-cmp(x[1],y[1])
        items = sorted(list(self.data.items()),cmp=cmpf)
        fmt = " %%%dd %%s" % len(str(items[0][1]))
        st = [ fmt % (count,inst) for inst,count in items[:limit] ]
        if len(items) > limit: st.append(" ...")
        return "\n".join(st)

    @staticmethod
    def read(filename,sep=','):
        """Read a complete dataset from comma separated file, no header.  sep
        can be used to specify another separator."""
        dataset = DataSet()

        with open(filename,'r') as f:
            lines = f.readlines()
            var_count = len(lines[0].split(sep))
            for line in lines:
                inst = ( int(x) for x in line.split(sep) )
                inst = Inst.from_list(inst,var_count)
                dataset[inst] += 1
        return dataset

    def save_as_csv(self,filename):
        with open(filename,'w') as f:
            for inst,count in self:
                for i in range(count):
                    f.write(",".join(str(val) for var,val in inst))
                    f.write("\n")

    def log_likelihood(self):
        """log likelihood of a dataset, given the data distribution"""
        N = float(self.N)
        return sum( count*math.log(count/N) for inst,count in self )

    @staticmethod
    def simulate(psdd,N,seed=None):
        """Simulate a dataset of size N from a given PSDD"""
        if seed is not None: random.seed(seed)

        dataset = DataSet()
        var_count = psdd.vtree.var_count
        for i in range(N):
            inst = [None]*(var_count+1)
            inst = psdd.simulate(inst=inst)
            inst = Inst.from_list(inst,var_count,zero_indexed=False)
            dataset[inst] += 1
        return dataset

    @staticmethod
    def instances(var_count):
        """generates all instantiations over var_count variables"""
        for i in range(2**var_count):
            yield Inst.from_bitset(i,var_count)

class Inst(tuple):
    """Immutable instantiation.

    Used for datasets, more efficient than InstMap"""

    def __new__(cls,tpl):
        tpl = Inst._normalize_inst(tpl)
        return super(Inst,cls).__new__(cls,tpl)

    def __init__(self,tpl):
        """Assumes tpl is one-indexed (i.e., tpl[0] is None).
        Use from_list otherwise"""
        #super(Inst,self).__init__(tpl) # AC not needed?
        self.inst = self
        self.var_count = super(Inst,self).__len__()-1
        self.size = 0
        self.varset = set()
        self.bitset = 0

        for var,val in enumerate(super(Inst,self).__iter__()):
            if var == 0: continue
            var,val = self.check_key_value(var,val)
            if val is None: continue
            self.size += 1
            self.varset.add(var)
            if val == 1:
                self.bitset += 1 << (self.var_count-var)

    @staticmethod
    def _normalize_inst(inst):
        for value in inst:
            if value == True:  value = 1
            if value == False: value = 0
            if value == -1:    value = None
            yield value

    @staticmethod
    def _none_pad(lst,front_pad=False,end_pad=False):
        """Returns iterator that front and end pads a list with None's

        If front_pad is True, then adds one None to the front
        IF end_pad is a var_count, add None's to the end until var_count"""
        if front_pad:
            yield None
        for lst_count,x in enumerate(lst):
            yield x
        if end_pad is not False:
            lst_count = lst_count+1 if front_pad else lst_count
            end_pad = end_pad-lst_count
            for j in range(end_pad):
                yield None

    @staticmethod
    def _bit_enumerator(bitset,var_count):
        for var in range(var_count,0,-1):
            val = bitset % 2
            bitset = bitset / 2
            yield val

    @classmethod
    def from_list(cls,lst,var_count,zero_indexed=True):
        """New Inst from list/tuple.  Will pad at end if needed."""
        inst = Inst._none_pad(lst,front_pad=zero_indexed,end_pad=var_count)
        return cls(inst)

    @classmethod
    def from_dict(cls,dct,var_count):
        """new (possibly incomplete) Inst from dictionary."""
        inst = ( dct.get(i) for i in range(var_count+1) )
        return cls(inst)

    @classmethod
    def from_bitset(cls,bitset,var_count):
        """new (complete) Inst from bitstring"""
        inst = Inst._bit_enumerator(bitset,var_count)
        inst = Inst._none_pad(inst,front_pad=True)
        return cls(inst)

    @classmethod
    def from_literal(cls,lit,var_count):
        """ new Inst from literal"""
        inst = ( lit > 0 if i == abs(lit) else None for i in range(var_count+1) )
        return cls(inst)

    def __len__(self):
        return self.size

    def check_key_value(self, key, value):
        if type(key) is not int:
            raise TypeError("keys must be integers")
        if value is not None and not -1 <= value <= 1:
            raise TypeError("value must be -1/None or 0/False or 1/True")

        var = abs(key)
        if not 0 < var <= self.var_count:
            msg = "key abs(%d) expected to be in [1,%d]" % (key,self.var_count)
            raise ValueError(msg)

        # canonical values (None/0/1)
        if value == True:  value = 1
        if value == False: value = 0
        if value == -1:    value = None
        return var,value

    def __getitem__(self, key):
        """inst[key] where abs(key) is in [1,var_count]"""
        var,value = self.check_key_value(key,None)
        return super(Inst,self).__getitem__(var)

    #def __missing__(self, key): pass

    def __iter__(self):
        """generator for (var,value) pairs"""
        for var in self.varset:
            yield var,self.inst[var]

    #def __reversed__(self): pass

    def __contains__(self, item):
        """returns true if item is a variable that is set to a value"""
        var,value = self.check_key_value(item,None)
        return self.inst[var] is not None

    def __cmp__(self,other):
        """comparison, first be size and then by lexicographic order

        Intended for two Inst with the same var_count"""
        if len(self) < len(other): return -1
        if len(self) > len(other): return 1
        return cmp(self.bitset,other.bitset)

    def __repr__(self):
        st = { 0:"0",1:"1",None:"-" }
        return "".join(st[val] for val in self.inst[1:])

    @staticmethod
    def lit_to_value(lit):
        return 0 if lit < 0 else 1

    def is_compatible(self,lit):
        """Returns true if inst is compatible with lit, and false otherwise"""

        value = self.lit_to_value(lit)
        var,value = self.check_key_value(lit,value)
        if self.inst[var] is None:
            return True
        else:
            return self.inst[var] == value

class InstMap:
    """Instantiation as a map/dict.  Better for partial instantiations.

    Updates in this class should be reflected in Inst"""

    def __init__(self):
        self.var_count = 0 # max variable
        self.inst = dict()
        self.bitset = 0

    @classmethod
    def from_list(cls,lst,zero_indexed=True):
        """new (possibly incomplete) Inst from list."""
        inst = cls()
        start = 0 if zero_indexed else 1
        for var,val in enumerate(lst[start:]):
            inst[var+1] = val
        return inst

    @classmethod
    def from_dict(cls,dct):
        """new (possibly incomplete) Inst from dictionary."""
        inst = cls()
        for var in dct:
            inst[var] = dct[var]
        return inst

    @classmethod
    def from_bitset(cls,bitset,var_count):
        """new (complete) Inst from bitstring"""
        inst = cls()
        for var in range(var_count,0,-1):
            val = bitset % 2
            bitset = bitset / 2
            inst[var] = val
        return inst

    @classmethod
    def from_literal(cls,lit):
        """ new Inst from literal"""
        inst = cls()
        var = abs(lit)
        inst[var] = Inst.lit_to_value(lit)
        return inst

    def __len__(self):
        return len(self.inst)

    def check_key_value(self, key, value):
        if type(key) is not int:
            raise TypeError("keys must be integers")
        if value is not None and not -1 <= value <= 1:
            raise TypeError("value must be -1/None or 0/False or 1/True")

        var = abs(key)
        if var == 0:
            msg = "key abs(%d) expected to not be 0" % key
            raise ValueError(msg)

        # canonical values (None/0/1)
        if value == True:  value = 1
        if value == False: value = 0
        if value == -1:    value = None
        return var,value

    def __getitem__(self, key):
        """inst[key]"""
        var,value = self.check_key_value(key,None)
        if var in self.inst:
            return self.inst[var]
        else:
            return None

    #def __missing__(self, key): pass

    def __setitem__(self, key, value):
        """inst[key] = value"""
        var,value = self.check_key_value(key,value)

        if var > self.var_count:
            self.bitset = self.bitset << (var-self.var_count)
            self.var_count = var

        if value == 1 and (var not in self.inst or self.inst[var] is 0):
            self.bitset += 1 << (self.var_count-var)
        if (value == 0 or value is None) and \
           (var in self.inst and self.inst[var] == 1):
            self.bitset -= 1 << (self.var_count-var)

        if value is None:
            if var in self.inst:
                del self.inst[var]
        else:
            self.inst[var] = value

    def __delitem__(self, key):
        """del inst[key]"""
        var,value = self.check_key_value(key,None)
        if key not in self.inst: return # do not throw error?
        if self.inst[var] == 1:
            self.bitset -= 1 << (self.var_count-var)
        del self.inst[var]

    def __iter__(self):
        """generator for (var,value) pairs"""
        for var in list(self.inst.keys()):
            yield var,self.inst[var]

    #def __reversed__(self): pass

    def __contains__(self, item):
        """returns true if item is a variable that is set to a value"""
        return item in self.inst

    def __cmp__(self, other):
        """comparison, first by size and then by lexicographic order

        Intended for two Inst with the same var_count"""
        if len(self) < len(other): return -1
        if len(self) > len(other): return 1

        if self.var_count > other.var_count:
            me  = self.bitset
            you = other.bitset << (self.var_count-other.var_count)
        elif self.var_count < other.var_count:
            me  = self.bitset << (other.var_count-self.var_count)
            you = other.bitset
        else:
            me  = self.bitset
            you = other.bitset

        return cmp(me,you)

    def __repr__(self,as_bitstring=True):
        if as_bitstring:
            st = { 0:"0",1:"1",None:"-" }
            inst = [ self.inst[var] if var in self.inst else None \
                     for var in range(1,self.var_count+1) ]
            return "".join(st[val] for val in inst)
        else:
            return " ".join("%d:%d" % (var,self.inst[var]) \
                            for var in sorted(self.inst.keys()))

    def concat(self, other):
        """concatenates self with other and returns new Inst"""
        new_inst = InstMap()
        for var,val in self:
            new_inst[var] = val
        for var,val in other:
            new_inst[var] = val
        return new_inst

    def copy(self):
        """Clones the Inst"""
        new_inst = InstMap()
        for var,val in self:
            new_inst[var] = val
        return new_inst

    def shrink(self):
        """Deletes unused variables.  e.g., --0-1 becomes 01"""
        new_inst = InstMap()
        for i,(var,val) in enumerate(self):
            new_inst[i+1] = val
        return new_inst

    @staticmethod
    def lit_to_value(lit):
        return 0 if lit < 0 else 1

    def is_compatible(self,lit):
        """Returns true if inst is compatible with lit, and false otherwise"""
        var = abs(lit)
        val = self.lit_to_value(lit)
        if var not in self.inst:
            return True
        else:
            return self.inst[var] == val

class WeightedInstMap(InstMap):
    """InstMap with weights.

    This class was created for PSDD model enumeration (best-k MPE)."""

    def __init__(self,weight=0.0):
        InstMap.__init__(self)
        self.weight = weight

    @classmethod
    def from_literal(cls,lit,weight=0.0):
        inst = cls(weight=weight)
        var = abs(lit)
        inst[var] = Inst.lit_to_value(lit)
        return inst

    def set_weight(self,weight):
        self.weight = weight

    def mult_weight(self,weight):
        self.weight *= weight

    def concat(self, other):
        """concatenates self with other and returns new Inst"""
        new_inst = WeightedInstMap()
        for var,val in self:
            new_inst[var] = val
        for var,val in other:
            new_inst[var] = val
        new_inst.weight = self.weight * other.weight
        return new_inst

    def __cmp__(self, other):
        if self.weight > other.weight: return -1
        if self.weight < other.weight: return 1
        # otherwise, compare lexical
        return InstMap.__cmp__(self,other)

    def __repr__(self,as_bitstring=True):
        if as_bitstring:
            st = { 0:"0",1:"1",None:"-" }
            inst = [ self.inst[var] if var in self.inst else None \
                     for var in range(1,self.var_count+1) ]
            st = "".join(st[val] for val in inst)
            st += " %.4f" % self.weight
            return st
        else:
            st = " ".join("%d:%d" % (var,self.inst[var]) \
                            for var in sorted(self.inst.keys()))
            st += " %.4f" % self.weight
            return st
