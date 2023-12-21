#!/usr/bin/env python

from os import path
import random
from collections import defaultdict
import locale # for printing numbers with commas
locale.setlocale(locale.LC_ALL, "en_US.UTF8")

#import pypsdd
from .. import Vtree,SddManager,PSddManager,SddNode,PSddNode
from .. import Timer,DataSet,Inst,InstMap
from .. import Prior,DirichletPrior,UniformSmoothing
from .. import io

def fmt(number):
    return locale.format("%d",number,grouping=True)

def run_test_inst():

    print("=== Inst ===")
    inst = Inst([None,1,True,-1,None,-1,None,-1,None,0,False])
    print("%s == 11------00" % (inst,))
    inst = Inst.from_dict({1:1,10:0},10)
    print("%s == 1--------0" % (inst,))
    print("1 in inst? %s == True" % (1 in inst))
    print("2 in inst? %s == False" % (2 in inst))
    inst = Inst.from_dict({1:1,3:True,8:None,9:-1,10:False},10)
    print(inst)
    varset = ",".join(str(var) for var in inst.varset)
    print("varset: (%d) %s" % (len(inst),varset))
    for var,val in inst:
        print("inst[%d] = %d" % (var,val))

    inst = Inst.from_list([1,1,1,0,1,1,-1,None],10,zero_indexed=True)
    print(inst)
    print(format(inst.bitset,'b'))

    inst = Inst.from_list([1,1,1,0,1,1,-1,None],10,zero_indexed=False)
    print(inst)
    print(format(inst.bitset,'b'))

    inst = Inst.from_bitset(2047,10)
    print(inst)
    print(format(inst.bitset,'b'))

    inst = Inst.from_literal(1,10)
    print(inst)
    print("1---------")

    inst = Inst.from_literal(-1,10)
    print(inst)
    print("0---------")


    """
    inst_list = [ Inst.from_list([0,1,0,1,0],5), \
                  Inst.from_list([0,1,0,1,0,1],5,zero_indexed=False), \
                  Inst.from_list([0,1,None,1,-1],5), \
                  Inst.from_bitset(0,5), \
                  Inst.from_bitset(31,5), \
                  Inst.from_list([None]*5,5) ]
    print inst_list
    print "[01010, 10101, 01-1-, 00000, 11111, -----]"
    print sorted(inst_list)
    print "[-----, 01-1-, 00000, 01010, 10101, 11111]"

    inst1 = Inst.from_list([-1,-1,1,-1,1,-1],6)
    inst2 = Inst.from_list([-1,0,-1,0,-1,0],6)
    inst3 = inst1.concat(inst2)
    print "%s + %s = %s" % (inst1,inst2,inst3)
    """

    print("=== InstMap ===")
    inst = InstMap()
    inst[1] = 1
    inst[2] = True
    inst[9] = 0
    inst[10] = False
    print("%s == 11------00" % inst) 
    inst[2] = None
    del inst[9]
    print("%s == 1--------0" % inst) 
    print("1 in inst? %s == True" % (1 in inst))
    print("2 in inst? %s == False" % (2 in inst))
    inst[3] = True
    print(inst)
    varset = ",".join(str(var) for var in list(inst.inst.keys()))
    print("varset: (%d) %s" % (len(inst),varset))
    for var,val in inst:
        print("inst[%d] = %d" % (var,val))

    inst[5] = 1
    inst[6] = 1
    inst[7] = 0
    inst[8] = 0
    inst[9] = None
    inst[10] = None
    print(inst)
    print(format(inst.bitset,'b'))

    inst[5] = 0
    inst[6] = None
    inst[7] = 1
    inst[8] = None
    inst[9] = 0
    inst[10] = 1
    print(inst)
    print(format(inst.bitset,'b'))

    del inst[5]
    del inst[6]
    del inst[7]
    del inst[8]
    del inst[9]
    del inst[10]
    print(inst)
    print(format(inst.bitset,'b'))

    inst_list = [ InstMap.from_list([0,1,0,1,0]), \
                  InstMap.from_list([0,1,0,1,0,1],zero_indexed=False), \
                  InstMap.from_list([0,1,None,1,-1]), \
                  InstMap.from_bitset(0,5), \
                  InstMap.from_bitset(31,5), \
                  InstMap.from_list([None]*5) ]
    print(inst_list)
    print("[01010, 10101, 01-1-, 00000, 11111, -----]")
    print(sorted(inst_list))
    print("[-----, 01-1-, 00000, 01010, 10101, 11111]")

    inst1 = InstMap.from_list([-1,-1,1,-1,1,-1])
    inst2 = InstMap.from_list([-1,0,-1,0,-1,0])
    inst3 = inst1.concat(inst2)
    print("%s + %s = %s" % (inst1,inst2,inst3))

def run_test(vtree_filename,sdd_filename,N=1024,seed=0):

    # READ SDD
    with Timer("reading vtree and sdd"):
        vtree = Vtree.read(vtree_filename)
        manager = SddManager(vtree)
        alpha = io.sdd_read(sdd_filename,manager)

    # CONVERT TO PSDD
    with Timer("converting to psdd"):
        pmanager = PSddManager(vtree)
        beta = pmanager.copy_and_normalize_sdd(alpha,vtree)
        #prior = DirichletPrior(2.0)
        prior = UniformSmoothing(1024.0)
        prior.initialize_psdd(beta)

    # SIMULATE DATASETS
    with Timer("drawing samples"):
        random.seed(seed)
        training,testing = [],[]
        for i in range(N):
            training.append(beta.simulate())
        for i in range(N):
            testing.append(beta.simulate())

    # SIMULATE DATASETS
    with Timer("drawing samples (into dict)"):
        random.seed(seed)
        training,testing = defaultdict(lambda: 1),defaultdict(lambda: 1)
        for i in range(N):
            training[tuple(beta.simulate())] += 1
        for i in range(N):
            testing[tuple(beta.simulate())] += 1

    # SIMULATE DATASETS
    with Timer("drawing samples new (list)"):
        random.seed(seed)
        training,testing = [],[]
        for i in range(N):
            inst = [None]*(manager.var_count+1)
            training.append(beta.simulate(inst=inst))
        for i in range(N):
            inst = [None]*(manager.var_count+1)
            testing.append(beta.simulate(inst=inst))

    # SIMULATE DATASETS
    """
    with Timer("drawing samples new (map)"):
        random.seed(seed)
        training,testing = [],[]
        for i in xrange(N):
            training.append(beta.simulate())
        for i in xrange(N):
            testing.append(beta.simulate())
    """

    # SIMULATE DATASETS
    with Timer("simulating datasets"):
        training = DataSet.simulate(beta,N,seed=seed)
        testing  = DataSet.simulate(beta,N,seed=(seed+1))

    # PRINT SOME STATS
    print("================================")
    print(" sdd model count: %s" % fmt(alpha.model_count(vtree)))
    print("       sdd count: %s" % fmt(alpha.count()))
    print("        sdd size: %s" % fmt(alpha.size()))
    print("================================")
    print("psdd model count: %s" % fmt(beta.model_count()))
    print("      psdd count: %s" % fmt(beta.count()))
    print("       psdd size: %s" % fmt(beta.size()))
    print("================================")
    print("     theta count: %s" % fmt(beta.theta_count()))
    print("      zero count: %s" % fmt(beta.zero_count()))
    print("      true count: %s" % fmt(beta.true_count()))
    print("================================")
    print("   training size: %d" % training.N)
    print("    testing size: %d" % testing.N)
    print(" unique training: %d" % len(training))
    print("  unique testing: %d" % len(testing))

    if manager.var_count <= PSddNode._brute_force_limit:
        pass

    return beta,manager

def run_test_basename(basename,N=2**14):
    print("######## " + basename)
    dirname = path.join(path.dirname(__file__),'examples')
    vtree_filename = path.join(dirname,basename + '.vtree')
    sdd_filename = path.join(dirname,basename + '.sdd')
    alpha,pmanager = run_test(vtree_filename,sdd_filename,N=N)
    print()
    return alpha,pmanager

if __name__ == '__main__':
    run_test_inst()
    alpha,manager = run_test_basename('ranking-3')
    run_test_basename('example')
    run_test_basename('true')
    run_test_basename('literal')
    #run_test_basename('false')
    run_test_basename('xor-16')
    run_test_basename('xor-32')
    run_test_basename('alarm')
    run_test_basename('c432')
