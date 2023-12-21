#!/usr/bin/env python

from os import path
import locale # for printing numbers with commas
locale.setlocale(locale.LC_ALL, "en_US.UTF8")
from pypsdd import Timer,Vtree,SddManager,SddNode
from pypsdd import io

def fmt(number):
    return locale.format("%d",number,grouping=True)

def run_test(vtree_filename,sdd_filename,\
             print_models=10,count_models=100,enum_models=0):
    # READ SDD
    with Timer("reading vtree and sdd"):
        vtree = Vtree.read(vtree_filename)
        manager = SddManager(vtree)
        alpha = io.sdd_read(sdd_filename,manager)

    with Timer("counting %d models" % count_models):
        for i in range(count_models):
            alpha.model_count(vtree)

    # PRINT SOME STATS
    print("================================")
    print(" sdd model count: %s" % fmt(alpha.model_count(vtree)))
    print("       sdd count: %s" % fmt(alpha.count()))
    print("        sdd size: %s" % fmt(alpha.size()))

    if print_models:
        models = []
        with Timer("enumerating models"):
            for model in alpha.models(vtree,lexical=True):
                st = "".join( str(val) for var,val in model )
                models.append(st)
                if len(models) >= print_models: break
        for model in models:
            print(model)
        print("%d models (%d max)" % (len(models),print_models))

    if enum_models:
        models = []

        with Timer("enumerating %d models" % enum_models):
            for model in alpha.models(vtree):
                models.append(model)
                if len(models) >= enum_models: break

        with Timer("evaluating %d models" % enum_models):
            for model in models:
                if not alpha.is_model(model):
                    print("error: non-model", model)
                if not alpha._is_bits_and_data_clear(): # random check
                    print("error: bits or data not clear")

    return alpha,manager

def run_test_basename(basename,enum_models=0):
    print("######## " + basename)
    dirname = path.join(path.dirname(__file__),'examples')
    vtree_filename = path.join(dirname,basename + '.vtree')
    sdd_filename = path.join(dirname,basename + '.sdd')
    alpha,manager = run_test(vtree_filename,sdd_filename,enum_models=enum_models)
    print()
    return alpha,manager

if __name__ == '__main__':
    run_test_basename('ranking-3')
    alpha,manager = run_test_basename('example',enum_models=1000)
    run_test_basename('true')
    run_test_basename('literal')
    run_test_basename('false')
    run_test_basename('xor-4')
    run_test_basename('xor-8')
    run_test_basename('xor-16')
    run_test_basename('xor-32')
    run_test_basename('alarm',enum_models=1000)
    run_test_basename('c432',enum_models=1000)
