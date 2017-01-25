from features import FeatureBank
from sketchSyntax import *
from rule import Rule

from random import random
import os
from time import time
import re



@sketchImplementation("alternation_cost")
def alternationCost(r): pass
@sketchImplementation("applyRule")
def applyRule(rule,i):
    pass
@sketchImplementation("make_word")
def makeWord(features): return features
@sketchImplementation("word_equal")
def wordEqual(w1,w2):
    pass
@sketchImplementation("phonological_rule")
def phonologicalRule(i): pass
@sketchImplementation("apply_rule")
def applyRule(r,w): pass
@sketchImplementation("concatenate")
def concatenate(x,y): pass
@sketchImplementation("concatenate3")
def concatenate3(x,y,z): pass
@sketchImplementation("word_length")
def wordLength(w): return len(w)
@sketchImplementation("rule_cost")
def ruleCost(r): return r.cost()
@sketchImplementation("rule_equal")
def ruleEqual(p,q): return p == q
@sketchImplementation("alternation_equal")
def alternationEqual(p,q): return p == q
@sketchImplementation("is_deletion_rule")
def isDeletionRule(r): return r.structuralChange == None
@sketchImplementation("fix_structural_change")
def fixStructuralChange(r): pass

def makeConstantVector(v):
    return Array(map(Constant,v))
def makeConstantMatrix(m):
    return Array([ makeConstantVector(v) for v in m ])
def makeConstantWord(bank, w):
    w = bank.variablesOfWord(w)
    w = Array([ Variable(v) for v in w ])
    return makeWord(w)


def makeSketch(bank):
    h = ""
    h += "#define NUMBEROFFEATURES %d\n" % len(bank.features)
    h += "#define True 1\n#define False 0\n"
    h += bank.sketch()
    h += "#include \"common.skh\"\n"
    h += makeSketchSkeleton()
    return h

def solveSketch(bank, unroll = 8):
    source = makeSketch(bank)
    with open("test.sk","w") as f:
        f.write(source)
    outputFile = "solver_output/%f" % random()
    command = "sketch --bnd-unroll-amnt %d test.sk > %s 2> %s" % (unroll, outputFile, outputFile)
    print "Invoking solver: %s"%command
    startTime = time()
    os.system(command)
    print "Ran the solver in %f"%(time() - startTime)
    output = open(outputFile,'r').read()
    if "The sketch could not be resolved." in output:
        return None
    else:
        return output


