import re
from features import FeatureBank
from sketchSyntax import *
from rule import Rule

from random import random
import os



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
#@sketchImplementation("unknown_word")
def sampleMorph():
    return define("Word", FunctionCall("unknown_word",[]))
@sketchImplementation("concatenate3")
def concatenate3(x,y,z): pass
@sketchImplementation("word_length")
def wordLength(w): return len(w)
@sketchImplementation("rule_cost")
def ruleCost(r): return r.cost()

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

def solveSketch(bank):
    source = makeSketch(bank)
    with open("test.sk","w") as f:
        f.write(source)
    outputFile = "solver_output/%f" % random()
    print "Invoking solver..."
    command = "sketch  --bnd-unroll-amnt 15 test.sk > %s" % outputFile
    print command
    os.system(command)
    print "Finished calling solver."
    return open(outputFile,'r').read()


