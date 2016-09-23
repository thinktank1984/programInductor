import re
from features import featureVectorMap,tokenize,FEATURELIST
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

def makeConstantMatrix(matrix):
    return "{ %s }" % ", ".join([ "{ %s }" % ", ".join(map(str,fs)) for fs in matrix ])

def makeConstantWord(word):
    matrix = [featureVectorMap[t] for t in tokenize(word) ]
    return makeConstantWordOfMatrix(matrix)
def makeConstantWordOfMatrix(matrix):
    return makeWord(makeConstantMatrix(matrix))


def makeSketch():
    global flipCounter
    h = ""
    h += "#define NUMBEROFFEATURES %d\n" % len(FEATURELIST)
    h += "#include \"common.skh\"\n"
    h += makeSketchSkeleton()
    return h

def solveSketch():
    source = makeSketch()
    with open("test.sk","w") as f:
        f.write(source)
    outputFile = "solver_output/%f" % random()
    os.system("sketch --slv-parallel --bnd-unroll-amnt 50 test.sk > %s" % outputFile)
    return open(outputFile,'r').read()

