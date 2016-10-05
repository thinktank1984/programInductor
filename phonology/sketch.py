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
#@sketchImplementation("unknown_word")
def sampleMorph():
    return define("Word", FunctionCall("unknown_word",[]))
@sketchImplementation("concatenate3")
def concatenate3(x,y,z): pass

def makeConstantVector(v):
    return "{ %s }" % ", ".join(map(str, v))

def makeConstantMatrix(matrix):
    return "{ %s }" % ", ".join([ "{ %s }" % ", ".join(map(str,fs)) for fs in matrix ])

def makeConstantWord(word):
    matrix = [featureVectorMap[t] for t in tokenize(word) ]
    return makeConstantWordOfMatrix(matrix)
def makeConstantWordOfMatrix(matrix):
    return makeWord(makeConstantMatrix(matrix))
def makeConstantPhoneme(p):
    vector = featureVectorMap[p] # list of boolean
    return makeConstantVector(map(str,vector))

def itePhoneme(c,p1,p2):
    f1 = featureVectorMap[p1]
    f2 = featureVectorMap[p2]
    f = [ (f1[j] if f1[j] == f2[j] else ite(c,f1[j],f2[j])) for j in range(len(f2)) ]
    return makeConstantVector(f)


def makeSketch():
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
    print "Invoking solver..."
    command = "sketch test.sk > %s" % outputFile # --bnd-unroll-amnt 50
    print command
    os.system(command)
    print "Finished calling solver."
    return open(outputFile,'r').read()

