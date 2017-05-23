# -*- coding: utf-8 -*-

from features import FeatureBank
from sketchSyntax import *

from random import random
import os
from time import time
import re
import tempfile


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


def makeSketch(bank, maximumMorphLength = 9, alternationProblem = False):
    h = ""
    if alternationProblem:
        h += "#define ALTERNATIONPROBLEM\n"
    h += "#define MAXIMUMMORPHLENGTH %d\n"%maximumMorphLength
    h += "#define NUMBEROFFEATURES %d\n" % len(bank.features)
    h += "#define True 1\n#define False 0\n"
    h += bank.sketch()
    h += "#include \"common.skh\"\n"
    h += makeSketchSkeleton()
    return h

lastFailureOutput = None
def solveSketch(bank, unroll = 8, maximumMorphLength = 9, alternationProblem = False, leavitt = False, showSource = False):
    global lastFailureOutput
    
    source = makeSketch(bank, maximumMorphLength, alternationProblem)

    # Temporary file for writing the sketch
    fd = tempfile.NamedTemporaryFile(mode = 'w',suffix = '.sk',delete = False,dir = '.')
    fd.write(source)
    fd.close()

    if showSource: print source

    # Temporary file for collecting the sketch output
    od = tempfile.NamedTemporaryFile(mode = 'w',delete = False,dir = './solver_output')
    od.write('') # just create the file that were going to overwrite
    od.close()
    outputFile = od.name
    
    command = "sketch  -V 10 --bnd-unroll-amnt %d %s > %s 2> %s" % (unroll, fd.name, outputFile, outputFile)
    print "Invoking solver: %s"%command
    startTime = time()
    os.system(command)
    print "Ran the solver in %f"%(time() - startTime)
    
    output = open(outputFile,'r').read()
    if not leavitt:
        os.remove(fd.name)
    
    if "not be resolved." in output or "Rejected" in output:
        lastFailureOutput = source+"\n"+output
        return None
    else:
        return output


def printSketchFailure():
    global lastFailureOutput
    print lastFailureOutput
