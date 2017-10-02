# -*- coding: utf-8 -*-

from features import FeatureBank
from sketchSyntax import *
from utilities import *

import math
from random import random
import os
from time import time
import re


@sketchImplementation("alternation_cost")
def alternationCost(r): pass

def applyRule(rule,i,unrollBound):
    if callable(rule): return rule(i,Constant(unrollBound))
    else: return FunctionCall("apply_rule", [rule,i,Constant(unrollBound)])
def applyRules(rules,d,b):
    for r in rules: d = applyRule(r,d,b)
    return d
@sketchImplementation("make_word")
def makeWord(features): return features
@sketchImplementation("word_equal")
def wordEqual(w1,w2):
    pass
@sketchImplementation("phonological_rule")
def phonologicalRule(i): pass
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
    h += "\n".join(["#define %s %s"%(k,v) for k,v in currentModelPreprocessorDefinitions().iteritems() ])
    h += "\n"
    h += "#include \"common.skh\"\n"
    h += makeSketchSkeleton()
    return h

globalTimeoutCounter = None
def setGlobalTimeout(seconds):
    global globalTimeoutCounter
    globalTimeoutCounter = seconds
lastFailureOutput = None
lastSketchOutput = None
def solveSketch(bank, unroll = 8, maximumMorphLength = 9, alternationProblem = False, leavitt = False, showSource = False, minimizeBound = None, timeout = None):
    global lastFailureOutput,lastSketchOutput,globalTimeoutCounter

    source = makeSketch(bank, maximumMorphLength, alternationProblem)

    # figure out how many bits you need for the minimization bound
    if minimizeBound != None:
        minimizeBound = int(math.ceil(math.log(minimizeBound + 1)/math.log(2)))
    else:
        minimizeBound = 5

    # Temporary file for writing the sketch
    temporarySketchFile = makeTemporaryFile('.sk')
    with open(temporarySketchFile,'w') as handle:
        handle.write(source)

    if showSource: print source

    # Temporary file for collecting the sketch output
    outputFile = makeTemporaryFile('',d = './solver_output')

    if timeout != None: timeout = ' --fe-timeout %d '%(int(timeout/60.0))
    elif globalTimeoutCounter != None:
        if int(globalTimeoutCounter/60.0) < 1:
            print "Exhausted global timeout budget."
            return 'timeout'
        timeout = ' --fe-timeout %d '%(int(globalTimeoutCounter/60.0))
    else: timeout = ''
    
    command = "sketch %s --bnd-mbits %d -V 10 --bnd-unroll-amnt %d %s > %s 2> %s" % (timeout,
                                                                                     minimizeBound,
                                                                                     unroll,
                                                                                     temporarySketchFile,
                                                                                     outputFile,
                                                                                     outputFile)
    print "Invoking solver: %s"%command
    startTime = time()
    flushEverything()
    os.system(command)
    print "Ran the solver in %02f sec"%(time() - startTime)
    if globalTimeoutCounter != None: globalTimeoutCounter -= (time() - startTime)
    flushEverything()
    
    output = open(outputFile,'r').read()
    if not leavitt:
        os.remove(temporarySketchFile)
        os.remove(outputFile)

    lastSketchOutput = output
    
    if "not be resolved." in output or "Rejected" in output or "Sketch front-end timed out" in output:
        lastFailureOutput = source+"\n"+output
        if "Sketch front-end timed out" in output: return 'timeout'
        return None
    else:
        return output


def printSketchFailure():
    global lastFailureOutput
    print lastFailureOutput
def printLastSketchOutput():
    global lastSketchOutput
    print lastSketchOutput


def deleteTemporarySketchFiles():
    os.system("rm tmp*sk")
    os.system("rm -r ~/.sketch/tmp/tmp*")
