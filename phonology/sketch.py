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

def applyRule(rule,i,untilSuffix, unrollBound):
    return FunctionCall("apply_rule", [rule,i, untilSuffix, Constant(unrollBound)])
def applyRules(rules,d, untilSuffix, b, doNothing = None):
    for j,r in enumerate(rules):
        if doNothing == None or (not doNothing[j]):
            d = applyRule(r,d, untilSuffix, b)
        else:
            d = doNothingRule(r,d,untilSuffix,Constant(b))
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
@sketchImplementation("do_nothing_rule")
def doNothingRule(*a): pass
@sketchImplementation("index_word")
def indexWord(*a): pass

def makeConstantVector(v):
    return Array(map(Constant,v))
def makeConstantMatrix(m):
    return Array([ makeConstantVector(v) for v in m ])
def makeConstantWord(bank, w):
    w = bank.variablesOfWord(w)
    return Constant('(new Word(l = %d, s = {%s}))'%(len(w),",".join(w)))
    

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

class SynthesisFailure(Exception):
    pass
class SynthesisTimeout(Exception):
    pass

globalTimeoutCounter = None
def setGlobalTimeout(seconds):
    global globalTimeoutCounter
    globalTimeoutCounter = seconds
def exhaustedGlobalTimeout():
    global globalTimeoutCounter
    return globalTimeoutCounter != None and int(globalTimeoutCounter/60.0) < 1

leaveSketches = False
def leaveSketchOutput():
    global leaveSketches
    leaveSketches = True

lastFailureOutput = None
lastSketchOutput = None
def solveSketch(bank, unroll = 8, maximumMorphLength = 9, alternationProblem = False, showSource = False, minimizeBound = None, timeout = None):
    global lastFailureOutput,lastSketchOutput,globalTimeoutCounter,leaveSketches

    leavitt = leaveSketches

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
        if exhaustedGlobalTimeout():
            print "Exhausted global timeout budget."
            raise SynthesisTimeout()
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
    if False and not leavitt:
        os.remove(temporarySketchFile)
        os.remove(outputFile)

    # Cleanup of temporary files
    temporaryOutputFolder = os.path.expanduser("~/.sketch/tmp")
    temporaryCleanupPath = temporaryOutputFolder + "/" + os.path.split(temporarySketchFile)[1]
    if os.path.exists(temporaryCleanupPath):
        #print "Removing temporary files ",temporaryCleanupPath
        os.system("rm -r " + temporaryCleanupPath)

    lastSketchOutput = output
    
    if "not be resolved." in output or "Rejected" in output or "Sketch front-end timed out" in output:
        lastFailureOutput = source+"\n"+output
        if "Sketch front-end timed out" in output: raise SynthesisTimeout()
        else: raise SynthesisFailure()
    elif "Program Parse Error" in output:
        print "FATAL: Could not parse program"
        print source
        print output
        assert False,"Sketch parse error"
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
