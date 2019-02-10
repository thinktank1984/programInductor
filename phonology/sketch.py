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
@sketchImplementation("is_insertion_rule")
def isInsertionRule(r): pass
@sketchImplementation("rule_uses_boundary")
def isBoundaryRule(r): pass
@sketchImplementation("do_nothing_rule")
def doNothingRule(*a): pass
@sketchImplementation("index_word")
def indexWord(*a): pass
@sketchImplementation("rule_does_nothing")
def ruleDoesNothing(*a): pass
@sketchImplementation("pattern_cost")
def patternCost(*a): return
@sketchImplementation("match_pattern")
def matchPattern(*a): return

def makeConstantVector(v):
    return Array(map(Constant,v))
def makeConstantMatrix(m):
    return Array([ makeConstantVector(v) for v in m ])
def makeConstantWord(bank, w):
    w = bank.variablesOfWord(w)
    return Constant('(new Word(l = %d, s = {%s}))'%(len(w),",".join(w)))
    

def makeSketch(bank, maximumMorphLength = 9, alternationProblem = False):
    global featuresAreDisabled
    global cleanIsDisabled
    global geometryIsEnabled
    h = ""
    if alternationProblem:
        h += "#define ALTERNATIONPROBLEM\n"
    if featuresAreDisabled:
        h += "#define DISABLEFEATURES\n"
    if cleanIsDisabled:
        h += "#define DISABLECLEAN\n"
    h += "#define MAXIMUMMORPHLENGTH %d\n"%maximumMorphLength
    h += "#define NUMBEROFFEATURES %d\n" % len(bank.features)
    h += "#define True 1\n#define False 0\n"
    h += bank.sketch(placeAssimilation=geometryIsEnabled,
                     nasalAssimilation=geometryIsEnabled)
    h += "\n".join(["#define %s %s"%(k,v) for k,v in currentModelPreprocessorDefinitions().iteritems() ])
    h += "\n"
    h += "#include \"common.skh\"\n"
    h += makeSketchSkeleton()
    return h

class SynthesisFailure(Exception):
    pass
class SynthesisTimeout(Exception):
    pass
class MemoryExhausted(Exception):
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

globalSketchTime = 0.0
def getGlobalSketchTime():
    global globalSketchTime
    return globalSketchTime

cleanIsDisabled = False
featuresAreDisabled = False
geometryIsEnabled = False
def disableFeatures():
    global featuresAreDisabled
    featuresAreDisabled = True
def disableClean():
    global cleanIsDisabled
    cleanIsDisabled = True
def isCleanDisabled():
    global cleanIsDisabled
    return cleanIsDisabled
def areFeaturesDisabled():
    global featuresAreDisabled
    return featuresAreDisabled
def enableGeometry():
    global geometryIsEnabled
    geometryIsEnabled = True
def disableGeometry():
    global geometryIsEnabled
    geometryIsEnabled = False
    
class useGlobalTimeout(object):
    def __init__(self):
        pass

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, type, value, traceback):
        dt = time() - self.start
        global globalTimeoutCounter
        if globalTimeoutCounter is not None: globalTimeoutCounter -= dt

lastFailureOutput = None
lastSketchOutput = None
def solveSketch(bank, unroll = 8, maximumMorphLength = 9, alternationProblem = False, showSource = False, minimizeBound = None, timeout = None):
    from command_server import send_to_command_server
    global lastFailureOutput,lastSketchOutput,globalTimeoutCounter,leaveSketches,globalSketchTime

    leavitt = leaveSketches
    
    # figure out how many bits you need for the minimization bound
    if minimizeBound != None:
        minimizeBound = int(math.ceil(math.log(minimizeBound + 1)/math.log(2)))
    else:
        minimizeBound = 5

    this_timeout = min(timeout or float('inf'),
                       globalTimeoutCounter or float('inf'))
    if this_timeout == float('inf'): timeout_str = ''
    else:
        if exhaustedGlobalTimeout():
            print "Exhausted global timeout budget."
            raise SynthesisTimeout()
        timeout_str = ' --fe-timeout %d '%(int(this_timeout/60.0))
    



    source = makeSketch(bank, maximumMorphLength, alternationProblem)
    source = """pragma options "--slv-p-cpus 1  --bnd-mbits %d -V 10 --bnd-unroll-amnt %d %s ";
"""%(minimizeBound,unroll,timeout_str) + source
    
    # Temporary file for writing the sketch
    temporarySketchFile = makeTemporaryFile('.sk')
    with open(temporarySketchFile,'w') as handle:
        handle.write(source)

    if showSource: print source

    # Temporary file for collecting the sketch output
    outputFile = makeTemporaryFile('',d = './solver_output')

    command = "sketch %s > %s 2> %s" % (temporarySketchFile,
                                        outputFile,
                                        outputFile)
    print "Invoking solver: %s"%command
    startTime = time()
    flushEverything()
    
    actualTime = send_to_command_server(command, this_timeout)
    timedOut = False
    if actualTime == 'timeout':
        actualTime = this_timeout
        timedOut = True
    print "Ran the solver in %02f sec (%02f wall clock, includes blocking)"%(actualTime, time() - startTime)
    globalSketchTime += actualTime
    if globalTimeoutCounter != None: globalTimeoutCounter -= actualTime
    flushEverything()
    
    output = open(outputFile,'r').read()
    if not leavitt:
        if (not 'LEAVESKETCH' in os.environ) or (os.environ['LEAVESKETCH'] == '0'):
            os.remove(temporarySketchFile)
            os.remove(outputFile)

    # Cleanup of temporary files
    temporaryOutputFolder = os.path.expanduser("~/.sketch/tmp")
    temporaryCleanupPath = temporaryOutputFolder + "/" + os.path.split(temporarySketchFile)[1]
    if os.path.exists(temporaryCleanupPath):
        #print "Removing temporary files ",temporaryCleanupPath
        os.system("rm -rf " + temporaryCleanupPath)

    lastSketchOutput = output

    timedOut = timedOut or "Sketch front-end timed out" in output
    
    if "not be resolved." in output or "Rejected" in output or timedOut:
        lastFailureOutput = source+"\n"+output
        if timedOut: raise SynthesisTimeout()
        else: raise SynthesisFailure()
    elif "Cannot allocate memory" in output: raise MemoryExhausted()
    elif "Program Parse Error" in output:
        print "FATAL: Could not parse program"
        print source
        print output
        assert False,"Sketch parse error"
    else:
        if actualTime > 5*60 and actualTime < 15*60:
            print "This sketch should be sent to Armando!"
            sendItToArmando(source, actualTime)
        return output


def printSketchFailure():
    global lastFailureOutput
    print lastFailureOutput
def printLastSketchOutput():
    global lastSketchOutput
    print lastSketchOutput

def sendItToArmando(source, time):
    temporarySketchFile = makeTemporaryFile('.sk',d="phonologySketches")
    with open(temporarySketchFile,'w') as handle:
        handle.write("// Ran in time " + str(time) + " seconds\n")
        handle.write(source)

    
def deleteTemporarySketchFiles():
    os.system("rm tmp*sk")
    os.system("rm -r ~/.sketch/tmp/tmp*")
