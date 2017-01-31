# -*- coding: utf-8 -*-



from problems import underlyingProblems, interactingProblems, alternationProblems
from rule import *

import re
import math
import numpy as np
import matplotlib.pyplot as plot
import pickle
import os

def mergeCounts(m,n):
    c = {}
    for f in set(m.keys() + n.keys()):
        c[f] = m.get(f,0.0) + n.get(f,0.0)
    return c

T = 10 # temperature

def getRulesFromComment(problem):
    return [ l for l in problem.description.split("\n") if '--->' in l ]
def getFeaturesFromComment(problem):
    return [ f for r in getRulesFromComment(problem) for f in re.findall('[\-\+]([a-zA-Z]+)',r) ]

PICKLES = [ "pickles/"+f for f in os.listdir("pickles") if f.endswith('.p') ]

def loadRules(pickledFile):
    ss = pickle.load(open(pickledFile, 'rb'))
    if isinstance(ss[0],Rule): # if each solution is just a single rule
        ss = [[s] for s in ss ]
    return ss
def expectedCounts(pickledFile):
    solutions = loadRules(pickledFile)
    alternation = 'alternation' in pickledFile

    def solutionCost(solution):
        if alternation:
            return sum([ r.alternationCost() for r in solution ])
        else:
            return sum([ r.cost() for r in solution ])

    posterior = [ math.exp(-solutionCost(s)/float(T)) for s in solutions ]
    z = sum(posterior)
    posterior = [ p/z for p in posterior ]

    def weightedCounts(w,solution):
        fc = {}
        sc = {}
        for r in solution:
            sc[r.skeleton()] = sc.get(r.skeleton(),0.0) + w
            for f in re.findall('[\-\+]([a-zA-Z]+)',str(r)):
                fc[f] = fc.get(f,0.0) + w
        return fc,sc

    featureCounts = {}
    skeletonCounts = {}
    for j in range(len(solutions)):
        # for r in solutions[j]: print r
        # print " ==  ==  == "
        fs,skeletons = weightedCounts(posterior[j], solutions[j])
        featureCounts =  mergeCounts(featureCounts, fs)
        skeletonCounts = mergeCounts(skeletonCounts, skeletons)
    return featureCounts, skeletonCounts
    
    
    
    
aggregateFeatureCounts = {}
aggregateSkeletonCounts = {}
for pickledFile in PICKLES:
    fc,sc = expectedCounts(pickledFile)
    aggregateFeatureCounts = mergeCounts(aggregateFeatureCounts, fc)
    aggregateSkeletonCounts = mergeCounts(aggregateSkeletonCounts, sc)

featureCounts = list(reversed(sorted([ (aggregateFeatureCounts[f], f) for f in aggregateFeatureCounts ])))
skeletonCounts = list(reversed(sorted([ (aggregateSkeletonCounts[f], f) for f in aggregateSkeletonCounts
                                        if aggregateSkeletonCounts[f] > 0.5 ])))
print "\n".join(map(str,skeletonCounts))

for counts in [featureCounts,skeletonCounts]:
    # make a histogram of which features were popular
    x = range(len(counts))
    plot.bar(x, [ c for c,f in counts ])
    plot.xticks(x, [f for c,f in counts ], rotation = 'vertical')

    plot.ylabel('Relative frequency')

    plot.show()

