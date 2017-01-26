# -*- coding: utf-8 -*-



from problems import underlyingProblems, interactingProblems, alternationProblems
from rule import *

import re
import math
import numpy as np
import matplotlib.pyplot as plot
import pickle
import os

def mergeFeatureCounts(m,n):
    c = {}
    for f in set(m.keys() + n.keys()):
        c[f] = m.get(f,0.0) + n.get(f,0.0)
    return c

T = 0.10 # temperature

def getRulesFromComment(problem):
    return [ l for l in problem.description.split("\n") if '--->' in l ]
def getFeaturesFromComment(problem):
    return [ f for r in getRulesFromComment(problem) for f in re.findall('[\-\+]([a-zA-Z]+)',r) ]

PICKLES = [ "pickles/"+f for f in os.listdir("pickles") if f.endswith('.p') ]

def loadRules(pickledFile): return pickle.load(open(pickledFile, 'rb'))
def expectedFeatureCounts(pickledFile):
    solutions = loadRules(pickledFile)
    alternation = 'alternation' in pickledFile

    def solutionCost(solution):
        if alternation:
            return sum([ r.alternationCost() for r in solution ])
        else:
            return sum([ r.cost() for r in solution ])

    posterior = [ math.exp(-solutionCost(s)/T) for s in solutions ]
    z = sum(posterior)
    posterior = [ p/z for p in posterior ]

    def weightedCounts(w,solution):
        counts = {}
        for r in solution:
            r = str(r)
            for f in re.findall('[\-\+]([a-zA-Z]+)',r):
                counts[f] = counts.get(f,0.0) + w
        return counts

    expectedCounts = {}
    for j in range(len(solutions)):
        # for r in solutions[j]: print r
        # print " ==  ==  == "
        expectedCounts = mergeFeatureCounts(expectedCounts, weightedCounts(posterior[j], solutions[j]))
    return expectedCounts
    
    
    
    
aggregateCounts = {}
for pickledFile in PICKLES:
    aggregateCounts = mergeFeatureCounts(aggregateCounts, expectedFeatureCounts(pickledFile))

counts = list(reversed(sorted([ (aggregateCounts[f], f) for f in aggregateCounts ])))

# make a histogram of which features were popular
x = range(len(counts))
plot.bar(x, [ c for c,f in counts ])
plot.xticks(x, [f for c,f in counts ], rotation = 'vertical')

plot.ylabel('Probability')

plot.show()

