# -*- coding: utf-8 -*-



from problems import underlyingProblems, interactingProblems, alternationProblems
from rule import *
from utilities import *

from time import time
import re
import math
import numpy as np
import matplotlib.pyplot as plot
import pickle
import os
from random import random



class UG():
    def logLikelihood(self, rules):
        return sum([self.ruleLogLikelihood(r) for r in rules ])
    
    def ruleLogLikelihood(self, r):
        return self.specificationLogLikelihood(r.focus) + self.specificationLogLikelihood(r.structuralChange) + self.guardLogLikelihood(r.leftTriggers) + self.guardLogLikelihood(r.rightTriggers)
    
    def specificationLogLikelihood(self, s):
        if isinstance(s, ConstantPhoneme):
            return self.constantLogLikelihood()
        if isinstance(s, EmptySpecification):
            return self.emptySpecificationLogLikelihood()
        if isinstance(s, FeatureMatrix):
            return self.featureMatrixLogLikelihood(s)
        raise Exception('Unknown specification:'+str(s))

    def featureMatrixLogLikelihood(self, matrix):
        return -1 + sum([ self.featureLogLikelihood(p,f) for p,f in matrix.featuresAndPolarities ])

    def featureLogLikelihood(self, p, f): return -1

    def guardLogLikelihood(self, g):
        return self.starLogLikelihood()*int(g.starred) + self.endLogLikelihood()*int(g.endOfString) + sum([ self.specificationLogLikelihood(s) for s in g.specifications ])

    def emptySpecificationLogLikelihood(self): return - EmptySpecification().cost()
    def constantLogLikelihood(self): return - ConstantPhoneme(None).cost()
    def starLogLikelihood(self): return -1
    def endLogLikelihood(self): return -1


class FlatUG(UG):
    def __init__(self): pass
    def logLikelihood(self, rules):
        return -sum([r.cost() for r in rules ])

class ChomskyUG(UG):
    def __init__(self): pass
    @staticmethod
    def minimizeRuleSet(rules):
        if len(rules) < 2: return rules
        doNotMerge = [rules[0]] + ChomskyUG.minimizeRuleSet(rules[1:])
        merged = [rules[0].merge(rules[1])] + ChomskyUG.minimizeRuleSet(rules[2:])
        if sum([r.cost() for r in merged ]) < sum([r.cost() for r in doNotMerge ]):
            return merged
        else:
            return doNotMerge
    
    def logLikelihood(self, rules):
        return -sum([r.cost() for r in ChomskyUG.minimizeRuleSet(rules) ])
        

class FeatureUG(UG):
    def __init__(self, featureFrequencies):
        z = float(sum(featureFrequencies.values()))
        self.likelihoods = dict([ (f,featureFrequencies[f]/z) for f in featureFrequencies ])

    def orderedPairs(self):
        return list(reversed(sorted([ (self.likelihoods[f], f) for f in self.likelihoods ])))
    
    def __str__(self):
        return "\n".join([ f + "\t" + str(l) for l,f in self.orderedPairs() ])

    def plot(self):
        x = range(len(self.likelihoods))
        plot.bar(x, [ c for c,f in self.orderedPairs() ])
        plot.xticks(x, [ f for c,f in self.orderedPairs() ], rotation = 'vertical')
        plot.ylabel('probability')

        plot.show()

    @staticmethod
    def fromPosterior(weightedSolutions):
        counts = {}
        for w,s in weightedSolutions:
            fs = [ f for r in s for f in re.findall('[\-\+]([a-zA-Z]+)',str(r)) ]
            counts = mergeCounts(counts, dict([(f,w) for f in fs ]))
        return FeatureUG(counts)
        
    def featureLogLikelihood(self, _, f):
        if not f in self.likelihoods:
            f = min([(self.likelihoods[g],g) for g in self.likelihoods])[1]
        return math.log(self.likelihoods[f])/math.log(len(self.likelihoods))

class SkeletonUG(UG):
    def __init__(self, skeletonFrequencies):
        z = float(sum(skeletonFrequencies.values()))
        self.likelihoods = dict([ (f,skeletonFrequencies[f]/z) for f in skeletonFrequencies ])

    def orderedPairs(self):
        return list(reversed(sorted([ (self.likelihoods[f], f) for f in self.likelihoods ])))
    
    def __str__(self):
        return "\n".join([ f + "\t" + str(l) for l,f in self.orderedPairs() ])

    def plot(self):
        pears = [ (c,s) for c,s in self.orderedPairs() if c*50 > 1]
        x = range(len(pears))
        plot.bar(x, [ c for c,f in pears ])
        plot.xticks(x, [ f for c,f in pears ], rotation = 'vertical')
        plot.ylabel('probability')

        plot.show()

    @staticmethod
    def fromPosterior(weightedSolutions):
        counts = {}
        for w,s in weightedSolutions:
            fs = [ r.skeleton() for r in s ]
            counts = mergeCounts(counts, dict([(f,w) for f in fs ]))
        return SkeletonUG(counts)
        
    def ruleLogLikelihood(self, r):
        s = r.skeleton()
        if not s in self.likelihoods:
            s = min([(self.likelihoods[g],g) for g in self.likelihoods ])[1]
        return math.log(self.likelihoods[s])

class SkeletonFeatureUG(UG):
    def __init__(self, skeletonFrequencies, featureFrequencies):
        self.skeletons = SkeletonUG(skeletonFrequencies)
        self.features = FeatureUG(featureFrequencies)

    def __str__(self):
        return str(self.skeletons) + "\n" + str(self.features)
    def plot(self):
        self.skeletons.plot()
        self.features.plot()

    def ruleLogLikelihood(self,r):
        fs = [ self.features.featureLogLikelihood(None, f)
               for f in re.findall('[\-\+]([a-zA-Z]+)',str(r)) ]
        return math.log(self.skeletons.likelihoods[r.skeleton()]) + sum(fs)

    @staticmethod
    def fromPosterior(weightedSolutions):
        return SkeletonFeatureUG(SkeletonUG.fromPosterior(weightedSolutions).likelihoods,
                                 FeatureUG.fromPosterior(weightedSolutions).likelihoods)

class ChineseUG():
    def __init__(self, matrixFrequencies):
        z = float(sum(matrixFrequencies.values()))
        self.matrixFrequencies = dict([ (f,matrixFrequencies[f]/z) for f in matrixFrequencies ])

    def featureMatrixLogLikelihood(self, matrix):
        return math.log(self.matrixFrequencies[str(matrix)])

savedSkeletonFeature = None    
def str2ug(n):
    global savedSkeletonFeature
    if n == 'flat': return FlatUG()
    if n == 'Chomsky': return ChomskyUG()
    if n == 'learned':
        if savedSkeletonFeature == None:
            allSolutions,solutionNames = loadAllSolutions()
            savedSkeletonFeature = estimateUG(allSolutions, SkeletonFeatureUG, temperature = 2.0, iterations = 2, jitter = 0.5)
        return savedSkeletonFeature
    assert False
    
    

def estimateUG(problemSolutions, k, iterations = 1, temperature = 1.0, jitter = 0.0):
    # build the initial posterior
    # each solution is a list of rules
    logPosteriors = [ normalizeLogDistribution([ (ChomskyUG().logLikelihood(s)/temperature + random()*jitter, s)
                                                 for s in solutions ])
                      for solutions in problemSolutions ]
    posteriors = [ [ (math.exp(w), s) for w,s in solutions ]
                   for solutions in logPosteriors ]

    for _ in range(iterations):
        flattenedPosterior = [ weightedSolution for solutions in posteriors for weightedSolution in solutions ]
        grammar = k.fromPosterior(flattenedPosterior)
        logPosteriors = [ normalizeLogDistribution([ (grammar.logLikelihood(s)/temperature, s) for s in solutions ])
                          for solutions in problemSolutions ]
        posteriors = [ [ (math.exp(w), s) for w,s in solutions ]
                       for solutions in logPosteriors ]

    return grammar

def loadAllSolutions():
    startTime = time()
    PICKLES = [ "pickles/"+f for f in os.listdir("pickles") if f.endswith('.p') ]
    def loadRules(pickledFile):
        ss = pickle.load(open(pickledFile, 'rb'))
        if isinstance(ss[0],Rule): # if each solution is just a single rule
            ss = [[s] for s in ss ]
        return ss
    allSolutions = map(loadRules, PICKLES)
    print "Loaded all solutions in %d seconds"%(int(time() - startTime))
    return allSolutions, PICKLES



if __name__ == '__main__':
    allSolutions,solutionNames = loadAllSolutions()
    for j,solution in enumerate(allSolutions):
        print solutionNames[j],
        solution = max(solution, key = lambda r: FlatUG().logLikelihood(r))
        print [FlatUG().logLikelihood([r]) for r in solution ]
        print [str(r) for r in solution ]
    for k in [SkeletonFeatureUG,SkeletonUG,FeatureUG]:
        g = estimateUG(allSolutions, k, temperature = 1.0, iterations = 2, jitter = 0.5)
        print g
#        g.plot()
