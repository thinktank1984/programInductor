# -*- coding: utf-8 -*-

from matrix import *

from pathos.multiprocessing import ProcessingPool as Pool
from time import time
import random

class RandomSampleSolver():
    def __init__(self, data, timeout, lower, upper):
        self.data = data
        self.timeout = timeout
        self.lower = lower
        self.upper = min(upper, len(self.data))

    def worker(self, seed):
        random.seed(seed)
        startTime = time()

        solutions = []

        while time() - startTime < self.timeout:
            # Sample another group of data
            size = random.choice(range(self.lower, self.upper + 1))
            startingPoint = choice(range(len(self.data) - size + 1))
            endingPoint = startingPoint + size
            subset = self.data[startingPoint:endingPoint]

            solutions += UnderlyingProblem(subset).counterexampleSolution(k = 4)

        return [ s.clearTransducers() for s in solutions ]

    def solve(self, numberOfWorkers = None):
        if numberOfWorkers == None: numberOfWorkers = numberOfCPUs()
        print "# of workers:",numberOfWorkers
        solutions = map(lambda j: self.worker(j), range(numberOfWorkers))#Pool(numberOfWorkers).

        # Now coalesce the rules and figure out how frequently they occurred
        ruleFrequencies = {} # map from the Unicode representation of a rule to (frequency,rule)
        for r in [ r for ss in solutions for s in ss for r in s.rules ]:
            (oldFrequency,_) = ruleFrequencies.get(unicode(r), (0,r))
            ruleFrequencies[unicode(r)] = (oldFrequency + 1, r)

        for f,r in sorted(ruleFrequencies.keys(), key = lambda fr: fr[0]):
            print f,"\t",unicode(r)
        
        # Now coalesce the morphologies and figure out how frequently those occurred
        morphologyFrequencies = {} # map from (frequency, prefixes, suffixes)
        for s in [ s for ss in solutions for s in ss ]:
            k = tuple([ unicode(m) for m in s.prefixes + s.suffixes ])
            (oldFrequency, oldValue) = morphologyFrequencies.get(k, (0,s.prefixes,s.suffixes))
            morphologyFrequencies[k] = (oldFrequency + 1, oldValue)
            
        for f, prefixes, suffixes in sorted(morphologyFrequencies.keys(), key = lambda fr: fr[0]):
            print f
            for p,s in zip(prefixes, suffixes):
                print p,'+stem+',s
            print 
        
