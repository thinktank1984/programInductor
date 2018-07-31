# -*- coding: utf-8 -*-

from compileRuleToSketch import compileRuleToSketch
from utilities import *
from solution import *
from features import FeatureBank, tokenize
from rule import * # Rule,Guard,FeatureMatrix,EMPTYRULE
from morph import Morph
from sketchSyntax import Expression,makeSketchSkeleton
from sketch import *
from supervised import SupervisedProblem
from latex import latexMatrix
from problems import *

from pathos.multiprocessing import ProcessingPool as Pool
import random
import sys
import pickle
import math
from time import time
import itertools
import copy

        

class AlignmentProblem(object):
    def __init__(self, data):
        self.bank = FeatureBank([ w for l in data for w in l if w != None ] + [u'?',u'*'])
        self.numberOfInflections = len(data[0])
        # wrap the data in Morph objects if it isn't already
        self.data = [ tuple( None if i == None else (i if isinstance(i,Morph) else Morph(tokenize(i)))
                             for i in Lex)
                      for Lex in data ]

        self.maximumObservationLength = max([ len(w) for l in self.data for w in l if w != None ])


    def solveSketch(self, minimizeBound = 31, maximumMorphLength=None):
        if maximumMorphLength is None: maximumMorphLength = self.maximumObservationLength
        return solveSketch(self.bank,
                           # unroll: +1 for extra UR size, +1 for guard buffer
                           self.maximumObservationLength + 2,
                           # maximum morpheme size
                           maximumMorphLength,
                           showSource = False, minimizeBound = minimizeBound)

    def solveAlignment(self):
        Model.Global()
        prefixes = [Morph.sample() for _ in range(self.numberOfInflections) ]
        suffixes = [Morph.sample() for _ in range(self.numberOfInflections) ]
        stems = [Morph.sample() for _ in self.data ]

        for surfaces,stem in zip(self.data, stems):
            for (p,s),x in zip(zip(prefixes, suffixes),surfaces):
                if x is None: continue
                condition(matchPattern(x.makeConstant(self.bank),
                                       concatenate3(p,stem,s)))

        # OBJECTIVE: (# inflections) * (stem lengths) + (# data points) * (affix len)
        # Because we pay for each stem once per inflection,
        # and pay for each affix once per data point
        observationsPerStem = float(sum(s is not None
                                        for ss in self.data
                                        for s in ss )) / len(stems)
        observationsPerAffix = sum( sum(ss[i] is not None
                                        for ss in self.data )
                                    for i in range(self.numberOfInflections) ) \
                                        / float(self.numberOfInflections)
        print "observations per stem",observationsPerStem
        print "observations per affix",observationsPerAffix

        r = observationsPerStem/observationsPerAffix
        if r < 2 and r > 0.5:
            ca = 1
            cs = 1
        elif r >= 2:
            ca = 1
            cs = 2
        elif r <= 0.5:
            ca = 2
            cs = 1
        else: assert False

        print "ca = ",ca
        print "cs = ",cs
            
        minimize(sum((patternCost(p) + patternCost(s)) * ca
                     for j,(p,s) in enumerate(zip(prefixes, suffixes))) + \
                 sum(patternCost(stem) * cs
                     for stem,ss in zip(stems, self.data) ))
        # for m in prefixes + suffixes:
        #     condition(patternCost(m) < 4)

        output = self.solveSketch()
        solution = Solution(rules=[],
                            prefixes=[Morph.parse(self.bank, output, p) for p in prefixes ],
                            suffixes=[Morph.parse(self.bank, output, p) for p in suffixes ],
                            underlyingForms={x: Morph.parse(self.bank, output, s)
                                             for x,s in zip(self.data, stems) })

        print solution
        return solution

    def restrict(self, newData):
        restriction = copy.copy(self)
        restriction.data = [ tuple( None if i == None else (i if isinstance(i,Morph) else Morph(tokenize(i)))
                               for i in Lex)
                             for Lex in newData ]
        return restriction

    def solveStem(self, ss, morphology):
        Model.Global()
        stem = Morph.sample()

        for (p,s),x in zip(zip(morphology.prefixes,
                               morphology.suffixes),
                           ss):
            if x is None: continue

            condition(matchPattern(x.makeConstant(self.bank),
                                   concatenate3(p.makeConstant(self.bank),
                                                stem,
                                                s.makeConstant(self.bank))))

        minimize(patternCost(stem))
        output = self.solveSketch()
        return Morph.parse(self.bank, output, stem)

    def guessMorphology(self, batchSize, numberOfSamples):
        from random import choice
        
        # For each inflection, collect those data points that use that inflection
        inflectionUsers = [ [ss for ss in self.data if ss[j] is not None ]
                            for j in range(self.numberOfInflections) ]
        while True:
            batches = []
            for _ in xrange(numberOfSamples):
                inflection = choice(xrange(self.numberOfInflections))
                batches.append(randomlyPermute(inflectionUsers[inflection])[:batchSize])

            usedInflections = {j
                for b in batches
                for ss in b
                for j,s in enumerate(ss)
                if s is not None}
            if len(usedInflections) == self.numberOfInflections: break

        solutions = []
        histogram = [{} for _ in xrange(self.numberOfInflections) ]
        for b in batches:
            try:
                s = self.restrict(b).solveAlignment()
                for i in xrange(self.numberOfInflections):
                    k = (s.prefixes[i], s.suffixes[i])
                    if any( ss[i] is not None for ss in b ):
                        histogram[i][k] = histogram[i].get(k,0) + 1
            except SynthesisFailure: continue
            
        print(histogram)
        prefixes = []
        suffixes = []
        for h in histogram:
            prefix, suffix = max(h.keys(),key=lambda k: h[k])
            prefixes.append(prefix)
            suffixes.append(suffix)

        s = Solution(rules=[],
                     prefixes=prefixes, suffixes=suffixes,
                     underlyingForms={})
        print "Morphological analysis:"
        print s

        for ss in self.data:
            try:
                s.underlyingForms[ss] = self.solveStem(ss, s)
            except SynthesisFailure: pass

        print "Successfully solved for", len(s.underlyingForms), "/", len(self.data), "underlying forms."

        return s
        

        
                
        
        
        


if __name__ == "__main__":
    from command_server import start_server
    import os
    os.system("mkdir  -p precomputedAlignments")
    start_server(1)

    for i,p in enumerate(MATRIXPROBLEMS):
        if not isinstance(p,Problem): continue
        if p.parameters is not None: continue
        solver = AlignmentProblem(p.data)
        if solver.numberOfInflections == 1: continue
        
        print p.description

        a = solver.guessMorphology(5,5)
        dumpPickle(a, "precomputedAlignments/"+str(i)+".p")
        print
    
