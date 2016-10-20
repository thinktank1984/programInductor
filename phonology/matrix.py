# -*- coding: utf-8 -*-

from features import FeatureBank, tokenize
from rule import Rule
from morph import Morph
from sketch import *

from problems import underlyingProblems

from random import random
import sys

class UnderlyingProblem():
    def __init__(self, problem):
        data = problem.data
        self.data = data
        self.bank = FeatureBank([ w for l in data for w in l  ])

        self.numberOfInflections = len(data[0])
        self.inflectionMatrix = [ [ self.bank.wordToMatrix(i) for i in Lex ] for Lex in data ]

        self.maximumObservationLength = max([ len(tokenize(w)) for l in data for w in l ])

    def sketchSolution(self):
        Model.Global()
        
        depth = 1 if len(sys.argv) < 3 else int(sys.argv[2])
        rules = [ Rule.sample() for _ in range(depth)  ]

        stems = [ Morph.sample() for _ in self.inflectionMatrix ]
        prefixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]
        suffixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]

        def applyRules(d):
            for r in rules: d = applyRule(r,d)
            return d
        surfaces = [ [ applyRules(concatenate3(prefixes[i], stems[l], suffixes[i]))
                   for i in range(self.numberOfInflections) ]
                 for l in range(len(stems)) ]

        for l in range(len(stems)):
            for i in range(self.numberOfInflections):
                condition(wordEqual(makeConstantWord(self.bank, self.data[l][i]),
                                    surfaces[l][i]))

        affixSize = sum([ wordLength(m) for m in  prefixes + suffixes ])
        ruleSize = sum([ ruleCost(r) for r in rules ])

        # Maximize affix size
        maximize(affixSize)
        output = solveSketch(self.bank, self.maximumObservationLength)
        if not output:
            print "Failed at morphological analysis."
            return
        for affix in prefixes + suffixes:
            condition(wordLength(affix) == len(Morph.parse(self.bank, output, affix)))
    
        print "Minimizing rules..."
        minimize(ruleSize)
        output = solveSketch(self.bank, self.maximumObservationLength)
        if not output:
            print "Failed at minimizing rules."
            return
        for r in rules:
            print Rule.parse(self.bank, output, r)
        for i in range(self.numberOfInflections):
            print "Inflection %d:\t"%i,
            print Morph.parse(self.bank, output, prefixes[i]),
            print "+ stem +",
            print Morph.parse(self.bank, output, suffixes[i])
                

data = underlyingProblems[int(sys.argv[1]) - 1]
print data.description
UnderlyingProblem(data).sketchSolution()

