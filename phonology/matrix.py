# -*- coding: utf-8 -*-

from features import FeatureBank, tokenize
from rule import Rule
from morph import Morph
from sketch import *

from problems import underlyingProblems

from random import random
import sys

class UnderlyingProblem():
    def __init__(self, problem, depth):
        data = problem.data
        self.depth = depth
        self.data = data
        self.bank = FeatureBank([ w for l in data for w in l  ])

        self.numberOfInflections = len(data[0])
        self.inflectionMatrix = [ [ self.bank.wordToMatrix(i) for i in Lex ] for Lex in data ]

        self.maximumObservationLength = max([ len(tokenize(w)) for l in data for w in l ])

    def conditionOnStem(self, rules, stem, prefixes, suffixes, surfaces):
        def applyRules(d):
            for r in rules: d = applyRule(r,d)
            return d
        prediction = [ applyRules(concatenate3(prefixes[i],stem,suffixes[i]))
                     for i in range(self.numberOfInflections) ]
        for i in range(self.numberOfInflections):
            condition(wordEqual(makeConstantWord(self.bank, surfaces[i]),
                                prediction[i]))
    
    def conditionOnData(self, rules, stems, prefixes, suffixes):
        for i in range(len(stems)):
            self.conditionOnStem(rules, stems[i], prefixes, suffixes, self.data[i])
    
    def solveAffixes(self):
        Model.Global()
        
        rules = [ Rule.sample() for _ in range(self.depth) ]
        stems = [ Morph.sample() for _ in self.inflectionMatrix ]
        prefixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]
        suffixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]

        self.conditionOnData(rules, stems, prefixes, suffixes)
        
        affixSize = sum([ wordLength(m) for m in  prefixes + suffixes ])
        maximize(affixSize)
        output = solveSketch(self.bank, self.maximumObservationLength)
        if not output:
            print "Failed at morphological analysis."
            assert False
        prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ]
        suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ]
        return (prefixes, suffixes)

    def solveRules(self, prefixes, suffixes):
        Model.Global()

        rules = [ Rule.sample() for _ in range(self.depth) ]
        stems = [ Morph.sample() for _ in self.inflectionMatrix ]

        # Make the morphology be a global definition
        prefixes = [ define("Word", p.makeConstant(self.bank)) for p in prefixes ]
        suffixes = [ define("Word", s.makeConstant(self.bank)) for s in suffixes ]

        self.conditionOnData(rules, stems, prefixes, suffixes)

        minimize(sum([ ruleCost(r) for r in rules ]))
        output = solveSketch(self.bank, self.maximumObservationLength)
        if not output:
            print "Failed at phonological analysis."
            assert False
        return [ Rule.parse(self.bank, output, r) for r in rules ]

    def verify(self, prefixes, suffixes, rules, inflections):
        Model.Global()

        stem = Morph.sample()

        # Make the morphology/phonology be a global definition
        prefixes = [ define("Word", p.makeConstant(self.bank)) for p in prefixes ]
        suffixes = [ define("Word", s.makeConstant(self.bank)) for s in suffixes ]
        rules = [ define("Rule", r.makeConstant(self.bank)) for r in rules ]

        self.conditionOnStem(rules, stem, prefixes, suffixes, inflections)

        output = solveSketch(self.bank, self.maximumObservationLength)
        if not output:
            print "Failed to verify",inflections
        else:
            print "Verified: stem =",Morph.parse(self.bank, output, stem)
        
    def sketchSolution(self):
        prefixes, suffixes = self.solveAffixes()

        print "Morphological analysis:"
        
        for i in range(self.numberOfInflections):
            print "Inflection %d:\t"%i,
            print prefixes[i],
            print "+ stem +",
            print suffixes[i]

        rules = self.solveRules(prefixes, suffixes)

        print "Phonological rules:"
        for r in rules: print r

        print "Beginning verification"
        for observation in self.data:
            self.verify(prefixes, suffixes, rules, observation)
        
                
if __name__ == '__main__':
    # Build a "problems" structure, which is a list of (problem, # rules)
    if sys.argv[1] == 'integration':
        problems = [(1,2),
                    (2,1),
                    (3,2),
                    (5,1)]
    else:
        depth = 1 if len(sys.argv) < 3 else int(sys.argv[2])
        problemIndex = int(sys.argv[1])
        problems = [(problemIndex,depth)]
    
    for problemIndex, depth in problems:
        data = underlyingProblems[problemIndex - 1]
        print data.description
        UnderlyingProblem(data, depth).sketchSolution()

