# -*- coding: utf-8 -*-

from utilities import *

from features import FeatureBank, tokenize
from rule import Rule
from morph import Morph
from sketch import *

from problems import underlyingProblems,interactingProblems
from countingProblems import CountingProblem

from random import random
import sys

class UnderlyingProblem():
    def __init__(self, data, depth, bank = None):
        self.depth = depth
        self.data = data
        self.bank = bank if bank != None else FeatureBank([ w for l in data for w in l  ])

        self.numberOfInflections = len(data[0])
        self.inflectionMatrix = [ [ self.bank.wordToMatrix(i) for i in Lex ] for Lex in data ]

        self.maximumObservationLength = max([ len(tokenize(w)) for l in data for w in l ])

    def sortDataByLength(self):
        # Sort the data by length
        data = sorted([ (sum(map(compose(len,tokenize),inflections)), inflections) for inflections in self.data ])
        self.data = [ d[1] for d in data ]


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
        for r in rules:
            condition(isDeletionRule(r) == 0)
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
        # We only care about maximizing the affix size
        # However we also need calculate the rules size, in order to make sure that the next minimization succeeds
        for r in rules:
            condition(ruleCost(r) < 20) # totally arbitrary
        
        output = solveSketch(self.bank, self.maximumObservationLength)
        if not output:
            print "Failed at morphological analysis."
            raise Exception('morphology')
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

    def solveTopRules(self, prefixes, suffixes, k):
        solutions = []
        for _ in range(k):
            Model.Global()

            r = Rule.sample()
            for other in solutions:
                condition(ruleEqual(r, other.makeConstant(self.bank)) == 0)
            stems = [ Morph.sample() for _ in self.inflectionMatrix ]

            # Make the morphology be a global definition
            prefixVariables = [ define("Word", p.makeConstant(self.bank)) for p in prefixes ]
            suffixVariables = [ define("Word", s.makeConstant(self.bank)) for s in suffixes ]

            self.conditionOnData([r], stems, prefixVariables, suffixVariables)
            minimize(ruleCost(r))
            output = solveSketch(self.bank, self.maximumObservationLength)
            if not output:
                print "Found %d rules."%len(solutions)
                break
            solutions.append(Rule.parse(self.bank, output, r))
        return solutions

    def verify(self, prefixes, suffixes, rules, inflections):
        Model.Global()

        stem = Morph.sample()

        # Make the morphology/phonology be a global definition
        prefixes = [ define("Word", p.makeConstant(self.bank)) for p in prefixes ]
        suffixes = [ define("Word", s.makeConstant(self.bank)) for s in suffixes ]
        rules = [ define("Rule", r.makeConstant(self.bank)) for r in rules ]

        self.conditionOnStem(rules, stem, prefixes, suffixes, inflections)

        output = solveSketch(self.bank, self.maximumObservationLength)
        return (output != None)

    @staticmethod
    def showMorphologicalAnalysis(prefixes, suffixes):
        print "Morphological analysis:"
        for i in range(len(prefixes)):
            print "Inflection %d:\t"%i,
            print prefixes[i],
            print "+ stem +",
            print suffixes[i]

    @staticmethod
    def showRules(rules):
        print "Phonological rules:"
        for r in rules: print r

    def sketchSolution(self):
        prefixes, suffixes = self.solveAffixes()
        UnderlyingProblem.showMorphologicalAnalysis(prefixes, suffixes)
        rules = self.solveRules(prefixes, suffixes)
        UnderlyingProblem.showRules(rules)

        return (prefixes, suffixes, rules)

    def topSolutions(self,n = 2,k = 1):
        self.depth = 1
        self.sortDataByLength()
        trainingData = self.data[0:n]
        trainingProblem = UnderlyingProblem(trainingData, 1)

        prefixes, suffixes = trainingProblem.solveAffixes()
        print prefixes
        UnderlyingProblem.showMorphologicalAnalysis(prefixes, suffixes)
        rules = trainingProblem.solveTopRules(prefixes, suffixes, k)
        UnderlyingProblem.showRules(rules)

        return (prefixes, suffixes, rules)

    def counterexampleTop(self):
        trainingData = [ self.data[0] ]

        while True:
            print "Training data:"
            for r in trainingData:
                for i in r: print i,
                print ""
            try:
                (prefixes, suffixes, rules) = UnderlyingProblem(trainingData, 1).sketchSolution()
            except:
                trainingData = trainingData[:-1]
                UnderlyingProblem(trainingData, 1).topSolutions(len(trainingData),10)
                return
            UnderlyingProblem.showMorphologicalAnalysis(prefixes, suffixes)
            UnderlyingProblem.showRules(rules)
            foundCounterexample = False
            for observation in self.data:
                if observation in trainingData: continue
                if not self.verify(prefixes, suffixes, rules, observation):
                    trainingData.append(observation)
                    foundCounterexample = True
                    print "COUNTEREXAMPLE:\t",
                    for i in observation: print i,"\t",
                    print ""
                    break
            if not foundCounterexample:
                print "Final solution:"
                UnderlyingProblem.showMorphologicalAnalysis(prefixes, suffixes)
                UnderlyingProblem.showRules(rules)
                break

            
        
        

    def counterexampleSolution(self):
        self.sortDataByLength()
        # Start out with the shortest 2 examples
        trainingData = [ self.data[0], self.data[1] ]

        while True:
            print "CEGIS: Training data:"
            for r in trainingData:
                for i in r: print i,
                print ""
            (prefixes, suffixes, rules) = UnderlyingProblem(trainingData, self.depth, self.bank).sketchSolution()
            print "Beginning verification"
            foundCounterexample = False
            for observation in self.data:
                if observation in trainingData: continue
                if not self.verify(prefixes, suffixes, rules, observation):
                    trainingData.append(observation)
                    foundCounterexample = True
                    print "COUNTEREXAMPLE:\t",
                    for i in observation: print i,"\t",
                    print ""
                    break
            if not foundCounterexample:
                print "Final solution:"
                UnderlyingProblem.showMorphologicalAnalysis(prefixes, suffixes)
                UnderlyingProblem.showRules(rules)
                break
                
                
                
if __name__ == '__main__':
    useCounterexamples = '-c' in sys.argv
    topSolutions = '-t' in sys.argv
    sys.argv = [a for a in sys.argv if not a.startswith('-') ]
    # Build a "problems" structure, which is a list of (problem, # rules)
    if sys.argv[1] == 'integration':
        problems = [(1,2),
                    (2,1),
                    (3,2),
                    (5,1),
                    (7,1),
                    (8,2),
                    (9,3),
                    # Chapter five problems
                    (51,2),
                    (52,2)]
    else:
        depth = 1 if len(sys.argv) < 3 else int(sys.argv[2])
        problemIndex = int(sys.argv[1])
        problems = [(problemIndex,depth)]
    
    for problemIndex, depth in problems:
        
        p = underlyingProblems[problemIndex - 1] if problemIndex < 10 else interactingProblems[problemIndex - 1 - 50]
        print p.description
        if problemIndex == 7:
            CountingProblem(p.data, p.parameters).sketchSolution()
        elif topSolutions:
            UnderlyingProblem(p.data, depth).counterexampleTop()
        elif useCounterexamples:
            UnderlyingProblem(p.data, depth).counterexampleSolution()
        else:
            (prefixes, suffixes, rules) = UnderlyingProblem(p.data, depth).sketchSolution()
            UnderlyingProblem.showRules(rules)
