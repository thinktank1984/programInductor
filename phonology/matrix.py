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
import pickle

class SynthesisFailure(Exception):
    pass

def sampleMorphWithLength(l):
    m = Morph.sample()
    condition(wordLength(m) == l)
    return m

class UnderlyingProblem():
    def __init__(self, data, depth, bank = None):
        self.depth = depth
        self.data = data
        self.bank = bank if bank != None else FeatureBank([ w for l in data for w in l  ])

        self.numberOfInflections = len(data[0])
        self.inflectionMatrix = [ [ self.bank.wordToMatrix(i) for i in Lex ] for Lex in data ]

        self.maximumObservationLength = max([ len(tokenize(w)) for l in data for w in l ])


    def sortDataByLength(self):
        # Sort the data by length. Break ties by remembering which one originally came first.
        dataTaggedWithLength = [ (sum(map(compose(len,tokenize),self.data[j])), j, self.data[j])
                                 for j in range(len(self.data)) ]
        self.data = [ d[2] for d in sorted(dataTaggedWithLength) ]


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
        '''Conditions on inflection matrix. This also modifies the rules in place! Always call this after calculating the cost of the rules.'''
        for r in rules:
            condition(isDeletionRule(r) == 0)
            condition(fixStructuralChange(r))
        for i in range(len(stems)):
            self.conditionOnStem(rules, stems[i], prefixes, suffixes, self.data[i])
    
    def solveAffixes(self, oldRules = []):
        Model.Global()
        
        numberOfNewRules = self.depth - len(oldRules)
        rules = [ define("Rule", r.makeConstant(self.bank)) for r in oldRules ]
        rules += [ Rule.sample() for _ in range(numberOfNewRules) ]
        stems = [ Morph.sample() for _ in self.inflectionMatrix ]
        prefixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]
        suffixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]

        # We only care about maximizing the affix size
        # However we also need calculate the rules size, in order to make sure that the next minimization succeeds
        for r in rules:
            condition(ruleCost(r) < 20) # totally arbitrary

        self.conditionOnData(rules, stems, prefixes, suffixes)
        
        affixSize = sum([ wordLength(m) for m in  prefixes + suffixes ])
        maximize(affixSize)

        output = solveSketch(self.bank, self.maximumObservationLength)
        if not output:
            raise SynthesisFailure("Failed at morphological analysis.")
        prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ]
        suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ]
        return (prefixes, suffixes)

    def solveRules(self, prefixes, suffixes, oldRules = []):
        Model.Global()

        numberOfNewRules = self.depth - len(oldRules)
        rules = [ define("Rule", r.makeConstant(self.bank)) for r in oldRules ]
        rules += [ Rule.sample() for _ in range(numberOfNewRules) ]
        stems = [ Morph.sample() for _ in self.inflectionMatrix ]

        # Make the morphology be a global definition
        prefixes = [ sampleMorphWithLength(len(p)) for p in prefixes ]
        suffixes = [ sampleMorphWithLength(len(s)) for s in suffixes ]

        minimize(sum([ ruleCost(r) for r in rules ]))
        self.conditionOnData(rules, stems, prefixes, suffixes)

        output = solveSketch(self.bank, self.maximumObservationLength)
        if not output:
            raise SynthesisFailure("Failed at phonological analysis.")

        rules = oldRules + [ Rule.parse(self.bank, output, r) for r in rules[len(oldRules):] ]
        prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ]
        suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ]
        return (rules, prefixes, suffixes)

    def solveUnderlyingForms(self, prefixes, suffixes, rules):
        Model.Global()
        rules = [ define("Rule", r.makeConstant(self.bank)) for r in rules ]
        prefixes = [ define("Word", p.makeConstant(self.bank)) for p in prefixes ]
        suffixes = [ define("Word", s.makeConstant(self.bank)) for s in suffixes ]
        stems = [ Morph.sample() for _ in self.inflectionMatrix ]
        self.conditionOnData(rules, stems, prefixes, suffixes)

        for stem in stems:
            minimize(wordLength(stem))
        
        output = solveSketch(self.bank, self.maximumObservationLength)
        if not output:
            raise SynthesisFailure("Failed at underlying form analysis.")

        return [ Morph.parse(self.bank, output, s) for s in stems ]

    def solveTopRules(self, prefixes, suffixes, underlyingForms, k):
        solutions = []
        
        for _ in range(k):
            Model.Global()

            rules = [ Rule.sample() for _ in range(self.depth) ]
            for other in solutions:
                condition(And([ ruleEqual(r, o.makeConstant(self.bank))
                                for r, o in zip(rules, other) ]) == 0)

            stems = [ define("Word", u.makeConstant(self.bank)) for u in underlyingForms ]

            # Make the morphology be a global definition
            prefixVariables = [ define("Word", p.makeConstant(self.bank)) for p in prefixes ]
            suffixVariables = [ define("Word", s.makeConstant(self.bank)) for s in suffixes ]

            self.conditionOnData(rules, stems, prefixVariables, suffixVariables)
            minimize(sum([ ruleCost(r) for r in rules ]))
            output = solveSketch(self.bank, self.maximumObservationLength)
            if not output:
                print "Found %d rules."%len(solutions)
                break
            solutions.append([ Rule.parse(self.bank, output, r) for r in rules ])
        return solutions

    def findCounterexample(self, prefixes, suffixes, rules, trainingData = []):
        print "Beginning verification"
        for observation in self.data:
            if observation in trainingData: continue
            if not self.verify(prefixes, suffixes, rules, observation):
                print "COUNTEREXAMPLE:\t",
                for i in observation: print i,"\t",
                print ""
                return observation

        return None


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

    def sketchSolution(self, oldRules = [], canAddNewRules = False):
        try:
            prefixes, suffixes = self.solveAffixes(oldRules)
            rules, prefixes, suffixes = self.solveRules(prefixes, suffixes, oldRules)
            UnderlyingProblem.showMorphologicalAnalysis(prefixes, suffixes)
            UnderlyingProblem.showRules(rules)
            return (prefixes, suffixes, rules)
        except SynthesisFailure:
            if canAddNewRules:
                self.depth += 1
                print "Expanding rule depth to %d"%self.depth
                return self.sketchSolution(oldRules = oldRules, canAddNewRules = canAddNewRules)
            else:
                return None
            
    def topSolutions(self, k=10):
        prefixes, suffixes, rules = self.sketchSolution(canAddNewRules = True)
        underlyingForms = self.solveUnderlyingForms(prefixes, suffixes, rules)
        print "Showing all plausible short rules with this morphological analysis:"
        solutions = self.solveTopRules(prefixes, suffixes, underlyingForms, k)
        for s in solutions:
            print "\n".join(map(str,s))
            print "............................."
        return solutions

    def counterexampleSolution(self):
        self.sortDataByLength()
        # Start out with the shortest 2 examples
        trainingData = [ self.data[0], self.data[1] ]

        while True:
            print "CEGIS: Training data:"
            for r in trainingData:
                for i in r: print i,
                print ""
            # expand the rule set until we can fit the training data
            (prefixes, suffixes, rules) = UnderlyingProblem(trainingData, self.depth, self.bank).sketchSolution(canAddNewRules = True)

            c = self.findCounterexample(prefixes, suffixes, rules, trainingData)
            if c == None:
                print "Final solution:"
                UnderlyingProblem.showMorphologicalAnalysis(prefixes, suffixes)
                UnderlyingProblem.showRules(rules)
                break
            else:
                trainingData.append(c)
        return prefixes, suffixes, rules, trainingData
                
def incrementalSynthesis(data):
    '''Incrementally grow a program rule by rule, while growing a training set Lexeme by Lexeme'''
    pass
    '''    trainingSet = [data[0]]
    bank = FeatureBank([ w for l in data for w in l  ])

    # what do we learn from the first example?
    prefixes, suffixes, rules = UnderlyingProblem(trainingSet, 1, bank).sketchSolution()
    rules = [r for r in rules if not r.doesNothing() ]

    foundCounterexample = True
    while foundCounterexample:
        foundCounterexample = False
        # find a counterexample
        for testingExample in [ e for e in data if not e in trainingSet ]:
            if UnderlyingProblem([testingExample],1,bank).verify(prefixes, suffixes, rules, testingExample):
                continue # consistent with the current model
            foundCounterexample = True
            trainingSet.append(testingExample)

            # Expand to include another rule but still use all of the old rules
            print "Current training examples:"
            for e in trainingSet:
                print "\t".join(e)
            prefixes, suffixes, rules = UnderlyingProblem(trainingSet, len(rules) + 1, bank).sketchSolution(rules)
            rules = [r for r in rules if not r.doesNothing() ]
            break
    
    print "Finished incremental synthesis."
    UnderlyingProblem.showRules(rules)
'''

            
    

def greedySynthesis(data):
    '''Incrementally grow program rule by rule, trying to the rule that covers as much data as possible'''
    pass
'''    trainingSet = [data[0]]
    bank = FeatureBank([ w for l in data for w in l  ])
    blacklist = [] # inconsistent with the training set - ignore these because we'll never get them

    # what do we learn from the first example?
    prefixes, suffixes, rules = UnderlyingProblem(trainingSet, 1, bank).sketchSolution()

    for example in data:
        if example in trainingSet: continue
        if example in blacklist: continue
        renderedExample = u"|".join(example)
        if UnderlyingProblem([example],1,bank).verify(prefixes, suffixes, rules, example):
            print "%s is consistent with the model."%renderedExample
            continue # consistent with the current model
        try: # try to revise the current model
            prefixes, suffixes, rules = UnderlyingProblem(trainingSet + [example], 1, bank).sketchSolution()
            trainingSet.append(example)
            print "%s updated the model."%renderedExample
        except SynthesisFailure: # inconsistent
            blacklist.append(example)
            print "%s is inconsistent with the rest of the training data."%renderedExample
    print "Greedy final answer:"
    UnderlyingProblem.showRules(rules)
'''
        
    
    
                        
                
if __name__ == '__main__':
    useCounterexamples = '-c' in sys.argv
    topSolutions = '-t' in sys.argv
    sys.argv = [a for a in sys.argv if not a.startswith('-') ]
    # Build a "problems" structure, which is a list of (problem, # rules)
    if sys.argv[1] == 'integration':
        problems = [(1,2),
                    (2,1),
                    (3,2),
#                    (4,3),
                    (5,1),
                    (7,1),
                    (8,2),
#                    (9,3),
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
            UnderlyingProblem(p.data, depth).topSolutions(20)
        elif useCounterexamples:
            UnderlyingProblem(p.data, depth).counterexampleSolution()
        else:
            (prefixes, suffixes, rules) = UnderlyingProblem(p.data, depth).sketchSolution()
            UnderlyingProblem.showRules(rules)
