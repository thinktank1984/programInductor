# -*- coding: utf-8 -*-

from utilities import *

from features import FeatureBank, tokenize
from rule import Rule
from morph import Morph
from sketch import *
from supervised import solveTopSupervisedRules

from problems import underlyingProblems,interactingProblems
from countingProblems import CountingProblem

from random import random
import sys
import pickle
import argparse

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
        self.maximumMorphLength = max(9,self.maximumObservationLength - 2)


    def sortDataByLength(self):
        # Sort the data by length. Break ties by remembering which one originally came first.
        dataTaggedWithLength = [ (sum(map(compose(len,tokenize),self.data[j])), j, self.data[j])
                                 for j in range(len(self.data)) ]
        self.data = [ d[2] for d in sorted(dataTaggedWithLength) ]


    def conditionOnStem(self, rules, stem, prefixes, suffixes, surfaces):
        def applyRules(d):
            for r in rules: d = applyRule(r,d)
            return d
        def buildUnderlyingForm(prefix, suffix):
            if isinstance(stem, Morph): # underlying form is fixed
                return (prefix + stem + suffix).makeConstant(self.bank)
            else: # underlying form is unknown
                return concatenate3(prefix, stem, suffix)
            
        prediction = [ applyRules(buildUnderlyingForm(prefixes[i],suffixes[i]))
                     for i in range(self.numberOfInflections) ]
        for i in range(self.numberOfInflections):
            condition(wordEqual(makeConstantWord(self.bank, surfaces[i]),
                                prediction[i]))
    
    def conditionOnData(self, rules, stems, prefixes, suffixes):
        '''Conditions on inflection matrix. This also modifies the rules in place! Always call this after calculating the cost of the rules.'''
        for r in rules:
            if len(rules) > 1: # optimizing heuristic
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

       
        affixSize = sum([ wordLength(m) for m in prefixes + suffixes ])
        stemSize = sum([ wordLength(m) for m in stems ])
        if len(self.data) > self.numberOfInflections:
            maximize(affixSize)
        elif len(self.data) < self.numberOfInflections:
            maximize(stemSize)
        else: # indifferent as to the morphologies so just optimize the rules
            minimize(sum([ruleCost(r) for r in rules ]))

        self.conditionOnData(rules, stems, prefixes, suffixes)

        output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
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

        output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
        if not output:
            raise SynthesisFailure("Failed at phonological analysis.")

        rules = oldRules + [ Rule.parse(self.bank, output, r) for r in rules[len(oldRules):] ]
        prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ]
        suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ]
        return (rules, prefixes, suffixes)

    def solveUnderlyingForms(self, prefixes, suffixes, rules):
        Model.Global()
        rules_ = [ define("Rule", r.makeConstant(self.bank)) for r in rules ]
        prefixes_ = [ define("Word", p.makeConstant(self.bank)) for p in prefixes ]
        suffixes_ = [ define("Word", s.makeConstant(self.bank)) for s in suffixes ]
        stems = [ Morph.sample() for _ in self.inflectionMatrix ]
        self.conditionOnData(rules_, stems, prefixes_, suffixes_)

        for stem in stems:
            minimize(wordLength(stem))
        
        output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
        if not output:
            raise SynthesisFailure("Failed at underlying form analysis.")

        us = [ Morph.parse(self.bank, output, s) for s in stems ]

        for j in range(len(self.inflectionMatrix)):
            for i in range(self.numberOfInflections):
                u = prefixes[i] + us[j] + suffixes[i]
                for r in rules:
                    print u
                    print "\t > ",r
                    u = r.apply(u)
                print Morph.fromMatrix(u)
                print Morph(tokenize(self.data[j][i]))
                assert Morph(tokenize(self.data[j][i])) == Morph.fromMatrix(u)
                print "\n"
            print "\n\n"

        return us

    def fastTopRules(self, prefixes, suffixes, underlyingForms, k, existingRules):
        k = int(math.ceil((float(k)**(1.0/depth))))

        inputs = [ prefixes[i] + underlyingForms[j] + suffixes[i]
                   for j in range(len(self.data))
                   for i in range(self.numberOfInflections) ]

        def f(xs, rs):
            if rs == []: return [[]]
            ys = [ rs[0].apply(x) for x in xs ]
            alternatives = solveTopSupervisedRules(zip(xs,ys), k, r[0])
            suffixes = f(ys, rs[1:])
            return [ [a] + s
                     for a in alternatives
                     for s in suffixes ]

        return f(inputs, existingRules)
        

    def solveTopRules(self, prefixes, suffixes, underlyingForms, k, existingRules = None):
        solutions = [] if existingRules == None else [existingRules]
        
        for _ in range(k - (1 if existingRules else 0)):
            Model.Global()

            rules = [ Rule.sample() for _ in range(self.depth) ]
            for other in solutions:
                condition(And([ ruleEqual(r, o.makeConstant(self.bank))
                                for r, o in zip(rules, other) ]) == 0)

            minimize(sum([ ruleCost(r) for r in rules ]))
            self.conditionOnData(rules, underlyingForms, prefixes, suffixes)
            output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
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

        for r in rules: condition(fixStructuralChange(r))

        self.conditionOnStem(rules, stem, prefixes, suffixes, inflections)

        output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
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
        solutions = self.solveTopRules(prefixes, suffixes, underlyingForms, k, existingRules = rules)
        for s in solutions:
            print "\n".join(map(str,s))
            print "............................."
        return prefixes, suffixes, solutions

    def counterexampleSolution(self, k = 1):
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
            self.depth = len(rules) # update depth because it might have grown

            counterexample = self.findCounterexample(prefixes, suffixes, rules, trainingData)
            if counterexample == None: # we found a solution that had no counterexamples
                print "Final solutions:"
                UnderlyingProblem.showMorphologicalAnalysis(prefixes, suffixes)
                underlyingForms = self.solveUnderlyingForms(prefixes, suffixes, rules)
                solutions = self.solveTopRules(prefixes, suffixes, underlyingForms, k, existingRules = rules)
                for s in solutions:
                    UnderlyingProblem.showRules(s)
                    print " ==  ==  == "
                return prefixes, suffixes, solutions                
            else:
                trainingData.append(counterexample)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Solve jointly for morphology and phonology given surface inflected forms of lexemes')
    parser.add_argument('problem')
    parser.add_argument('-c','--counterexamples', action = 'store_true')
    parser.add_argument('-t','--top', default = 1, type = int)
    arguments = parser.parse_args()
    if arguments.problem == 'integration':
        problems = [1,
                    2,
                    3,
                    #4,
                    5,
                    6,
                    7,
                    8,
                    #9,
                    # Chapter five problems
                    51,
                    52]
    else:
        problemIndex = int(arguments.problem)
        problems = [problemIndex]
    
    for problemIndex in problems:
        p = underlyingProblems[problemIndex - 1] if problemIndex < 10 else interactingProblems[problemIndex - 1 - 50]
        print p.description
        ss = None # solutions to save out to the pickled file
        if problemIndex == 7:
            ss = CountingProblem(p.data, p.parameters).topSolutions(arguments.top)
        elif not arguments.counterexamples:
            _,_,ss = UnderlyingProblem(p.data, 1).topSolutions(arguments.top)
        elif arguments.counterexamples:
            _,_,ss = UnderlyingProblem(p.data, 1).counterexampleSolution(arguments.top)
        if ss != None:
            pickle.dump(ss, open("pickles/matrix_"+str(problemIndex)+".p","wb"))
