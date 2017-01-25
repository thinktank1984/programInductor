# -*- coding: utf-8 -*-

from features import FeatureBank, tokenize
from rule import Rule
from sketch import *

from problems import alternationProblems

from random import random
import sys
import pickle

class AlternationProblem():
    def __init__(self, alternation, corpus):
        self.surfaceToUnderlying = alternation
        self.deepToSurface = dict([(self.surfaceToUnderlying[k],k) for k in self.surfaceToUnderlying ])
        
        # only extract the relevant parts of the corpus for the problem
        corpus = [w for w in corpus
                  if [p for p in self.surfaceToUnderlying.values() + self.surfaceToUnderlying.keys()
                      if p in w ] ]
        self.bank = FeatureBank(corpus + alternation.keys() + alternation.values())
        self.surfaceForms = corpus

        self.maximumObservationLength = max([ len(tokenize(w)) for w in corpus ])
        
        def substitute(p):
            if p in self.surfaceToUnderlying:
                return self.surfaceToUnderlying[p]
            return p
        def substituteBackwards(p):
            for k in self.surfaceToUnderlying:
                if self.surfaceToUnderlying[k] == p: return k
            return p
        def applySubstitution(f, w):
            return "".join([ f(p) for p in tokenize(w) ])

        self.deepAlternatives = [ (applySubstitution(substitute, w), applySubstitution(substituteBackwards, w))
                                  for w in corpus ]
        

    def scoreSolution(self):
        errors = 0
        for i in range(len(self.surfaceFeatures)):
            proposedSolution = self.rule.applyRule(self.underlyingFeatures[i])
            actualSolution = self.surfaceFeatures[i]
            for j in range(max(len(proposedSolution),len(actualSolution))):
                proposedSound = [k]
                if j < len(proposedSolution): proposedSound = proposedSolution[j]
                actualSound = []
                if j < len(actualSolution): actualSound = actualSolution[j]
                difference = len(set(proposedSound) ^ set(actualSound))
                if difference > 0:
                    errors += difference
        return errors

    
        
    def sketchSolution(self, k = 10):
        depth = 1 if len(sys.argv) < 3 else int(sys.argv[2])
        solutions = []

        for _ in range(k):
            Model.Global()
                
            whichOrientation = flip()
            rules = [ Rule.sample() for _ in range(depth)  ]
            minimize(sum([ alternationCost(r) for r in rules ]))
            for other in solutions:
                condition(Or([ alternationEqual(other[j].makeConstant(self.bank), rules[j]) == 0 for j in range(depth) ]))
            
            for r in rules:
                condition(isDeletionRule(r) == 0)
                condition(fixStructuralChange(r))
        
            for j in range(len(self.surfaceForms)):
                surface = makeConstantWord(self.bank, self.surfaceForms[j])
                deep1 = makeConstantWord(self.bank, self.deepAlternatives[j][0])
                deep2 = makeConstantWord(self.bank, self.deepAlternatives[j][1])
                deep = ite(whichOrientation,
                           deep1,
                           deep2)
                prediction = deep
                for r in rules:
                    prediction = applyRule(r, prediction)

                condition(wordEqual(surface, prediction))

            output = solveSketch(self.bank, self.maximumObservationLength)
            if output:        
                print "Solution found using constraint solving:"
                print "With the expected orientation?",parseFlip(output, whichOrientation)
                rules = [ Rule.parse(self.bank, output, r) for r in rules ]
                for r in rules: print r
                solutions.append(rules)
            else:
                print "Failed to find a solution"
                break
        return solutions

if __name__ == '__main__':
#    setTemporarySketchName("testAlternation.sk")
    if sys.argv[1] == 'integration':
        problems = list(range(1,12))
    else:
        problems = [int(sys.argv[1])]
    for problemIndex in problems:
        data = alternationProblems[problemIndex - 1]
        print data.description
        for alternation in data.parameters["alternations"]:
            print "Analyzing alternation:"
            for k in alternation:
                print "\t",k,"\t",alternation[k]
            problem = AlternationProblem(alternation, data.data)
            solutions = problem.sketchSolution()
            pickle.dump(solutions, open("pickles/alternation_"+str(problemIndex)+".p","wb"))



