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
        

    def topSolutions(self, k = 10):
        solutions = []

        for _ in range(k):
            newSolution = self.sketchSolution(solutions)
            if newSolution == None: break
            
            solutions.append(newSolution)
        return solutions

        
    def sketchSolution(self, solutions):
        depth = 1 if len(sys.argv) < 3 else int(sys.argv[2])

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

        output = solveSketch(self.bank, self.maximumObservationLength, alternationProblem = True)
        if not output:
            print "Failed to find a solution"
            return None

        # now minimize the focus and structural change, so that it looks cleaner
        removeSoftConstraints()
        for r in rules:
            condition(alternationEqual(r, Rule.parse(self.bank, output, r).makeConstant(self.bank)))
        minimize(sum([ ruleCost(r) for r in rules ]))
        output = solveSketch(self.bank, self.maximumObservationLength, alternationProblem = True)
        if not output:
            print "Failed to find a solution with minimal focus on structural change"
            return None   
        print "Solution found using constraint solving:"
        print "With the expected orientation?",parseFlip(output, whichOrientation)
        rules = [ Rule.parse(self.bank, output, r) for r in rules ]
        for r in rules: print r
        return rules


if __name__ == '__main__':
#    setTemporarySketchName("testAlternation.sk")
    if sys.argv[1] == 'integration':
        problems = list(range(1,12))
    else:
        problems = [int(sys.argv[1])]
    for problemIndex in problems:
        data = alternationProblems[problemIndex - 1]
        print data.description
        for j, alternation in enumerate(data.parameters["alternations"]):
            print "Analyzing alternation:"
            for k in alternation:
                print "\t",k,"\t",alternation[k]
            problem = AlternationProblem(alternation, data.data)
            solutions = problem.topSolutions()
            if len(solutions) > 0:
                pickle.dump(solutions, open("pickles/alternation_"+str(problemIndex)+"_"+str(j)+".p","wb"))
            



