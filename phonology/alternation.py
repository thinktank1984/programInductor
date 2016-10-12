# -*- coding: utf-8 -*-

from features import FeatureBank, tokenize
from rule import Rule
from sketch import *

from problems import alternationProblems

from random import random
import sys

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

    
        
    def sketchSolution(self):
        Model.Global()
        
        depth = 1 if len(sys.argv) < 3 else int(sys.argv[2])
        
        whichOrientation = flip()
        rules = [ Rule.sample() for _ in range(depth)  ]
        
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

        cost = alternationCost(rules[0])
        for r in rules[1:]:
            cost = cost + alternationCost(r)
        minimize(cost)

        output = solveSketch(self.bank)
        
        print "Solution found using constraint solving:"
        print "With the expected orientation?",parseFlip(output, whichOrientation)
        for r in rules:
            print Rule.parse(self.bank, output, r)
        
data = alternationProblems[int(sys.argv[1]) - 1]
print data.description
for alternation in data.parameters["alternations"]:
    print "Analyzing alternation:"
    for k in alternation:
        print "\t",k,"\t",alternation[k]
    problem = AlternationProblem(alternation, data.data)
    problem.sketchSolution()



