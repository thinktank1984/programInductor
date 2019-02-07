# -*- coding: utf-8 -*-

from morph import *
from features import FeatureBank, tokenize
from rule import Rule
from sketch import *
from solution import *

from command_server import *
from problems import *

from multiprocessing import Pool
import sys
import pickle
import argparse
import math

RULETEMPERATURE = 3

class AlternationProblem():
    def __init__(self, alternation, corpus):
        self.surfaceToUnderlying = alternation
        self.deepToSurface = dict([(self.surfaceToUnderlying[k],k) for k in self.surfaceToUnderlying ])
        
        # only extract the relevant parts of the corpus for the problem
        corpus = [Morph(w) for w in corpus
                  if [p for p in self.surfaceToUnderlying.values() + self.surfaceToUnderlying.keys()
                      if p in w ] ]
        self.bank = FeatureBank(corpus + alternation.keys() + alternation.values())
        self.surfaceForms = corpus

        self.maximumObservationLength = max([ len(w) for w in corpus ])
        
        def substitute(p):
            if p in self.surfaceToUnderlying:
                return self.surfaceToUnderlying[p]
            return p
        def substituteBackwards(p):
            for k in self.surfaceToUnderlying:
                if self.surfaceToUnderlying[k] == p: return k
            return p
        def applySubstitution(f, w):
            return Morph([ f(p) for p in w.phonemes ])

        self.deepAlternatives = [ (applySubstitution(substitute, w), applySubstitution(substituteBackwards, w))
                                  for w in corpus ]

        # if the conjectured underlying forms never occur in the surface data,
        # then we need to add a constraint saying that it has the expected orientation
        surfacePhonemes = [ t for s in corpus for t in s.phonemes ]
        self.forceExpected = not any([ (p in alternation.values()) for p in surfacePhonemes ])
        print "Force expected orientation?",self.forceExpected

        # How many nats are saved by employing this alternation?'''
        deepPhonemes = [ self.surfaceToUnderlying.get(p,p) for p in surfacePhonemes ]
        savingsPerPhoneme = math.log(len(set(surfacePhonemes))) - math.log(len(set(deepPhonemes)))
        print "Savings per phoneme:",savingsPerPhoneme
        self.descriptionLengthSavings = savingsPerPhoneme*len(surfacePhonemes)
        print "Total savings:",self.descriptionLengthSavings
        
        
        

    def topSolutions(self, k = 10):
        solutions = []

        for _ in range(k):
            newSolution = self.sketchSolution(solutions)
            if newSolution == None: break
            
            solutions.append(newSolution)
        return solutions

    def sketchSolution(self, solutions):
        depth = 2

        Model.Global()

        whichOrientation = flip()
        if self.forceExpected:
            condition(whichOrientation)
        rules = [ Rule.sample() for _ in range(depth)  ]
        minimize(sum([ alternationCost(r) for r in rules ] + \
                     [ ite(ruleDoesNothing(r),Constant(0),Constant(1))
                       for r in rules ]))
        for other, _ in solutions:
            print other
            condition(Or([ alternationEqual(other[j].makeConstant(self.bank), rules[j]) == 0 for j in range(depth) ]))

        for r in rules:
            condition(isDeletionRule(r) == 0)

        for j in range(len(self.surfaceForms)):
            surface = self.surfaceForms[j].makeConstant(self.bank)
            deep1 = self.deepAlternatives[j][0].makeConstant(self.bank)
            deep2 = self.deepAlternatives[j][1].makeConstant(self.bank)
            deep = ite(whichOrientation,
                       deep1,
                       deep2)
            prediction = deep
            for r in rules:
                prediction = applyRule(r, prediction, Constant(0), len(self.surfaceForms[j]))

            auxiliaryCondition(wordEqual(self.surfaceForms[j].makeConstant(self.bank), prediction))

        try:
            output = solveSketch(self.bank, self.maximumObservationLength + 1, alternationProblem = True)
        except SynthesisFailure:
            print "Failed to find a solution"
            return None

        # now minimize the focus and structural change, so that it looks cleaner
        removeSoftConstraints()
        for r in rules:
            condition(alternationEqual(r, Rule.parse(self.bank, output, r).makeConstant(self.bank)))
        minimize(sum([ ruleCost(r) for r in rules ]))
        try:
            output = solveSketch(self.bank, self.maximumObservationLength + 1, alternationProblem = True)
        except SynthesisFailure:
            print "Failed to find a solution with minimal focus on structural change"
            return None   
        print "Solution found using constraint solving:"
        expectedOrientation = parseFlip(output, whichOrientation)
        print "With the expected orientation?",expectedOrientation
        rules = [ Rule.parse(self.bank, output, r) for r in rules ]
        for r in rules: print r
        totalRuleCost = sum([r.cost() for r in rules ])
        if totalRuleCost/RULETEMPERATURE > self.descriptionLengthSavings:
            print "No net compression!"
            return None
    
        # print "Rule cost: %f"%(totalRuleCost)
        # print "Threshold temperature for this alternation:", (totalRuleCost / self.descriptionLengthSavings)
        # totalRuleCost = sum([r.alternationCost() for r in rules ])
        # print "AlternationRule cost: %f"%(totalRuleCost)
        # print "(alternative) Threshold temperature for this alternation:", (totalRuleCost / self.descriptionLengthSavings)

        if expectedOrientation:
            sub = list(self.surfaceToUnderlying.iteritems())
        else:
            sub = [ (v,k) for k,v in self.surfaceToUnderlying.iteritems() ]
        return rules, sub

def handleProblem(problem):
    print problem.description

    compositeSubstitution = []
    allTheRules = []
    
    for j, alternation in enumerate(problem.parameters["alternations"]):
        print "Analyzing alternation:"
        for k in alternation:
            print "\t",k,"\t",alternation[k]
        problem = AlternationProblem(alternation, problem.data)
        solutions = problem.topSolutions(arguments.top)
        if solutions != []:
            allTheRules += solutions[0][0]
            compositeSubstitution += solutions[0][1]
        
    if len(solutions) > 0:
        composite = AlternationSolution(map(Morph,problem.data),dict(compositeSubstitution),allTheRules)
        fn = "experimentOutputs/alternation/"+arguments.problem+".p"
        print "Exporting to",fn
        dumpPickle(composite, fn)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Analyze an alternation to determine whether it is valid and how much it compresses the data.')
    parser.add_argument('problem')
    parser.add_argument('-t','--top', default = 1, type = int)
    parser.add_argument('-m','--cores', default = 1, type = int)
    
    arguments = parser.parse_args()
    start_server(arguments.cores)
    
    try:
        problem = Problem.named[arguments.problem]
    except:
        print("Could not find problem %s"%problem)
        sys.exit(0)            

    handleProblem(problem)
