# -*- coding: utf-8 -*-

from features import FeatureBank, tokenize
from rule import Rule
from sketch import *

from problems import alternationProblems, toneProblems

from multiprocessing import Process
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

        # if the conjectured underlying forms never occur in the surface data,
        # then we need to add a constraint saying that it has the expected orientation
        surfacePhonemes = [ t for s in corpus for t in tokenize(s) ]
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
        depth = 1

        Model.Global()

        whichOrientation = flip()
        if self.forceExpected:
            condition(whichOrientation)
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
        totalRuleCost = sum([r.cost() for r in rules ])
        if totalRuleCost/RULETEMPERATURE > self.descriptionLengthSavings:
            print "No net compression!"
            return None
    
        # print "Rule cost: %f"%(totalRuleCost)
        # print "Threshold temperature for this alternation:", (totalRuleCost / self.descriptionLengthSavings)
        # totalRuleCost = sum([r.alternationCost() for r in rules ])
        # print "AlternationRule cost: %f"%(totalRuleCost)
        # print "(alternative) Threshold temperature for this alternation:", (totalRuleCost / self.descriptionLengthSavings)

        
        return rules


def handleProblem(problemIndex, arguments):
    data = toneProblems[0] if problemIndex == 'tone' else alternationProblems[problemIndex - 1]
    print data.description
    for j, alternation in enumerate(data.parameters["alternations"]):
        print "Analyzing alternation:"
        for k in alternation:
            print "\t",k,"\t",alternation[k]
        problem = AlternationProblem(alternation, data.data)
        solutions = problem.topSolutions(arguments.top)
        if arguments.pickle and len(solutions) > 0:
            pickle.dump(solutions, open("pickles/alternation_"+str(problemIndex)+"_"+str(j)+".p","wb"))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Analyze an alternation to determine whether it is valid and how much it compresses the data.')
    parser.add_argument('problem')
    parser.add_argument('-t','--top', default = 1, type = int)
    parser.add_argument('-p','--pickle', action = 'store_true')
    arguments = parser.parse_args()
    if arguments.problem == 'integration':
        problems = ["tone"] + list(range(1,12))
    elif arguments.problem == 'tone':
        problems = ["tone"]
    else:
        problems = [int(arguments.problem)]
    for problemIndex in problems:
        process = Process(target = handleProblem, args = (problemIndex,arguments))
        process.start()
