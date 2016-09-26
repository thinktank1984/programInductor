# -*- coding: utf-8 -*-

from features import featureMap, tokenize
from rule import Rule
from sketch import *

from problems import alternationProblems

from random import random
import sys

class AlternationProblem():
    def __init__(self, problem):
        surfaceToUnderlying = problem.parameters["alternations"]
        corpus = problem.data
        self.surfaceFeatures = [ [ featureMap[t] for t in tokenize(w) ] for w in corpus ]
        self.surfaceMatrices = [ [ featureVectorMap[t] for t in tokenize(w) ] for w in corpus ]
        
        def substitute(p):
            if p in surfaceToUnderlying:
                return surfaceToUnderlying[p]
            return p
        def substituteBackwards(p):
            for k in surfaceToUnderlying:
                if surfaceToUnderlying[k] == p: return k
            return p
        
        self.underlyingFeatures = [ [ featureMap[substitute(t)] for t in tokenize(w) ] for w in corpus ]
        self.underlyingMatrices = [ [ featureVectorMap[substitute(t)] for t in tokenize(w) ] for w in corpus ]
        self.underlyingMatricesAlternative = [ [ featureVectorMap[substituteBackwards(t)] for t in tokenize(w) ] for w in corpus ]
        self.rule = Rule()

    def scoreSolution(self):
        errors = 0
        for i in range(len(self.surfaceFeatures)):
            proposedSolution = self.rule.applyRule(self.underlyingFeatures[i])
            actualSolution = self.surfaceFeatures[i]
            for j in range(max(len(proposedSolution),len(actualSolution))):
                proposedSound = []
                if j < len(proposedSolution): proposedSound = proposedSolution[j]
                actualSound = []
                if j < len(actualSolution): actualSound = actualSolution[j]
                difference = len(set(proposedSound) ^ set(actualSound))
                if difference > 0:
                    errors += difference
        return errors
    
    def sketchSolution(self):
        depth = 1 if len(sys.argv) < 3 else int(sys.argv[2])
        
        whichOrientation = flip()
        rules = [ Rule.sample() for _ in range(depth)  ]
        
        for j in range(len(self.surfaceMatrices)):
            surface = makeConstantWordOfMatrix(self.surfaceMatrices[j])
            deep1 = makeConstantWordOfMatrix(self.underlyingMatrices[j])
            deep2 = makeConstantWordOfMatrix(self.underlyingMatricesAlternative[j])
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
        
        output = solveSketch()

        print "Solution found using constraint solving:"
        print "With the expected orientation?",parseFlip(output, whichOrientation)
        for r in rules:
            print Rule.parse(output, r)
        
data = alternationProblems[int(sys.argv[1]) - 1]
print data.description
problem = AlternationProblem(data)
problem.sketchSolution()

'''
print "Trying to solve using stochastic search!"
def stochasticSearch(problem, populationSize, branchingFactor, numberOfIterations):
    seed = Rule(set([]),
                set([]),
                [set([])],
                [set([])])
    problem.rule = seed
    print "Initial score:",problem.scoreSolution()
    population = [seed.mutate() for _ in range(populationSize) ]

    for iteration in range(numberOfIterations):
        print "Iteration",iteration

        population = [r.mutate() for r in population for _ in range(branchingFactor) ]
        scores = []
        for j in range(len(population)):
            problem.rule = population[j]
            scores.append(problem.scoreSolution() + random())
        population = sorted(zip(scores,population))[:populationSize]
        print "Best:"
        print population[0][1]
        print population[0][0]
        population = [p[1] for p in population ]
            
stochasticSearch(problem,500,50,10)
'''
