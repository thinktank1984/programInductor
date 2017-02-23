# -*- coding: utf-8 -*-

from utilities import *

from features import FeatureBank, tokenize, featureMap
from rule import Rule
from morph import Morph
from sketch import *

from random import choice,seed
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import numpy as np
import argparse

seed(4)

TEMPERATURE = 2.0

def sampleVowel():
    return choice([u"i",u"ɩ",u"e",u"ə",u"ɛ",u"æ",u"a",u"u",u"ʊ",u"o",u"ɔ"])
def sampleConsonant():
    return choice([u"p",u"b",u"f",u"v",u"β",u"m",u"θ",u"d",u"t",u"ð",u"z",u"ǰ",u"s",u"n",u"š",u"k",u"g",u"ŋ",u"h",u"w",u"y",u"r",u"l"])
def sampleSyllable():
    v = sampleVowel()
    k = sampleConsonant()
    return k + v
def sampleAB():
    while True:
        s = sampleSyllable()
        d = sampleSyllable()
        if len(set(tokenize(s))&set(tokenize(d))) == 0:
            return s,d
def sampleABA(n):
    l = []
    for _ in range(n):
        s,d = sampleAB()
        l.append(s + d + s)
    return l
def sampleABB(n):
    l = []
    for _ in range(n):
        s,d = sampleAB()
        l.append(d + s + s)
    return l
def sampleABX(n):
    l = []
    x = sampleConsonant()
    for _ in range(n):
        s,d = sampleAB()
        l.append(s + d + x)
    return l
def sampleAAX(n):
    l = []
    x = sampleConsonant()
    for _ in range(n):
        a = sampleSyllable()
        l.append(a + a + x)
    return l

def topSolutions(depth, observations, k = 1):
    if depth == 0: k = 0 # top k doesn't apply here
    
    bank = FeatureBank([ w for w in observations ])
    maximumObservationLength = max([ len(tokenize(w)) for w in observations ]) + 1

    def conditionOnExample(r, x):
        print x
        u = Morph.sample()
        x = makeConstantWord(bank, x)
        condition(wordEqual(x, applyRules(r, u)))
        return u

    Model.Global()
    rules = [ Rule.sample() for _ in range(depth) ]
    def applyRules(r,x):
        if len(r) == 0: return x
        return applyRules(r[1:], applyRule(r[0], x))

    underlyingForms = [ conditionOnExample(rules,x) for x in observations ]
    cost = sum([ wordLength(u) for u in underlyingForms ])
    minimize(cost)

    output = solveSketch(bank, maximumObservationLength)
    optimalCost = sum([ len(Morph.parse(bank, output, u)) for u in underlyingForms ])
    print "Minimized the cost of the underlying forms: %d"%optimalCost
    removeSoftConstraints()

    condition(cost == optimalCost)
    ruleCostExpression = sum([ ruleCost(r) for r in rules ])
    if len(rules) > 0:
        minimize(ruleCostExpression)

    solutions = []
    solutionCosts = []
    for _ in range(k):
        # Excludes solutions we have already found
        for other,_ in solutions:
            condition(And([ ruleEqual(r,o.makeConstant(bank)) for r,o in zip(rules,other) ]) == 0)
        #because we want a nice front, force it to have a worst set of rules
        # this constraint actually supersedes the previous one
        if len(solutionCosts) > 0:
            condition(ruleCostExpression > min([ rc for rc,uc in solutionCosts ]))
        

        output = solveSketch(bank, maximumObservationLength)

        print "Underlying forms:"
        us = [ Morph.parse(bank, output, u) for u in underlyingForms ]
        print "\n".join(map(str,us))
        rs = [ Rule.parse(bank, output, r) for r in rules ]
        rc = sum([r.cost() for r in rs ])
        uc = sum([len(u) for u in us ])
        print "Rules:"
        print "\n".join(map(str,rs))
        solutions.append((rs,us))
        print "Costs:",(rc,uc)
        solutionCosts.append((rc,uc))

    if len(solutions) > 0:
        optimalCost, optimalSolution = min([(uc + float(rc)/TEMPERATURE, s)
                                            for ((rc,uc),s) in zip(solutionCosts, solutions) ])
        print "Optimal solution:"
        print "\n".join(map(str,optimalSolution[0]))
        print "Optimal cost:",optimalCost
    
    
    return solutions, solutionCosts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Generate and analyze synthetic rule learning problems ala Gary Marcus ABA/ABB patterns')
    parser.add_argument('-p','--problem', default = 'aba', type = str)
    parser.add_argument('-t','--top', default = 1, type = int)
    parser.add_argument('-d','--depth', default = 3, type = int)
    parser.add_argument('-n','--number', default = 4, type = int)
    parser.add_argument('-e','--experiments', default = 1, type = int)
    parser.add_argument('-q','--quiet', action = 'store_true')
    
    arguments = parser.parse_args()

    pointsFromEachExperiment = [] # pareto curve for each experiment

    sampling = {'aba': sampleABA,
                'abb': sampleABB,
                'abx': sampleABX,
                'aax': sampleAAX,
                }
        
    for experiment in range(arguments.experiments):
        print "Experiment %d:"%(1+experiment)
        trainingData = sampling[arguments.problem](arguments.number)
        surfaceLength = sum([len(tokenize(w)) for w in trainingData ])

        points = []
        for d in range(0,arguments.depth + 1):
            solutions, costs = topSolutions(d, trainingData, arguments.top)
            points += costs
        pointsFromEachExperiment.append(points)
        
    print pointsFromEachExperiment
    colors = cm.rainbow(np.linspace(0, 1, len(pointsFromEachExperiment)))
    if not arguments.quiet:
        for points,color in zip(pointsFromEachExperiment,colors):
            plot.scatter([ -p[0] for p in points],
                         [ float(surfaceLength)/p[1] for p in points],
                         alpha = 1.0/(arguments.experiments+2), s = 100, color = color)
        plot.ylabel("Fit to data (Compression ratio)")
        plot.xlabel("Parsimony (-Program length)")
        plot.title("Pareto front for %s, %d example%s"%(arguments.problem,arguments.number,'' if arguments.number == 1 else 's'))
        plot.show()
