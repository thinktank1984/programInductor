# -*- coding: utf-8 -*-

from utilities import *

from features import FeatureBank, tokenize, featureMap
from rule import Rule
from morph import Morph
from sketch import *
from matrix import UnderlyingProblem

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
    return k + v + u"-"
def sampleAB():
    while True:
        s = sampleSyllable()
        d = sampleSyllable()
        if len(set(tokenize(s))&set(tokenize(d))) == 1: # they should have exactly one thing in common: syllable boundary
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
    x = sampleSyllable()
    for _ in range(n):
        s,d = sampleAB()
        l.append(s + d + x)
    return l
def sampleAAX(n):
    l = []
    x = sampleSyllable()
    for _ in range(n):
        a = sampleSyllable()
        l.append(a + a + x)
    return l

def removePointsNotOnFront(points):
    points = list(set(points))

    toRemove = []
    for p in points:
        for q in points:
            if q[0] < p[0] and q[1] < p[1]:
                toRemove.append(p)
    return [ p for p in points if not p in toRemove ]


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
        print u"\n".join(trainingData)
        surfaceLength = sum([len(tokenize(w)) for w in trainingData ])

        points = []
        for d in range(0,arguments.depth + 1):
            worker = UnderlyingProblem([(w,) for w in trainingData ],d)
            solutions, costs = worker.paretoFront(arguments.top, TEMPERATURE)
            points += costs
        pointsFromEachExperiment.append(removePointsNotOnFront(points))
        
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
