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
    return sorted([ (x,y) for x,y in points
                    if not any([ a >= x and b >= y and (a,b) != (x,y)
                                 for a,b in points ])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Generate and analyze synthetic rule learning problems ala Gary Marcus ABA/ABB patterns')
    parser.add_argument('-p','--problem', default = 'aba', type = str)
    parser.add_argument('-t','--top', default = 1, type = int)
    parser.add_argument('-d','--depth', default = 3, type = int)
    parser.add_argument('-n','--number', default = 4, type = int)
    parser.add_argument('-q','--quiet', action = 'store_true')
    parser.add_argument('--save', default = None, type = str)
    parser.add_argument('--load', default = None, type = str)
    
    
    arguments = parser.parse_args()

    sampling = {'aba': sampleABA,
                'abb': sampleABB,
                'abx': sampleABX,
                'aax': sampleAAX,
                }
        
    trainingData = sampling[arguments.problem](arguments.number)
    print u"\n".join(trainingData)
    surfaceLength = sum([len(tokenize(w)) for w in trainingData ])

    costToSolution = {}
    if arguments.load != None:
        assert not arguments.quiet
        costToSolution = loadPickle(arguments.load)
    else:
        for d in range(0,arguments.depth + 1):
            worker = UnderlyingProblem([(w,) for w in trainingData ])
            solutions, costs = worker.paretoFront(d, arguments.top, TEMPERATURE, useMorphology = True)
            for solution, cost in zip(solutions, costs): costToSolution[cost] = solution

    if arguments.save != None:
        assert arguments.load == None
        dumpPickle(costToSolution, arguments.save)
        
    colors = cm.rainbow(np.linspace(0, 1, 1))
    if not arguments.quiet:
        #plot.rc('text', usetex=True)
        #plot.rc('font', family='serif')
        plot.scatter([ -p[0] for p in costToSolution],
                     [ -p[1]/float(arguments.number) for p in costToSolution],
                     alpha = 1.0/(1+2), s = 100, color = colors)
        plot.ylabel("Fit to data (-average UR size)")
        plot.xlabel("Parsimony (-Program length)")
        plot.title("Pareto front for %s, %d example%s"%(arguments.problem,arguments.number,'' if arguments.number == 1 else 's'))
                
        # these are matplotlib.patch.Patch properties
        ax = plot.gca()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        plot.text(0.05, 0.05, u"\n".join(trainingData), transform=ax.transAxes,
                  fontsize=14, verticalalignment='bottom', horizontalalignment='left', bbox=props)

        front = removePointsNotOnFront([ (-c1,-c2/float(arguments.number)) for c1,c2 in costToSolution.keys() ])
        # diagram the front itself
        plot.plot([ x for x,y in front ], [ y for x,y in front ],'--')

        # Decide which points to label with the corresponding program
        solutionsToLabel = list(front)
        for c2 in set([ c2 for c1,c2 in costToSolution ]):
            y = -c2/float(arguments.number)
            candidates = [ -c1 for c1,_c2 in costToSolution if _c2 == c2 and not (-c1,y) in front ]
            if candidates != []:
                chosen = choice(candidates)
                solutionsToLabel.append((chosen,y))

        # illustrate the synthesized programs along the front
        dy = 0.3
        for c1,c2 in sorted(costToSolution.keys(), key = lambda cs: (cs[1],cs[0])):
            solution = costToSolution[(c1,c2)]
            x1 = -c1
            y1 = -c2/float(arguments.number)
            x2 = x1
            y2 = y1 + dy
            print x2,y2
            print solution.pretty()

            if not (x1,y1) in solutionsToLabel: continue
            dy = -1*dy
            
            plot.text(x2,y2, solution.pretty(),
                      fontsize=12, bbox=props,
                      verticalalignment = 'bottom' if dy < 0 else 'top',
                      horizontalalignment = 'center')
            ax.annotate('',
                        xy = (x2,y2),xycoords = 'data',
                        xytext = (x1,y1),textcoords = 'data',
                        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3'))



        #plot.ylim([1,2])
#        plot.xlim([1,2])
        plot.show()
