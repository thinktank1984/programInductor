# -*- coding: utf-8 -*-

from utilities import *

from features import FeatureBank, tokenize, featureMap
from rule import Rule
from morph import Morph
from sketch import *

from random import choice
import matplotlib.pyplot as plot

TEMPERATURE = 3.0

def sampleSyllable():
    v = choice([u"i",u"ɩ",u"e",u"ə",u"ɛ",u"æ",u"a",u"u",u"ü",u"ʊ",u"o",u"ö",u"ɔ"])
    k = choice([u"p",u"b",u"f",u"v",u"β",u"m",u"θ",u"d",u"t",u"ð",u"z",u"ǰ",u"s",u"n",u"š",u"k",u"g",u"ŋ",u"h",u"w",u"y",u"r",u"l"])
    return k + v
def sampleABA():
    s = sampleSyllable()
    d = sampleSyllable()
    if len(set(tokenize(s))&set(tokenize(d))) == 0:
        return s + d + s
    return sampleABA()
def sampleABB():
    s = sampleSyllable()
    d = sampleSyllable()
    if len(set(tokenize(s))&set(tokenize(d))) == 0:
        return d + s + s
    return sampleABB()

'''
ABA Rules:
Underlying forms:
/ p o f e /
/ u g v a /
/ a b h i /
/ t i k e /
/ l a r o /
Rules:
# ---> -1 / # [  ] [  ]* _ 
# ---> -2 / # [  ] [  ]* _ 

'''

def topSolutions(depth, observations):
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
    minimize(sum([ ruleCost(r) for r in rules ]))
    output = solveSketch(bank, maximumObservationLength)

    print "Underlying forms:"
    underlyingForms = [ Morph.parse(bank, output, u) for u in underlyingForms ]
    print "\n".join(map(str,underlyingForms))
    rules = [ Rule.parse(bank, output, r) for r in rules ]
    rc = sum([r.cost() for r in rules ])
    uc = sum([len(u) for u in underlyingForms ])
    print "Rules:"
    print "\n".join(map(str,rules))
    return underlyingForms, rules, (rc, uc)

if __name__ == 'main':
    ABA = [sampleABA() for _ in range(6) ]
    ABB = [sampleABB() for _ in range(6) ]
    trainingData = ABA
    points = []
    for d in range(1,5):
        rules, forms, p = topSolutions(d, trainingData)
        points.append(p)
    print points
    plot.plot([ p[0] for p in points],
              [ p[1] for p in points])
    plot.show()

