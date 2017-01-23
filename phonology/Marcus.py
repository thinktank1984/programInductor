# -*- coding: utf-8 -*-

from utilities import *

from features import FeatureBank, tokenize
from rule import Rule
from morph import Morph
from sketch import *

ABA = [u"pofepo",
       u"ugvaug",
       u"abhiab",
       u"tiketi",
       u"larola"]

ABB = [ x[0:2] + x[2:4] + x[2:4] for x in ABA ]

trainingData = ABA
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

depth = 2

bank = FeatureBank([ w for l in trainingData for w in l ])
maximumObservationLength = max([ len(tokenize(w)) for w in trainingData ]) + 1

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

underlyingForms = [ conditionOnExample(rules,x) for x in trainingData ]
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
print "\n".join([ str(Morph.parse(bank, output, u)) for u in underlyingForms ])
print "Rules:"
print "\n".join([ str(Rule.parse(bank, output, r)) for r in rules ])
