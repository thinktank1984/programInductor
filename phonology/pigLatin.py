# -*- coding: utf-8 -*-

from utilities import *

from features import FeatureBank, tokenize
from rule import Rule
from morph import Morph
from sketch import *

'''
Pig Latin rules:
1. Copy the first consonant to the end:
   # > 1 / # C_1 [ ]* _ 
2. Delete the first consonant:
   C ---> Ø / # _ 
3. Append e:
   # > e / _
'''

# learn to delete the first character
pigLatinExamples1 = [(u"pɩg", u"ɩg"),#pe"),
                    (u"latɩn", u"atɩn"),#le"),
                    (u"no", u"o"),#ne"),
                    # (u"it", u"ite"),
                    # (u"ask", u"ask")
]
# learn to append "e"
pigLatinExamples2 = [(u"pɩg", u"pɩge"),#pe"),
                    (u"latɩn", u"latɩne"),#le"),
                    (u"no", u"noe"),#ne"),
                    (u"it", u"ite"),
                     (u"ask", u"aske")
]
# learn to copy the first letter only if it is a consonant
pigLatinExamples3 = [(u"pɩg", u"pɩgp"),#pe"),
                    (u"latɩn", u"latɩnl"),#le"),
                    (u"no", u"non"),#ne"),
                    (u"it", u"it"),
                     (u"ask", u"ask")
]
# learn pig Latin. System produces:
#    # ---> -1 / # [ -vowel ] [  ]* _ 
#    # ---> e /  _ 
#    [ -vowel ] ---> Ø / # _ 
pigLatinExamples4 = [(u"pɩg", u"ɩgpe"),#pe"),
                    (u"latɩn", u"atɩnle"),#le"),
                    (u"no", u"one"),#ne"),
                    (u"it", u"ite"),
                     (u"ask", u"aske")
]
pigLatinExamples = pigLatinExamples4

depth = 3

bank = FeatureBank([ w for l in pigLatinExamples for w in l ])
maximumObservationLength = max([ len(tokenize(w)) for l in pigLatinExamples for w in l ]) + 1

def conditionOnExample(r, x, y):
    y = makeConstantWord(bank, y)
    x = makeConstantWord(bank, x)
    condition(wordEqual(y, applyRules(r, x, 10)))

Model.Global()
rules = [ Rule.sample() for _ in range(depth) ]
def applyRules(r,x):
    if len(r) == 0: return x
    return applyRules(r[1:], applyRule(r[0], x))

for x,y in pigLatinExamples:
    conditionOnExample(rules, x, y)
minimize(sum([ ruleCost(r) for r in rules ]))
output = solveSketch(bank, maximumObservationLength)

if not output:
    print "Failed to discover a system of rules."
    assert False
for r in rules:
    print Rule.parse(bank, output, r)
