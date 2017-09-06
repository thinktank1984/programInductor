# -*- coding: utf-8 -*-

from utilities import *

from supervised import SupervisedProblem
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
# Ø ---> -2 / # [ -vowel ] [  ]* _ #
# [ -vowel ] ---> Ø / # _ 
# Ø ---> e /  _ #
pigLatinExamples4 = [(u"pɩg", u"ɩgpe"),#pe"),
      (u"latɩn", u"atɩnle"),#le"),
      (u"no", u"one"),#ne"),
      (u"it", u"ite"),
      (u"ask", u"aske")
]
pigLatinExamples = pigLatinExamples4

depth = 3

solution = SupervisedProblem([ (Morph(tokenize(x)), Morph(tokenize(y))) for x,y in pigLatinExamples ]).solve(depth)
if solution == None:
    print "No solution."
else:
    for r in solution: print r
