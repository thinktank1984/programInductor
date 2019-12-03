# -*- coding: utf-8 -*-

from result import *
from compileRuleToSketch import compileRuleToSketch
from utilities import *
from solution import *
from features import FeatureBank, tokenize
from rule import * # Rule,Guard,FeatureMatrix,EMPTYRULE
from morph import Morph,observeWord
from sketchSyntax import Expression,makeSketchSkeleton
from sketch import *
from supervised import SupervisedProblem
from textbook_problems import *
from latex import latexMatrix

from pathos.multiprocessing import ProcessingPool as Pool
import random
import sys
import pickle
import math
from time import time
import itertools
import copy
import os

#from command_server import start_server
#start_server(1)
os.system("python command_server.py 1&"); os.system("sleep 1")

data = [(u"imit", u"imita", u"imiči", u"imiči"), (u"ulag", u"ulaga", u"ulagi", u"ulagi")]
globalModel([w for ws in data for w in ws ])

affix1 = Morph.sample()
affix2 = Morph.sample()
affix3 = Morph.sample()

rule1 = Rule.sample()
rule2 = Rule.sample()

stems = [Morph.sample() for i in range( len(data) )]

# binary variables the model needs to reason about
suppletive = flip() # says third slot is its own form
onetwo = flip() # correspond to order of affixes
twoone = flip()
condition(suppletive + onetwo + twoone == 1)
#condition(suppletive == 0) # <- if you change the cost of the morphemes, may end up choosing the intended phonological analysis
						   # e.g. increase cost of affixes
#output = solveSketch(None, unroll=10, maximumMorphLength=10)
#print(parseFlip(output, suppletive))
#print(parseFlip(output, onetwo))
#print(parseFlip(output, twoone))

new_data = zip(stems, data) # generates a list of tuples with each ith element corresponding to a tuple at position i in each list

actual_affix = ite(suppletive, affix3, ite(onetwo, concatenate(affix1, affix2), concatenate(affix2, affix1)))

for stem, (surface1, surface2, surface3, surface4) in new_data:
	maximumLength = max(len(surface1), len(surface2), len(surface3), len(surface4)) + 1
	predicted_surface1 = applyRule(rule2, applyRule(rule1, stem, maximumLength), maximumLength) # first number specifies for morpheme boundaries (length of string until suffix (len(prefix + stem))); second number is bounded amount that rule can look at when applying rule (should be bigger than the longest stem)
	predicted_surface2 = applyRule(rule2, applyRule(rule1, concatenate(stem, affix1), maximumLength), maximumLength) # first number specifies for morpheme boundaries (length of string until suffix (len(prefix + stem))); second number is bounded amount that rule can look at when applying rule (should be bigger than the longest stem)
	predicted_surface3 = applyRule(rule2, applyRule(rule1, concatenate(stem, affix2), maximumLength), maximumLength)
	predicted_surface4 = applyRule(rule2, applyRule(rule1, concatenate(stem, actual_affix), maximumLength), maximumLength)

	observeWord(surface1, predicted_surface1)
	observeWord(surface2, predicted_surface2)
	observeWord(surface3, predicted_surface3)
	observeWord(surface4, predicted_surface4)

#print(sum(wordLength(stem) for stem in stems))

minimize(ruleCost(rule1) + ruleCost(rule2) + sum(wordLength(stem) for stem in stems) + wordLength(affix1) + wordLength(affix2) + wordLength(affix3)) # make affix3 cost 10x more

output = solveSketch(None, unroll=10, maximumMorphLength=10)
if output is not None:
		print "successfully resolved constraints"
		print([Morph.parse(stem) for stem in stems])
		print(Rule.parse(rule1))
		print(Rule.parse(rule2))
		print(Morph.parse(affix1))
		print(Morph.parse(affix2))
		print(Morph.parse(affix3))
		print(parseFlip(output, suppletive))
		print(parseFlip(output, onetwo))
		print(parseFlip(output, twoone))
else:
        print "could not successfully solve constraints"
