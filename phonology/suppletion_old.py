# -*- coding: utf-8 -*-

from result import *
from compileRuleToSketch import compileRuleToSketch
from utilities import *
from problems import *
from solution import *
from features import FeatureBank, tokenize
from rule import * # Rule,Guard,FeatureMatrix,EMPTYRULE
from morph import Morph,observeWord
from sketchSyntax import Expression,makeSketchSkeleton
from sketch import *
from supervised import SupervisedProblem
from textbook_problems import *
from latex import latexMatrix
import matplotlib.pyplot as plt
import numpy as np
import re
import codecs
import itertools as it

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

## some utility functions
## gets the rule costs from the list of solutions
def get_x(ls):
	return [ls[i][0] for i in range( len(ls) )]

## gets the lexicon costs from the list of solutions
def get_y(ls):
	return [ls[i][1] for i in range( len(ls) )]

## gets a unicode representation of the grammar learned
def get_components(s):
	UR = ", ".join( [str(v) for k, v in s.underlyingForms.items()] )
	suffix1, suffix2, suffix3 = s.suffixes
	rule1, rule2 = s.rules
	data = s.underlyingForms.keys() #  \nData explained: %s

	# check for which affix strategy is used
	s, ot, to = s.suppletion
	if s: out = "Stem URs: %s \nAffix URs: %s, %s, %s" % (UR, suffix1, suffix2, suffix3)
	elif ot: out = "Stem URs: %s \nAffix URs: %s, %s, %s" % (UR, suffix1, suffix2, suffix1 + suffix2)
	else: out = "Stem URs: %s \nAffix URs: %s, %s, %s" % (UR, suffix1, suffix2, suffix2 + suffix1)

	# check if one or two rules are learned
	if rule2 == "[  ] ---> [  ] /  _ ": out = out + "\nRule 1: %s" % (rule1)
	else: out = out +  "\nRule 1: %s \nRule 2: %s" % (rule1, rule2)

	return unicode(out, "utf-8")

## checks to see whether the solution is able to account for the data point
def getStem(solution, inflections):
    s,ot,to = solution.suppletion
    if s: thirdSuffix = solution.suffixes[2]
    elif ot: thirdSuffix = solution.suffixes[0] + solution.suffixes[1]
    elif to: thirdSuffix = solution.suffixes[1] + solution.suffixes[0]
    solution = Solution(rules=solution.rules,
                        prefixes=[Morph(u"")] + solution.prefixes,
                        suffixes=[Morph(u""),solution.suffixes[0],solution.suffixes[1],thirdSuffix])
    inflections = tuple(Morph(x) if isinstance(x,(unicode,str)) else x
                        for x in inflections)

    print "Going to verify this data: ",inflections,"\nagainst this solution:\n",solution

    stem = solution.transduceUnderlyingForm(FeatureBank.ACTIVE,inflections)
    if stem is not None:
        print "Successfully verified: stem is", stem
    else:
        print "Could not verify"
    return stem

## setting up the cost functions
disableConstantPhonemes() # don't allow individual segments unless it is an insertion process
enableCV() # give [+/- vowel] features a cost of 1 instead of 2

## creating a command server to run sketch
os.system("python command_server.py 4&"); os.system("sleep 4")

## read and import data
filename = "opaque/dataset1-cf.txt"
with codecs.open(filename, encoding='utf-8') as f:
    content = f.read().splitlines()

points = np.array([0, 18, 33, 59, 62, 40])
complete_data = [tuple(content[i].split('\t')) for i in range(len(content))] #[imit 0, inad 18,  iluk 33, unug 59, ulid 62, umat 40] # [0, 18, 33, 59, 62, 40] <- forms used for pareto front (roughly equal distribution of segments)
data = [complete_data[i] for i in points]
print(data)

## setting up the model
solutions = [] # a list of past solutions: element is Solution object (see solution.py for implementation)
solutionCosts = [] # a list of past solution costs: (ruleCost, lexiconCost)

# set up an arbitrary number of solutions to look for; set to 1 in order to calculate global optimum of the data
for i in range(10):
	globalModel([ w for ws in data for w in ws ]) # create model and feature bank

	stems = [Morph.sample() for i in range( len(data) )]

	#affix1 = Morph.sample()
	#affix2 = Morph.sample()
	#affix3 = Morph.sample()
	suffixes = [Morph.sample() for i in range( len(data[0]) - 1 )] # create suffixes = to # of column -1 (one column is the stem in isolation)

	#rule1 = Rule.sample()
	#rule2 = Rule.sample()
	rules = [Rule.sample() for i in range ( 2 )] # set up in advance how many rules to consider

	# binary variables the model needs to reason about
	suppletive = flip() # says third slot is its own form
	onetwo = flip() # correspond to order of affixes
	twoone = flip()
	condition(suppletive + onetwo + twoone == 1)
	#condition(suppletive == 0) # <- if you change the cost of the morphemes, may end up choosing the intended phonological analysis
						   	# e.g. increase cost of affixes

	new_data = zip(stems, data) # generates a list of tuples with each ith element corresponding to a tuple at position i in each list

	actual_suffix = ite(  suppletive, suffixes[2], ite(onetwo, concatenate(suffixes[0], suffixes[1]), concatenate(suffixes[1], suffixes[0]))  ) # checks to see which of the binary variables is heads and sets the affix to be whatever that condition is

	for stem, (surface1, surface2, surface3, surface4) in new_data: # for each data point (paradigm)
		maximumLength = max(len(surface1), len(surface2), len(surface3), len(surface4)) + 1 # set bounded amount that rule can look at when applying rule (should be bigger than the longest stem)
		predicted_surface1 = applyRule(rules[1], applyRule(rules[0], stem, maximumLength), maximumLength) # number is bounded amount that rule can look at when applying rule (should be bigger than the longest stem)
		predicted_surface2 = applyRule(rules[1], applyRule(rules[0], concatenate(stem, suffixes[0]), maximumLength), maximumLength) # first number specifies for morpheme boundaries (length of string until suffix (len(prefix + stem))); second number is bounded amount that rule can look at when applying rule (should be bigger than the longest stem)
		predicted_surface3 = applyRule(rules[1], applyRule(rules[0], concatenate(stem, suffixes[1]), maximumLength), maximumLength)
		predicted_surface4 = applyRule(rules[1], applyRule(rules[0], concatenate(stem, actual_suffix), maximumLength), maximumLength)

		observeWord(surface1, predicted_surface1)
		observeWord(surface2, predicted_surface2)
		observeWord(surface3, predicted_surface3)
		observeWord(surface4, predicted_surface4)

	lexiconCostExpression = sum([ wordLength(u) for u in stems ]) + sum([ wordLength(s) for s in suffixes ])
	lexiconCostVariable = unknownInteger(8) # so that we can recover the cost of the lexicon later, number corresponds to max number of bits to encode stems
	condition(lexiconCostVariable == lexiconCostExpression)
	minimize(lexiconCostExpression) # minimize the cost of the lexicon

	ruleCostExpression = sum([ ruleCost(r) for r in rules ])
	ruleCostVariable = unknownInteger()
	condition(ruleCostVariable == ruleCostExpression) # so that we can recover the cost of the lexicon later
	minimize(ruleCostExpression) # minimize the cost of the lexicon

	# compute pareto front; tell it to check if the rule cost / lexicon cost is greater than any of the old grammars, OR if it is the same
	# to do standard thing, just get rid of the outer OR and AND
	for rc, lc in solutionCosts:
		# condition(Or( [Or([ruleCostVariable < rc, lexiconCostVariable < lc]), And([ruleCostVariable == rc, lexiconCostVariable == lc])] ))
		condition(Or([ruleCostVariable < rc, lexiconCostVariable < lc]))
		# condition(ruleCostVariable + lexiconCostVariable == rc + lc)

	# check if both the rules posited are the same as in any of the old grammars (this is so that it can consider multiple different solutions)
	# get rid of this if you only want one solution per point
	#for oldSolution in solutions:
	#	condition( Not(And([ruleEqual(rules[0], oldSolution.rules[0].makeConstant()),  ruleEqual(rules[1], oldSolution.rules[1].makeConstant())])))

	# invoke the solver; break if it can't find a solution that matches the criterion above
	try:
		output = solveSketch(None, unroll=10, maximumMorphLength=10)
	except SynthesisFailure:
		print("Failed to find solution")
		break

	# push all this info into some data structure
	stems = [Morph.parse(stem) for stem in stems]
	rules = [Rule.parse(r) for r in rules]
	suffixes = [Morph.parse(s) for s in suffixes]
	prefixes = [Morph('')] * len(suffixes)
	underlyingForms = dict(zip(data, stems))
	suppletion = (parseFlip(output, suppletive), parseFlip(output, onetwo), parseFlip(output, twoone))

	sol = Solution(rules = rules, prefixes = prefixes, suffixes = suffixes, underlyingForms = underlyingForms)
	sol.suppletion = suppletion

	solutions.append(sol)
	solutionCosts.append((parseInteger(output, ruleCostVariable), parseInteger(output, lexiconCostVariable)))

	## Uncomment to check whether the data is accounted for by the solution
	'''
	        for xs in complete_data:
	            getStem(sol,xs)
	'''

data = zip(solutionCosts, solutions)
sorted_data = sorted(data, key=lambda tup: tup[0])
solutionCosts = list(zip(*sorted_data))[0]

plt.clf()
x = get_x(solutionCosts)
y = get_y(solutionCosts)
plt.plot(x, y,'bo-')

# setting up the labels
for xy, s in sorted_data:
    lab = get_components(s) + "\n" + str(xy)
    print(xy)
    print(lab)

    plt.annotate(lab, # this is the text
                xy, # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='left') # horizontal alignment can be left, right or center

language = "opaque"
plt.title("Pareto frontier of %s language" % (language), fontsize = 18)
plt.xlabel("Rule cost", fontsize = 12)
plt.ylabel("Lexicon cost", fontsize = 12)

#for the transparent language
#plt.xticks(np.arange(7,22,1))
#plt.yticks(np.arange(20,30,1))

#for the opaque language
plt.xticks(np.arange(9,15,1))
plt.yticks(np.arange(24,33,1))

plt.show()
# print(plotting_stuff)
# dumpPickle(plotting_stuff, "experimentOutputs/suppletion_outputs.pkl")
