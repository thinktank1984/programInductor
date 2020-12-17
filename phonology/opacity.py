# -*- coding: utf-8 -*-

from result import *
from compileRuleToSketch import compileRuleToSketch
from utilities import *
from problems import *
from solution import *
from features import FeatureBank, tokenize, switchFeatures
from rule import * # Rule,Guard,FeatureMatrix,EMPTYRULE
from morph import Morph,observeWord,observeWordIfNotMemorized
from sketchSyntax import Expression,makeSketchSkeleton
from sketch import *
from supervised import SupervisedProblem
from textbook_problems import *
from latex import latexMatrix
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

switchFeatures('Riggle')

## ==============================================================================
##      SOME UTILITY FUNCTIONS
## ==============================================================================
## gets the rule costs from the list of solutions
def get_x(ls):
    return [ls[i][0] for i in range( len(ls) )]

## gets the lexicon costs from the list of solutions
def get_y(ls):
    return [ls[i][1] for i in range( len(ls) )]

## gets a unicode representation of the grammar learned
def get_components(solution):
    UR = ", ".join( [str(v) for k, v in solution.underlyingForms.items()] )
    suffix1, suffix2 = solution.suffixes
    data = solution.underlyingForms.keys()
    out = "Stem URs: %s \nAffix URs: %s, %s" % (UR, suffix1, suffix2)

    # check if one or two rules are learned
    out = out + "\n" + ("\n".join("Rule %d: %s"%(ri,r) for ri,r in enumerate(solution.rules) )) + "\n"
    out = unicode(out, "utf-8")

    for ms,surfaces in zip(solution.memorize,solution.data):
        for m,surface in zip(ms,surfaces):
            if m:
                out += u"mem " + unicode(surface) + "  "
    return out

## checks to see whether the solution is able to account for the data point
def getStem(solution, inflections, canMemorize=False):
    # morphology is subtle:
    # the first inflection is just the stem
    # the last inflection is concatenation of first/second inflected forms
    solution = Solution(rules=solution.rules,
                        prefixes=[Morph(u"")] * len(inflections),
                        suffixes=[Morph(u"")] + solution.suffixes + (solution.suffixes[0]+solution.suffixes[1]))
    inflections = tuple(Morph(x) if isinstance(x,(unicode,str)) else x
                        for x in inflections)

    result  = solution.transduceUnderlyingForm(FeatureBank.ACTIVE,inflections, canMemorize=canMemorize)
    if result is None:
        print "Could not recover a stem:",inflections
        return None

    stem,memorizeVector = result[0],result[1]
    return stem, tuple([ thing_to_memorize if was_memorized else None
                         for was_memorized, thing_to_memorize in zip(memorizeVector, inflections) ])


## ==============================================================================
##      SIMULATION
## ==============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data",nargs='?')
    parser.add_argument("test",nargs='?') # test dataset to compute likelihood over
    parser.add_argument("--numberOfRules","-n",type=int,default=3)
    parser.add_argument("--export",default=None)
    parser.add_argument("--load",default=None)
    arguments = parser.parse_args()

    ## setting up the cost functions
    disableConstantPhonemes() # don't allow individual segments unless it is an insertion process
    enableCV() # give [+/- vowel] features a cost of 1 instead of 2
    os.system("python command_server.py 4&"); os.system("sleep 4")

    ## read and import data
    filename = arguments.data
    points = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    '''
    if arguments.data == "opaque":
        filename = "opaque/dataset2-cf.txt"
        points = np.array([0, 1, 2, 3, 4, 5])
    else:
        filename = "opaque/dataset2-f.txt"
        points = np.array([0, 1, 2, 3, 4, 5])
    '''

    with codecs.open(filename, encoding='utf-8') as f:
        content = f.read().splitlines()

    complete_data = [tuple(content[i].split('\t')) for i in range(len(content))]
    data = [complete_data[i] for i in points]
    n_inflections = len(data[0])
    print("complete data:")
    for inflections in data:
        print(u"\t".join(inflections))
    print "solving with <=",arguments.numberOfRules,"rules"

    ## setting up the model
    solutions = [] # a list of past solutions: element is Solution object (see solution.py for implementation)
    solutionCosts = [] # a list of past solution costs: (ruleCost, lexiconCost)
    if arguments.load is None: # we are not loading a old solution set so we need compute it
        # set up an arbitrary number of solutions to look for
        for i in range(10):
            globalModel([ w for ws in data for w in ws ]) # create model and feature bank

            stems = [Morph.sample() for i in range( len(data) )]
            memorize = [[flip() for j in data[0]] for i in data]
            suffixes = [Morph.sample() for i in range( len(data[0]) - 2 )] # create suffixes = to # of column -1 (one column is the stem in isolation)
            rules = [Rule.sample() for i in range ( arguments.numberOfRules )] # set up in advance how many rules to consider
            new_data = zip(stems, data) # generates a list of tuples with each ith element corresponding to a tuple at position i in each list

            for i, (stem, surfaces) in enumerate(new_data): # for each data point (paradigm)
                maximumLength = max([len(s) for s in surfaces]) + 1 # set bounded amount that rule can look at when applying rule (should be bigger than the longest stem)
                predicted_stem = applyRules(rules, stem, 0, maximumLength) # number is bounded amount that rule can look at when applying rule (should be bigger than the longest stem)
                if len(surfaces[0]) > 0:
                    observeWordIfNotMemorized(surfaces[0], predicted_stem, memorize[i][0])

                # for the rest
                for j in range(1, n_inflections):
                    if j == n_inflections-1:
                        predicted_complex = applyRules(rules, concatenate(concatenate(stem, suffixes[0]), suffixes[1]), 0, maximumLength) # first number specifies for morpheme boundaries (length of string until suffix (len(prefix + stem))); second number is bounded amount that rule can look at when applying rule (should be bigger than the longest stem)
                        if len(surfaces[j]) > 0:
                            observeWordIfNotMemorized(surfaces[j], predicted_complex, memorize[i][j])
                    else:
                        predicted_complex = applyRules(rules, concatenate(stem, suffixes[j-1]), 0, maximumLength) # first number specifies for morpheme boundaries (length of string until suffix (len(prefix + stem))); second number is bounded amount that rule can look at when applying rule (should be bigger than the longest stem)
                        if len(surfaces[j]) > 0:
                            observeWordIfNotMemorized(surfaces[j], predicted_complex, memorize[i][j])

            ## Lexicon Cost
            memorizeCostExpression = sum( ite(f, Constant(len(d)), Constant(0)) for fs, ds in zip(memorize, data) for f, d in zip(fs, ds) )
            lexiconCostExpression = sum([ wordLength(u) for u in stems ]) + sum([ wordLength(s) for s in suffixes ]) + memorizeCostExpression
            lexiconCostVariable = unknownInteger(8) # so that we can recover the cost of the lexicon later, number corresponds to max number of bits to encode stems
            condition(lexiconCostVariable == lexiconCostExpression)
            minimize(lexiconCostExpression) # minimize the cost of the lexicon

            ## Rule Cost
            ruleCostExpression = sum([ ruleCost(r) for r in rules ])
            ruleCostVariable = unknownInteger()
            condition(ruleCostVariable == ruleCostExpression) # so that we can recover the cost of the lexicon later
            minimize(ruleCostExpression) # minimize the cost of the lexicon

            # compute pareto front; tell it to check if the rule cost / lexicon cost is greater than any of the old grammars, OR if it is the same
            # to do standard thing, just get rid of the outer OR and AND
            for rc, lc in solutionCosts:
                condition(Or([ruleCostVariable < rc, lexiconCostVariable < lc]))

            # invoke the solver; break if it can't find a solution that matches the criterion above
            try:
                output = solveSketch(None, unroll=10, maximumMorphLength=10, minimizeBound=60)
            except SynthesisFailure:
                print("Failed to find solution")
                break

            # push all this info into some data structure
            stems = [Morph.parse(stem) for stem in stems]
            rules = [Rule.parse(r) for r in rules]
            suffixes = [Morph.parse(s) for s in suffixes]
            prefixes = [Morph('')] * len(suffixes)
            underlyingForms = dict(zip(data, stems))
            memorize = [[parseFlip(output,m) for m in ms] for ms in memorize]

            sol = Solution(rules = rules, prefixes = prefixes, suffixes = suffixes, underlyingForms = underlyingForms)
            sol.memorize = memorize
            sol.data = data

            solutions.append(sol)
            solutionCosts.append((parseInteger(output, ruleCostVariable), parseInteger(output, lexiconCostVariable)))

        data = zip(solutionCosts, solutions)

    if arguments.export:
        assert arguments.load is None, "cannot both export and load"
        with open(arguments.export,"wb") as handle:
            pickle.dump(data, handle)

    if arguments.load:
        assert arguments.export is None, "cannot both export and load"
        with open(arguments.load,"rb") as handle:
            data = pickle.load(handle)

    sorted_data = sorted(data, key=lambda tup: tup[0])
    solutionCosts = list(zip(*sorted_data))[0]

    import matplotlib.pyplot as plt
    plt.clf()
    x = get_x(solutionCosts)
    y = get_y(solutionCosts)
    plt.plot(x, y,'bo-')

    # setting up the labels
    for xy, s in sorted_data:
        lab = get_components(s) + "\n" + str(xy)

        plt.annotate(lab, # this is the text
                    xy, # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='left') # horizontal alignment can be left, right or center

    language = arguments.data
    plt.title("Pareto frontier of %s language" % (language), fontsize = 18)
    plt.xlabel("Rule cost", fontsize = 12)
    plt.ylabel("Lexicon cost", fontsize = 12)

    #for the transparent language
    plt.xticks(np.arange(7,22,1))
    plt.yticks(np.arange(20,40,1))


    plt.show()
