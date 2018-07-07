# -*- coding: utf-8 -*-


from problems import underlyingProblems, interactingProblems, alternationProblems, MATRIXPROBLEMS, Problem
from rule import *
from utilities import *
from fragmentGrammar import *
from time import time
import re
import math
#import matplotlib.pyplot as plot
import pickle
import os
from random import random
import cProfile
from problems import MATRIXPROBLEMS


def worker(arguments):
    if arguments.task == 'fromGroundTruth':
        groundTruthSolutions = []
        for problem in MATRIXPROBLEMS:
            if isinstance(problem,Problem):
                for s in problem.solutions:
                    print s
                    groundTruthSolutions.append(parseSolution(s))
        print "Successfully loaded %s solutions"%(len(groundTruthSolutions))
        groundTruthRules = [ [r] for s in groundTruthSolutions for r in s.rules ]
        print "Going to induce a fragment grammar from %d rules"%(len(groundTruthRules))
        g = induceFragmentGrammar(groundTruthRules, CPUs = arguments.CPUs)
    elif arguments.task == 'fromFrontiers':
        frontiers = [ loadPickle('frontierPickles/empiricalUniversal/matrix_%d.p'%j) for j in range(arguments.problems) ]
        print "Successfully loaded %s frontiers."%(len(frontiers))
        # convert frontier objects to list of list of rules
        g = induceFragmentGrammar([ rs
                                    for frontier in frontiers
                                    for rs in frontier.frontiers ],
                                  CPUs = arguments.CPUs)

    if arguments.export != None:
        exportPath = arguments.export
        print "Exporting universal grammar to %s"%(exportPath)
        g.export(exportPath)

    

if __name__ == '__main__':
    import argparse
    from parseSPE import parseSolution

    parser = argparse.ArgumentParser(description = 'Infer universal grammars')
    parser.add_argument('task',choices = ['fromGroundTruth','fromFrontiers'])
    parser.add_argument('--export', type = str, default = None)
    parser.add_argument('--problems', type = int, default = 0)
    parser.add_argument('--CPUs', type = int, default = numberOfCPUs())
    
    arguments = parser.parse_args()

    worker(arguments)
    
