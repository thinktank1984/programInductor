# -*- coding: utf-8 -*-

from features import featureMap, tokenize, wordToMatrix
from rule import Rule
from sketch import *

from problems import underlyingProblems

from random import random
import sys

class UnderlyingProblem():
    def __init__(self, problem):
        data = problem.data

        self.numberOfInflections = len(data[0])
        self.inflectionMatrix = [ [ wordToMatrix(i) for i in Lex ] for Lex in data ]

    def sketchSolution(self):
        Model.Global()
        
        depth = 1 if len(sys.argv) < 3 else int(sys.argv[2])
        rules = [ Rule.sample() for _ in range(depth)  ]

        stems = [ sampleMorph() for _ in self.inflectionMatrix ]
        prefixes = [ sampleMorph() for _ in range(self.numberOfInflections) ]
        suffixes = [ sampleMorph() for _ in range(self.numberOfInflections) ]

        def applyRules(d):
            for r in rules: d = applyRule(r,d)
            return d
        surfaces = [ [ applyRules(concatenate3(prefixes[i], stems[l], suffixes[i]))
                   for i in range(self.numberOfInflections) ]
                 for l in range(len(stems)) ]

        for l in range(len(stems)):
            for i in range(self.numberOfInflections):
                condition(wordEqual(makeConstantWordOfMatrix(self.inflectionMatrix[l][i]), surfaces[l][i]))
        output = solveSketch()
        print output

        for r in rules:
            print Rule.parse(output, r)

data = underlyingProblems[int(sys.argv[1]) - 1]
print data.description
UnderlyingProblem(data).sketchSolution()
