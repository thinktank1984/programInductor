# -*- coding: utf-8 -*-


from features import FeatureBank, tokenize
from rule import Rule
from morph import Morph
from sketch import *


class SupervisedProblem():
    def __init__(self, examples):
        self.examples = examples
        self.bank = FeatureBank([ w for x,y in self.examples for w in [x,y]  ])
        self.maximumObservationLength = max([len(m) for e in examples for m in e ]) + 1
        self.maximumMorphLength = self.maximumObservationLength


    def fastTopK(self, k, existingRule = None):
        solutions = [] if existingRule == None else [existingRule]

        for _ in range(k - (1 if existingRule else 0)):
            Model.Global()
            rule = Rule.sample()
            for other in solutions:
                condition(ruleEqual(rule, other.makeConstant(self.bank)) == 0)
            minimize(ruleCost(rule))

            for x,y in self.examples:
                auxiliaryCondition(wordEqual(applyRule(rule, x.makeConstant(self.bank), max(len(x),len(y)) + 1),
                                             y.makeConstant(self.bank)))
            output = solveSketch(self.bank, self.maximumObservationLength + 1, self.maximumMorphLength)
            if not output: break

            solutions.append(Rule.parse(self.bank, output, rule))
        return solutions

    def solve(self, d):
        Model.Global()
        rules = [ Rule.sample() for _ in range(d) ]
        minimize(sum([ ruleCost(r) for r in rules ]))

        for x,y in self.examples:
            auxiliaryCondition(wordEqual(applyRules(rules, x.makeConstant(self.bank), max(len(x),len(y)) + 1),
                                         y.makeConstant(self.bank)))
        try:
            output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
        except SynthesisFailure:
            #printLastSketchOutput()
            return None
        
        return [ Rule.parse(self.bank, output, r)
                 for r in rules ] 

