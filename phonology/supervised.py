# -*- coding: utf-8 -*-


from features import FeatureBank, tokenize
from rule import Rule
from morph import Morph
from sketch import *


def solveTopSupervisedRules(examples, k, existingRule = None):
    # print "SUPERVISEDINPUTS:"
    # for x,y in examples:
    #     print x,
    #     print "\t",y

        
    solutions = [] if existingRule == None else [existingRule]
    bank = FeatureBank([ ''.join(w.phonemes) for (x,y) in examples for w in [x,y] ])

    maximumObservationLength = max([len(m) for e in examples for m in e ]) + 1
    maximumMorphLength = maximumObservationLength

    for _ in range(k - (1 if existingRule else 0)):
        Model.Global()
        rule = Rule.sample()
        for other in solutions:
            condition(ruleEqual(rule, other.makeConstant(bank)) == 0)
        minimize(ruleCost(rule))

        condition(fixStructuralChange(rule))

        for x,y in examples:
            condition(wordEqual(applyRule(rule, x.makeConstant(bank)),
                                y.makeConstant(bank)))
        output = solveSketch(bank, maximumObservationLength, maximumMorphLength)
        if not output: break
                   
        solutions.append(Rule.parse(bank, output, rule))
    # print "SUPERVISEDOUTPUTS:"
    # for r in solutions:
    #     print r
    return solutions
