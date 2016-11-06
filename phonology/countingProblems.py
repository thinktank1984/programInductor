# -*- coding: utf-8 -*-

from features import FeatureBank, tokenize
from rule import Rule
from morph import Morph
from sketch import *


class CountingProblem():
    def __init__(self, data, count):
        self.data = data
        self.count = count
        self.bank = FeatureBank([ w for w in data ])

        self.maximumObservationLength = max([ len(tokenize(w)) for w in data ]) + 1

    def sketchSolution(self):
        Model.Global()

        r = Rule.sample()

        morphs = {}
        morphs[1] = Morph.sample()
        morphs[4] = Morph.sample()
        morphs[5] = Morph.sample()
        morphs[9] = Morph.sample()
        morphs[10] = Morph.sample()

        for j in range(len(self.data)):
            o = self.data[j]
            k = self.count[j]
            if k <= 10:
                condition(wordEqual(makeConstantWord(self.bank, o),
                                    applyRule(r,morphs[k])))
            elif k%10 == 0:
                condition(wordEqual(makeConstantWord(self.bank, o),
                                    applyRule(r,concatenate(morphs[k/10], morphs[10]))))
            elif k < 20:
                condition(wordEqual(makeConstantWord(self.bank, o),
                                    applyRule(r,concatenate(morphs[10], morphs[k - 10]))))
            else:
                assert False

        minimize(ruleCost(r))

        output = solveSketch(self.bank, self.maximumObservationLength)
        
        if not output:
            print "Failed at phonological analysis."
            assert False

        print Rule.parse(self.bank, output, r)

