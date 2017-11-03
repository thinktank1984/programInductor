# -*- coding: utf-8 -*-

from features import FeatureBank, tokenize
from latex import latexWord
from rule import *
from morph import Morph
from sketch import *
from utilities import *

class CountingProblem():
    def __init__(self, data, count):
        self.data = [ Morph(tokenize(x)) for x in data ]
        self.count = count
        self.bank = FeatureBank([ w for w in data ])

        self.maximumObservationLength = max([ len(w) for w in self.data ]) + 1

    def latex(self):
        r = "\\begin{tabular}{ll}\n"
        for k,o in zip(self.count, self.data):
            if k < 11:
                r += "%d&%s\\\\\n"%(k,latexWord(o))
            elif k%10 == 0:
                r += "%d = %d + 10 & %s\\\\\n"%(k,k/10,latexWord(o))
            elif k < 20:
                r += "%d = 10 + %d & %s\\\\\n"%(k,k-10,latexWord(o))
            else:
                assert False
        r += "\n\\end{tabular}\n"
        return r

    # def heldOutSolution(self, k, testing, inductiveBiases):
    #     if testing == 0.0:
    #         return self.topSolutions(k),None,None

    #     training,testing = randomTestSplit(list(range(len(self.data))), testing)
    #     slave = CountingProblem([ d for j,d in enumerate(self.data) if j in training ],
    #                             [ d for j,d in enumerate(self.count) if j in training ])
    #     solution = slave.topSolutions(k)

        
    def solveFrontiers(self, seed, k):
        # Ignore the seed because this problem is very easy to solve
        return [ [r] for r in self.topSolutions(k) ]

    def topSolutions(self, k = 10):
        solutions = []
        oldMorphology = None
        for _ in range(k):
            r,oldMorphology = self.sketchSolution(solutions,oldMorphology)
            if r == None: break
            solutions.append(r)
        return solutions

    def sketchSolution(self, existingRules, existingMorphology = None):
        Model.Global()

        r = Rule.sample()
        for o in existingRules:
            condition(ruleEqual(r, o.makeConstant(self.bank)) == 0)

        morphs = {}
        morphs[1] = Morph.sample()
        morphs[4] = Morph.sample()
        morphs[5] = Morph.sample()
        morphs[9] = Morph.sample()
        morphs[10] = Morph.sample()

        if existingMorphology:
            for k,v in existingMorphology.iteritems():
                condition(wordEqual(v.makeConstant(self.bank),
                                    morphs[k]))

        for j in range(len(self.data)):
            o = self.data[j]
            k = self.count[j]
            if k <= 10:
                condition(wordEqual(o.makeConstant(self.bank),
                                    applyRule(r,morphs[k],self.maximumObservationLength)))
            elif k%10 == 0:
                condition(wordEqual(o.makeConstant(self.bank),
                                    applyRule(r,concatenate(morphs[k/10], morphs[10]),self.maximumObservationLength)))
            elif k < 20:
                condition(wordEqual(o.makeConstant(self.bank),
                                    applyRule(r,concatenate(morphs[10], morphs[k - 10]),self.maximumObservationLength)))
            else:
                assert False

        minimize(ruleCost(r))

        try:
            output = solveSketch(self.bank,
                                 unroll = self.maximumObservationLength + 2,
                                 maximumMorphLength = self.maximumObservationLength + 1)
        except SynthesisFailure:
            print "Failed at phonological analysis."
            return None

        r = Rule.parse(self.bank, output, r)
        print r.pretty()
        return r,dict([ (k,Morph.parse(self.bank, output, m)) for k,m in morphs.iteritems() ])

