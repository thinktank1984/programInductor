# -*- coding: utf-8 -*-

from compileRuleToSketch import compileRuleToSketch
from utilities import *
from solution import *
from features import FeatureBank, tokenize
from rule import * # Rule,Guard,FeatureMatrix,EMPTYRULE
from morph import Morph
from sketchSyntax import Expression,makeSketchSkeleton
from sketch import *
from supervised import SupervisedProblem
from latex import latexMatrix

from pathos.multiprocessing import ProcessingPool as Pool
import random
import sys
import pickle
import math
from time import time
import itertools
import copy

def sampleMorphWithLength(l):
    m = Morph.sample()
    condition(wordLength(m) == l)
    return m
        

class UnderlyingProblem(object):
    def __init__(self, data, bank = None, useSyllables = False, UG = None):
        self.UG = UG
        
        if bank != None: self.bank = bank
        else:
            self.bank = FeatureBank([ w for l in data for w in l if w != None ] + ([u'-'] if useSyllables else []))

        self.numberOfInflections = len(data[0])
        for d in data: assert len(d) == self.numberOfInflections
        
        # wrap the data in Morph objects if it isn't already
        self.data = [ [ None if i == None else (i if isinstance(i,Morph) else Morph(tokenize(i)))
                        for i in Lex] for Lex in data ]

        self.maximumObservationLength = max([ len(w) for l in self.data for w in l if w != None ])

        self.wordBoundaries = any([ (u'##' in w.phonemes) for l in self.data for w in l if w ])

    def solveSketch(self, minimizeBound = 31):
        return solveSketch(self.bank,
                           # unroll: +1 for extra UR size, +1 for guard buffer
                           self.maximumObservationLength + 2,
                           # maximum morpheme size
                           self.maximumObservationLength,
                           showSource = False, minimizeBound = minimizeBound)

    def debugSolution(self,s,u):
        for i in range(self.numberOfInflections):
            x = s.prefixes[i] + u + s.suffixes[i]
            print "Debugging inflection %d, which has UR = %s"%(i + 1,x)
            usedRules = []
            for j,r in enumerate(s.rules):
                y = self.applyRuleUsingSketch(r,x,len(s.prefixes[i]) + len(u))
                print "Rewrites to %s using rule\t%s"%(y,r)
                if x != y: usedRules.append(j)
                x = y
            print "Used rules:",usedRules
            print

    def illustrateSolution(self, solution):
        for x in self.data:
            u = solution.transduceUnderlyingForm(self.bank, x, getTrace = True)
            if u == None: print "COUNTEREXAMPLE:"," ~ ".join(map(str,x))
            else:
                print "PASSES:"," ~ ".join(map(str,x))
                (ur, traces) = u
                used = {}
                for trace in traces:
                    if trace == None: continue
                    print trace
                    for j in range(len(solution.rules)):
                        if trace[j] != trace[j + 1]: used[j] = True
                print "Uses rules",list(sorted(used.keys())),":"
                for j in sorted(used.keys()):
                    print solution.rules[j].pretty()
                print 

    def applyRuleUsingSketch(self,r,u,untilSuffix):
        '''u: morph; r: rule; untilSuffix: int'''
        Model.Global()
        result = Morph.sample()
        _r = r.makeDefinition(self.bank)
        condition(wordEqual(result,applyRule(_r,u.makeConstant(self.bank),
                                             Constant(untilSuffix), self.maximumObservationLength + 1)))
        try:
            output = self.solveSketch()
        except SynthesisFailure:
            print "applyRuleUsingSketch: UNSATISFIABLE for %s %s %s"%(u,r,untilSuffix)
            assert False
        except SynthesisTimeout:
            print "applyRuleUsingSketch: TIMEOUT for %s %s %s"%(u,r,untilSuffix)
            assert False
        return Morph.parse(self.bank, output, result)
        
    
    def applyRule(self, r, u):
        assert False,'UnderlyingProblem.applyRule: deprecated'

    def conditionPhoneme(self, expression, index, phoneme):
        variables = self.bank.variablesOfWord(phoneme)
        assert len(variables) == 1
        phoneme = Constant(variables[0])
        condition(phoneme == indexWord(expression, index))
    def constrainUnderlyingRepresentation(self, stem, prefixes, suffixes, surfaces):
        # Remove what we can
        trimmed = []
        for prefix, suffix, surface in zip(prefixes, suffixes, surfaces):
            if surface == None: continue
            trimmed.append(surface[len(prefix) : len(surface) - len(suffix)])
        for j in range(99):
            if any(j >= len(t) for t in trimmed): break
            if all(trimmed[0][j] == t[j] for t in trimmed):
                self.conditionPhoneme(stem, Constant(j), trimmed[0][j])
            else: break
        for j in range(99):
            if any(j >= len(t) for t in trimmed): break
            if all(trimmed[0][len(trimmed[0]) - j - 1] == t[len(t) - j - 1] for t in trimmed):
                self.conditionPhoneme(stem,
                                      wordLength(stem) - (j + 1),
                                      trimmed[0][len(trimmed[0]) - j - 1])
                                      
            else: break
            

    def sortDataByLength(self):
        # Sort the data by length. Break ties by remembering which one originally came first.
        dataTaggedWithLength = [ (sum([ len(w) if w != None else 0 for w in self.data[j]]),
                                  j,
                                  self.data[j])
                                 for j in range(len(self.data)) ]
        self.data = [ d[2] for d in sorted(dataTaggedWithLength) ]


    def conditionOnStem(self, rules, stem, prefixes, suffixes, surfaces, auxiliaryHarness = False):
        """surfaces : list of numberOfInflections elements, each of which is a morph object"""
        assert self.numberOfInflections == len(surfaces)
        
        for i,surface in enumerate(surfaces):
            if surface == None: continue
            
            prediction = applyRules(rules,
                                    concatenate3(prefixes[i],stem,suffixes[i]),
                                    wordLength(prefixes[i]) + wordLength(stem),
                                    len(surface) + 1)
            predicate = wordEqual(surface.makeConstant(self.bank), prediction)
            if auxiliaryHarness: auxiliaryCondition(predicate)
            else: condition(predicate)
    
    def conditionOnData(self, rules, stems, prefixes, suffixes, observations = None, auxiliaryHarness = False):
        '''Conditions on inflection matrix.'''
        if observations == None: observations = self.data
        for stem, observation in zip(stems, observations):
            self.conditionOnStem(rules, stem, prefixes, suffixes, observation,
                                 auxiliaryHarness = auxiliaryHarness)
    
    def solveUnderlyingForms(self, solution):
        '''Takes in a solution w/o underlying forms, and gives the one that has underlying forms'''
        if solution.underlyingForms != [] and getVerbosity() > 0:
            print "WARNING: solveUnderlyingForms: Called with solution that already has underlying forms"

        return Solution(rules = solution.rules,
                        prefixes = solution.prefixes,
                        suffixes = solution.suffixes,
                        underlyingForms = [ solution.transduceUnderlyingForm(self.bank, inflections)
                                            for inflections in self.data ])

    def fastTopRules(self, solution, k, maximumNumberOfSolutions = None):
        if k == 1: return [solution]
        if maximumNumberOfSolutions != None:
            # enforce k^d < maximumNumberOfSolutions
            # k < maximumNumberOfSolutions**(1/d)
            k = int(min(k,maximumNumberOfSolutions**(1.0/k)))

        fs = self.solveFrontiers(solution,k)
        
        return [ Solution(underlyingForms = solution.underlyingForms,
                          prefixes = solution.prefixes, suffixes = solution.suffixes,
                          rules = list(rs))
                 for rs in itertools.product(*fs) ]
        

    def solveFrontiers(self, solution, k):
        '''Takes as input a "seed" solution, and solves for K rules for each rule in the original seed solution. Returns a list of len(solution.rules), each of which has k rules.'''
        if k == 1: return [[r] for r in solution.rules ]
        
        xs = [ solution.prefixes[i] + solution.underlyingForms[j] + solution.suffixes[i]
               for j in range(len(self.data))
               for i in range(self.numberOfInflections) ]
        untilSuffix = [ Constant(len(solution.prefixes[i] + solution.underlyingForms[j]))
                        for j in range(len(self.data))
                        for i in range(self.numberOfInflections) ]
        frontiers = []
        for r in solution.rules:
            ys = [ self.applyRuleUsingSketch(r,x,us)
                   for x,us in zip(xs,untilSuffix) ]
            alternatives = SupervisedProblem(zip(xs,untilSuffix,ys)).fastTopK(k, r)
            frontiers.append(alternatives)
            xs = ys

        return frontiers

    def solveTopRules(self, solution, k):
        '''Takes as input a "seed" solution, and expands it to k solutions with the same morphological cost'''
        solutions = [solution]
        
        for _ in range(k - 1):
            Model.Global()

            rules = [ Rule.sample() for _ in range(len(solution.rules)) ]
            for other in solutions:
                condition(And([ ruleEqual(r, o.makeConstant(self.bank))
                                for r, o in zip(rules, other.rules) ]) == 0)

            # Keep morphology variable! Just ensure it has the same cost
            prefixes = [ sampleMorphWithLength(len(p)) for p in solution.prefixes ]
            suffixes = [ sampleMorphWithLength(len(p)) for p in solution.suffixes ]
            stems = [ Morph.sample() for p in solution.underlyingForms ]
            
            self.conditionOnData(rules, stems, prefixes, suffixes)
            self.minimizeJointCost(rules, stems, prefixes, suffixes)

            try:
                output = self.solveSketch()
            except SynthesisFailure,SynthesisTimeout:
                if getVerbosity() > 0:
                    print "Found %d/%d solutions."%(len(solutions),k)
                break
            solutions.append(Solution(suffixes = [ Morph.parse(self.bank, output, m) for m in suffixes ],
                                      prefixes = [ Morph.parse(self.bank, output, m) for m in prefixes ],
                                      rules = [ Rule.parse(self.bank, output, r) for r in rules ]))
        return solutions

    def findCounterexamples(self, solution, trainingData = []):
        if getVerbosity() > 0:
            print "Beginning verification"
        for observation in self.data:
            if not self.verify(solution, observation):
                if observation in trainingData:
                    print "FATAL: Failed to verify ",observation,"which is in the training data."
                    print "The solution we were verifying was:"
                    print solution
                    assert False
                    continue
                print "COUNTEREXAMPLE:\t",
                for i in observation: print i,"\t",
                print ""
                yield observation

    def findCounterexample(self, solution, trainingData = []):
        # Returns the first counterexample or None if there are no counterexamples
        return next(self.findCounterexamples(solution, trainingData), None)

    def verify(self, solution, inflections):
        '''Checks whether the model can explain these inflections.
        If it can then it returns an underlying form consistent with the data.
        Otherwise it returns False.
        '''
        return solution.transduceUnderlyingForm(self.bank, inflections) != None
    
    def minimizeJointCost(self, rules, stems, prefixes, suffixes, costUpperBound = None, morphologicalCosts = None):
        if self.UG:
            self.UG.sketchUniversalGrammar(self.bank)
            
        # guess the size of each stem to be its corresponding smallest observation length
        if morphologicalCosts == None:
            approximateStemSize = [ min([ len(w) for w in i if w != None ])
                                    for i in self.data ]
        else:
            approximateStemSize = [ min([ len(w) - (0 if morphologicalCosts[j] == None else morphologicalCosts[j])
                                          for j,w in enumerate(i) if w != None ])
                                    for i in self.data ]
            
        affixAdjustment = []
        for j in range(self.numberOfInflections):
            adjustment = 0
            if morphologicalCosts != None and morphologicalCosts[j] != None: adjustment = morphologicalCosts[j]                
            elif self.numberOfInflections > 5: # heuristic: adjust when there are at least five inflections
                for Lex,stemSize in zip(self.data,approximateStemSize):
                    if Lex[j] != None:  # this lexeme was annotated for this inflection; use it as a guess
                        adjustment = len(Lex[j]) - stemSize
                        break
            else: adjustment = 0
            affixAdjustment.append(adjustment)
                
        affixSize = sum([ wordLength(prefixes[j]) + wordLength(suffixes[j]) - affixAdjustment[j]
                          for j in range(self.numberOfInflections) ])

        # We subtract a constant from the stems size in order to offset the cost
        # Should have no effect upon the final solution that we find,
        # but it lets sketch get away with having to deal with smaller numbers
        stemSize = sum([ wordLength(m)-
                         (approximateStemSize[j] if self.numberOfInflections > 1
                          else len(self.data[j][0]) - 4)
                         for j,m in enumerate(stems) ])

        ruleSize = sum([ruleCost(r) for r in rules ])
        totalCost = define("int",ruleSize + stemSize + affixSize)
        if costUpperBound != None:
            if getVerbosity() > 1: print "conditioning upon total cost being less than",costUpperBound
            condition(totalCost < costUpperBound)
        minimize(totalCost)

        return totalCost

    def sketchJointSolution(self, depth, canAddNewRules = False, costUpperBound = None,
                            fixedRules = None, fixedMorphology = None, auxiliaryHarness = False):
        try:
            Model.Global()
            if fixedRules == None:
                rules = [ Rule.sample() for _ in range(depth) ]
            else:
                rules = [ r.makeDefinition(self.bank) for r in fixedRules ]
            stems = [ Morph.sample() for _ in self.data ]
            prefixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]
            suffixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]

            if fixedMorphology != None:
                for p,k in zip(prefixes,fixedMorphology.prefixes):
                    condition(wordEqual(p,k.makeConstant(self.bank)))
                for s,k in zip(suffixes,fixedMorphology.suffixes):
                    condition(wordEqual(s,k.makeConstant(self.bank)))
            if self.wordBoundaries:
                for prefix, suffix in zip(prefixes, suffixes):
                    condition(Or([wordLength(prefix) == 0, wordLength(suffix) == 0]))

            morphologicalCosts = None
            if fixedMorphology: morphologicalCosts = [ len(prefix) + len(suffix)
                                                       for prefix, suffix in zip(fixedMorphology.prefixes,
                                                                                 fixedMorphology.suffixes) ]

            self.minimizeJointCost(rules, stems, prefixes, suffixes, costUpperBound, morphologicalCosts)

            self.conditionOnData(rules, stems, prefixes, suffixes,
                                 auxiliaryHarness = auxiliaryHarness)

            output = self.solveSketch()
            print "Final hole value:",parseMinimalCostValue(output)

            solution = Solution(prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ],
                                suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ],
                                underlyingForms = [ Morph.parse(self.bank, output, s) for s in stems ],
                                rules = [ Rule.parse(self.bank, output, r) for r in rules ] if fixedRules == None else fixedRules)
            solution.showMorphologicalAnalysis()
            solution.showRules()
            return solution
        
        except SynthesisFailure:
            if canAddNewRules:
                depth += 1
                print "Expanding rule depth to %d"%depth
                return self.sketchJointSolution(depth, canAddNewRules = canAddNewRules,
                                                auxiliaryHarness = auxiliaryHarness)
            else:
                return None
        # pass this exception onto the caller
        #except SynthesisTimeout:


    def counterexampleSolution(self, k = 1, threshold = float('inf'), initialTrainingSize = 2, fixedMorphology = None, initialDepth = 1, maximumDepth = 3):
        # Start out with the shortest examples
        #self.sortDataByLength()
        if self.numberOfInflections == 1 or initialTrainingSize == 0:
            initialTrainingSize = len(self.data)
        trainingData = self.data[:initialTrainingSize]

        depth = initialDepth

        solution = None

        while True:
            print "CEGIS: Training data:"
            print formatTable([ map(unicode,r) for r in trainingData ], separation = 1)

            solverTime = time() # time to sketch the solution
            # expand the rule set until we can fit the training data
            try:
                solution = self.restrict(trainingData).sketchJointSolution(depth, canAddNewRules = True, fixedMorphology = fixedMorphology, auxiliaryHarness = True)            
                depth = solution.depth() # update depth because it might have grown
                solverTime = time() - solverTime

                counterexample = self.findCounterexample(solution, trainingData)
                if counterexample != None:
                    trainingData.append(counterexample)
                    continue
            except SynthesisTimeout: return [solution] if solution else []
            
            # we found a solution that had no counterexamples
            #print "Final set of counterexamples:"
            #print latexMatrix(trainingData)

            # When we expect it to be tractable, we should try doing a little bit deeper
            if depth < maximumDepth and self.numberOfInflections < 3:
                slave = self.restrict(trainingData)
                try:
                    expandedSolution = slave.sketchJointSolution(depth + 1,
                                                                 fixedMorphology = fixedMorphology,
                                                                 auxiliaryHarness = True)
                except SynthesisTimeout: return [solution]
                if expandedSolution.cost() <= solution.cost():
                    solution = expandedSolution
                    print "Better compression achieved by expanding to %d rules"%(depth + 1)
                    depth += 1
                    try: counterexample = self.findCounterexample(expandedSolution, trainingData)
                    except SynthesisTimeout: return [solution]
                    
                    if counterexample != None:
                        trainingData.append(counterexample)
                        print "Despite being better, there is a counterexample; continue CEGIS"
                        continue # do another round of counterexample guided synthesis
                    else:
                        print "Also, expanded rules have no counter examples."
                else:
                    print "Sticking with depth of %d"%(depth)
                    
            print "Final solutions:"
            print solution
            try: solution = self.solveUnderlyingForms(solution)
            except SynthesisTimeout: return [solution]

            # Do we have enough time in our budget to not be fast?
            try:
                if solverTime*k < threshold:
                    solutions = self.solveTopRules(solution, k)
                else:
                    print "Using the optimized top rules."
                    solutions = self.fastTopRules(solution, k)
            except SynthesisTimeout: return [solution]

            return solutions

    def computeSolutionScores(self,solution,invariant):
        # Compute the description length of everything
        degree = numberOfCPUs()
        if hasattr(self,'numberOfCPUs'): degree = self.numberOfCPUs
        descriptionLengths = parallelMap(degree,
                                         lambda x: self.inflectionsDescriptionLength(solution, x), self.data)
        everythingCost = sum(descriptionLengths)
        invariantCost = sum([ len(u) for u in solution.underlyingForms ]) 
        return {'solution': solution,
                'modelCost': solution.modelCost(self.UG),
                'everythingCost': everythingCost,
                'invariantCost': invariantCost}

    def solutionDescriptionLength(self,solution,data = None):
        if data == None: data = self.data
        if getVerbosity() > 3:
            print "Calculating description length of:"
            print solution
        return sum([self.inflectionsDescriptionLength(solution,i)
                    for i in data ])
        
    def inflectionsDescriptionLength(self, solution, inflections):
        if getVerbosity() > 3: print "Transducing UR of:",u"\t".join(map(unicode,inflections))
        ur = solution.transduceUnderlyingForm(self.bank, inflections)
        if getVerbosity() > 3: print "\tUR = ",ur
        if ur != None:
            return len(ur)
        else:
            # Dumb noise model
            if True:
                return sum([ len(s) for s in inflections if s != None ])
            else:
                # Smart noise model
                # todo: we could also incorporate the morphology here if we wanted to
                subsequenceLength = multiLCS([ s.phonemes for s in inflections if s != None ])
                return sum([ len(s) - subsequenceLength for s in inflections if s != None ])
    

    def paretoFront(self, depth, k, temperature, useMorphology = False):
        assert self.numberOfInflections == 1
        self.maximumObservationLength += 1

        def affix():
            if useMorphology: return Morph.sample()
            else: return Morph([]).makeConstant(self.bank)
        def parseAffix(output, morph):
            if useMorphology: return Morph.parse(self.bank, output, morph)
            else: return Morph([])
            
        Model.Global()
        rules = [ Rule.sample() for _ in range(depth) ]

        stems = [ Morph.sample() for _ in self.data ]
        prefixes = [ affix() for _ in range(self.numberOfInflections) ]
        suffixes = [ affix() for _ in range(self.numberOfInflections) ]

        for i in range(len(stems)):
            self.conditionOnStem(rules, stems[i], prefixes, suffixes, self.data[i])

        morphologicalCoefficient = 3
        stemCostExpression = sum([ wordLength(u) for u in stems ])
        stemCostVariable = unknownInteger(numberOfBits = 6)
        condition(stemCostVariable == stemCostExpression)
        minimize(stemCostExpression)
        ruleCostExpression = sum([ ruleCost(r) for r in rules ] + [ wordLength(u)*morphologicalCoefficient for u in suffixes + prefixes ])
        ruleCostVariable = unknownInteger()
        condition(ruleCostVariable == ruleCostExpression)
        if len(rules) > 0 or useMorphology:
            minimize(ruleCostExpression)

        solutions = []
        solutionCosts = []
        for _ in range(k):
            # Excludes solutions we have already found
            for rc,uc in solutionCosts:
                if len(solutions) < k/2:
                    # This condition just says that it has to be a
                    # different trade-off. Gets things a little bit off of
                    # the front
                    condition(And([ruleCostVariable == rc,stemCostVariable == uc]) == 0)
                else:
                    # This condition says that it has to actually be on
                    # the pareto - a stronger constraint
                    condition(Or([ruleCostVariable < rc, stemCostVariable < uc]))

            try:
                output = self.solveSketch(minimizeBound = 64)
            except SynthesisFailure:
                print "Exiting Pareto procedure early due to unsatisfied"
                break
            except SynthesisTimeout:
                print "Exiting Pareto procedure early due to timeout"
                break

            s = Solution(suffixes = [ parseAffix(output, m) for m in suffixes ],
                         prefixes = [ parseAffix(output, m) for m in prefixes ],
                         rules = [ Rule.parse(self.bank, output, r) for r in rules ],
                         underlyingForms = [ Morph.parse(self.bank, output, m) for m in stems ])
            solutions.append(s)
            print s

            rc = sum([r.cost() for r in s.rules ] + [len(a)*morphologicalCoefficient for a in s.prefixes + s.suffixes ])
            uc = sum([len(u) for u in s.underlyingForms ])
            print "Costs:",(rc,uc)
            actualCosts = (parseInteger(output, ruleCostVariable), parseInteger(output, stemCostVariable))
            print "Actual costs:",actualCosts
            if not (actualCosts == (rc,uc)):
                print output
            assert actualCosts == (rc,uc)
            (rc,uc) = actualCosts
            solutionCosts.append((rc,uc))

        print " pareto: got %d solutions of depth %d"%(len(solutions),depth)
        
        if len(solutions) > 0:
            optimalCost, optimalSolution = min([(uc + float(rc)/temperature, s)
                                                for ((rc,uc),s) in zip(solutionCosts, solutions) ])
            print "Optimal solution:"
            print optimalSolution
            print "Optimal cost:",optimalCost

        return solutions, solutionCosts

    def stochasticSearch(self, iterations, width):
        population = [Solution([EMPTYRULE],
                               [Morph([])]*self.numberOfInflections,
                               [Morph([])]*self.numberOfInflections)]
        for i in range(iterations):
            # expand the population
            children = [ parent.mutate(self.bank)
                         for parent in population
                         for _ in range(width) ]
            population += children
            populationScores = [ (self.solutionDescriptionLength(s) + s.modelCost(),s)
                                 for s in population ]
            populationScores.sort()
            population = [ s
                           for _,s in populationScores[:width] ]
            setVerbosity(4)
            mdl = self.solutionDescriptionLength(population[0])
            setVerbosity(0)
            print "MDL:",mdl+population[0].modelCost()

    def restrict(self, newData):
        """Creates a new version of this object which is identical but has different training data"""
        restriction = copy.copy(self)
        restriction.data = [ [ None if i == None else (i if isinstance(i,Morph) else Morph(tokenize(i)))
                               for i in Lex] for Lex in newData ]
        return restriction


if __name__ == '__main__':
    from parseSPE import parseSolution
    from problems import sevenProblems

    s = parseSolution(sevenProblems[1].solutions[0])
    solver = UnderlyingProblem(sevenProblems[1].data)

    solver.debugSolution(s,Morph(tokenize(u"koš^yil^y")))
    
