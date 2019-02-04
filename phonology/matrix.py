# -*- coding: utf-8 -*-

from result import *
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
    def __init__(self, data, problemName=None, bank = None, useSyllables = False, UG = None,
                 fixedMorphology = None):
        self.problemName = problemName
        self.UG = UG

      
        if bank != None: self.bank = bank
        else:
            self.bank = FeatureBank([ w for l in data for w in l if w != None ] + ([u'-'] if useSyllables else []))

        self.numberOfInflections = len(data[0])
        for d in data: assert len(d) == self.numberOfInflections
        
        # wrap the data in Morph objects if it isn't already
        self.data = [ tuple( None if i == None else (i if isinstance(i,Morph) else Morph(tokenize(i)))
                             for i in Lex)
                      for Lex in data ]

        self.maximumObservationLength = max([ len(w) for l in self.data for w in l if w != None ])

        self.wordBoundaries = any([ (u'##' in w.phonemes) for l in self.data for w in l if w ])

        # fixedMorphology : list of morphologies, one for each inflection
        # Each morphology is either None (don't fix it) or a pair of (prefix, suffix)
        if fixedMorphology == None: fixedMorphology = [None]*self.numberOfInflections
        self.fixedMorphology = fixedMorphology
        assert len(self.fixedMorphology) == self.numberOfInflections

        self.pervasiveTimeout = None

        self.precomputedAlignment = None

    def loadAlignment(self, fn):
        print " [+] Loaded the following alignment from",fn
        self.precomputedAlignment = loadPickle(fn)
        print self.precomputedAlignment

    def solveSketch(self, minimizeBound = 31, maximumMorphLength=None):
        if maximumMorphLength is None: maximumMorphLength = self.maximumObservationLength
        return solveSketch(self.bank,
                           # unroll: +1 for extra UR size, +1 for guard buffer
                           self.maximumObservationLength + 2,
                           # maximum morpheme size
                           maximumMorphLength,
                           showSource = False, minimizeBound = minimizeBound,
                           timeout=self.pervasiveTimeout)

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
                                             Constant(untilSuffix), len(u) + 2)))
        try:
            output = self.solveSketch(maximumMorphLength=len(u) + 2)
        except SynthesisFailure:
            print "applyRuleUsingSketch: UNSATISFIABLE for %s %s %s"%(u,r,untilSuffix)
            printSketchFailure()
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
    def constrainUnderlyingRepresentation(self, stem, prefixes, suffixes, surfaces, verbose=True):
        '''stem: sketch variable
        prefixes, suffixes, surfaces: Morph'''
        # Remove what we can
        trimmed = []
        for prefix, suffix, surface in zip(prefixes, suffixes, surfaces):
            if surface == None or suffix == None or prefix == None: continue
            trimmed.append(surface[len(prefix) : len(surface) - len(suffix)])
        if len(trimmed) < 2: return

        for j in range(99):
            if any(j >= len(t) for t in trimmed): break
            if all(trimmed[0][j] == t[j] for t in trimmed):
                self.conditionPhoneme(stem, Constant(j), trimmed[0][j])
            else: break
        if verbose:
            print "Constraining underlying prefix for %s to %s"%(u" ~ ".join(map(unicode,surfaces)), trimmed[0][:j])
        
        for j in range(99):
            if any(j >= len(t) for t in trimmed): break
            if all(trimmed[0][len(trimmed[0]) - j - 1] == t[len(t) - j - 1] for t in trimmed):
                self.conditionPhoneme(stem,
                                      wordLength(stem) - (j + 1),
                                      trimmed[0][len(trimmed[0]) - j - 1])
                                      
            else: break
        if verbose:
            print "Constraining underlying suffix for %s to %s"%(u" ~ ".join(map(unicode,surfaces)),
                                                                 trimmed[0][::-1][:j][::-1])
            

    def sortDataByLength(self):
        # Sort the data by length. Break ties by remembering which one originally came first.
        dataTaggedWithLength = [ (sum([ len(w) if w != None else 0 for w in self.data[j]]),
                                  j,
                                  self.data[j])
                                 for j in range(len(self.data)) ]
        self.data = [ tuple(d[2]) for d in sorted(dataTaggedWithLength) ]

    def excludeBoundaryAndInsertions(self, rules):
        """
        insertions/deletions can destroy context that would tell us where the morpheme boundaries are;
        therefore you cannot have insertion/deletion and also have boundary rules
        """
        clauses = []
        for r in rules:
            clauses.append(isInsertionRule(r))
            clauses.append(isDeletionRule(r))
        anyInsertion = Or(clauses)
        for r in rules:
            condition(Not(And([anyInsertion, isBoundaryRule(r)])))


    def conditionOnStem(self, rules, stem, prefixes, suffixes, surfaces, auxiliaryHarness = False):
        """surfaces : list of numberOfInflections elements, each of which is a morph object"""
        assert self.numberOfInflections == len(surfaces)

        if self.wordBoundaries:
            wb = Constant(self.bank.variablesOfWord(u'##')[0])
            for i in xrange(max(len(s) for s in surfaces if s is not None )):
                condition(Or([Constant(i) >= wordLength(stem),
                              indexWord(stem, Constant(i)) != wb]))
        
        for i,surface in enumerate(surfaces):
            if surface == None: continue
            if self.wordBoundaries: condition(wordLength(prefixes[i]) == 0)
            
            prediction = applyRules(rules,
                                    concatenate3(prefixes[i],stem,suffixes[i]),
                                    wordLength(prefixes[i]) + wordLength(stem),
                                    len(surface) + 1)
            predicate = wordEqual(surface.makeConstant(self.bank), prediction)
            if auxiliaryHarness: auxiliaryCondition(predicate)
            else: condition(predicate)
    def conditionOnStem_1a(self, rules, stem, prefixes, suffixes, surfaces):
        """Exactly one of the surface forms will be not an auxiliarycondition
        surfaces : list of numberOfInflections elements, each of which is a morph object"""
        assert self.numberOfInflections == len(surfaces)

        conditions = [(surface, prefix, suffix)
                      for surface, prefix, suffix in zip(surfaces, prefixes, suffixes)
                      if surface is not None ]
        conditions = randomlyPermute(conditions)
        
        for i,(surface,prefix,suffix) in enumerate(conditions):
            prediction = applyRules(rules,
                                    concatenate3(prefix,stem,suffix),
                                    wordLength(prefix) + wordLength(stem),
                                    len(surface) + 1)
            predicate = wordEqual(surface.makeConstant(self.bank), prediction)
            if i > 0: auxiliaryCondition(predicate)
            else: condition(predicate)
    
    def conditionOnData(self, rules, stems, prefixes, suffixes, observations = None, auxiliaryHarness = False):
        '''Conditions on inflection matrix.'''
        if observations == None: observations = self.data
        for stem, observation in zip(stems, observations):
            self.conditionOnStem(rules, stem, prefixes, suffixes, observation,
                                 auxiliaryHarness = auxiliaryHarness)
            if self.precomputedAlignment is not None and observation in self.precomputedAlignment.underlyingForms:
                pattern = self.precomputedAlignment.underlyingForms[observation]
                condition(pattern.match(self.bank,stem))

    def conditionOnPrecomputedMorphology(self, prefixes, suffixes):
        if self.precomputedAlignment is None: return
        for i in xrange(self.numberOfInflections):
            if any( ss[i] is not None for ss in self.data ):
                condition(self.precomputedAlignment.prefixes[i].match(self.bank, prefixes[i]))
                condition(self.precomputedAlignment.suffixes[i].match(self.bank, suffixes[i]))
                
    
    def solveUnderlyingForms(self, solution, batchSize = 10):
        '''Takes in a solution w/o underlying forms, and gives the one that has underlying forms'''
        if len(solution.underlyingForms) != 0 and getVerbosity() > 0:
            print "WARNING: solveUnderlyingForms: Called with solution that already has underlying forms"

        return solution.transduceManyStems(self.bank, self.data, batchSize = batchSize)

    def expandFrontier(self, solution, k, CPUs = None):
        '''Takes as input a "seed" solution, and solves for K rules for each rule in the original seed solution. Returns a Frontier object.'''
        if k == 1: return solution.toFrontier()

        CPUs = CPUs or numberOfCPUs()

        # Construct the training data for each rule
        xs = []
        ys = []
        
        xs.append([ solution.prefixes[i] + solution.underlyingForms[x] + solution.suffixes[i]
                    for x in self.data
                    for i in range(self.numberOfInflections) ])
        untilSuffix = [ Constant(len(solution.prefixes[i] + solution.underlyingForms[x]))
                        for x in self.data
                        for i in range(self.numberOfInflections) ]
        for r in solution.rules:
            assert len(xs[-1]) == len(untilSuffix)
            ys.append(parallelMap(CPUs, lambda (x,us): self.applyRuleUsingSketch(r,x,us),
                                  zip(xs[-1],untilSuffix) ))
            xs.append(ys[-1])
        for x,y in zip(xs, ys):
            print "Training data for rule:"
            for a,b in zip(x,y):
                print a," > ",b

        # Now that we have the training data, we can solve for each of the rules' frontier
        frontiers = parallelMap(CPUs, lambda (j,r): SupervisedProblem(zip(xs[j],untilSuffix,ys[j])).topK(k,r),
                                enumerate(solution.rules))

        return Frontier(frontiers,
                        prefixes = solution.prefixes,
                        suffixes = solution.suffixes,
                        underlyingForms = solution.underlyingForms)
                        


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
        '''morphologicalCosts: a list of integers. Each integer is a guess as
to the total size of the morphology for that inflection. If instead
the integer is None then we have no guess for that one.'''
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
                          else len(self.data[j][0]) - 1)
                         for j,m in enumerate(stems) ])

        ruleSize = sum([ruleCost(r) for r in rules ])
        totalCost = define("int",ruleSize + stemSize + affixSize)
        if costUpperBound != None:
            if getVerbosity() > 1: print "conditioning upon total cost being less than",costUpperBound
            condition(totalCost < costUpperBound)
        minimize(totalCost)

        return totalCost

    def sketchJointSolution(self, depth, canAddNewRules = False, costUpperBound = None,
                            fixedRules = None, auxiliaryHarness = False):
        try:
            Model.Global()
            if fixedRules == None:
                rules = [ Rule.sample() for _ in range(depth) ]
            else:
                rules = [ r.makeDefinition(self.bank) for r in fixedRules ]
            stems = [ Morph.sample() for _ in self.data ]
            prefixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]
            suffixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]

            for j,m in enumerate(self.fixedMorphology):
                if m != None:
                    (p,s) = m
                    condition(wordEqual(prefixes[j],p.makeConstant(self.bank)))
                    condition(wordEqual(suffixes[j],s.makeConstant(self.bank)))
            if self.wordBoundaries:
                for prefix, suffix in zip(prefixes, suffixes):
                    condition(Or([wordLength(prefix) == 0, wordLength(suffix) == 0]))

            morphologicalCosts = [ None if m == None else len(m[0]) + len(m[1])
                                   for m in self.fixedMorphology ]

            self.minimizeJointCost(rules, stems, prefixes, suffixes, costUpperBound, morphologicalCosts)

            self.conditionOnData(rules, stems, prefixes, suffixes,
                                 auxiliaryHarness = auxiliaryHarness)
            self.conditionOnPrecomputedMorphology(prefixes, suffixes)

            output = self.solveSketch()
            print "Final hole value:",parseMinimalCostValue(output)

            solution = Solution(prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ],
                                suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ],
                                underlyingForms = {x: Morph.parse(self.bank, output, s)
                                                   for x,s in zip(self.data, stems) },
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


    def counterexampleSolution(self, k = 1, threshold = float('inf'), initialTrainingSize = 2, initialDepth = 1, maximumDepth = 3):
        result = Result(self.problemName)
        
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
                solution = self.restrict(trainingData).sketchJointSolution(depth, canAddNewRules = True, auxiliaryHarness = True)
                result.recordSolution(solution)
                depth = solution.depth() # update depth because it might have grown
                solverTime = time() - solverTime

                counterexample = self.findCounterexample(solution, trainingData)
                if counterexample != None:
                    trainingData.append(counterexample)
                    continue
            except SynthesisTimeout: return result.lastSolutionIsFinal()
            
            # we found a solution that had no counterexamples

            # When we expect it to be tractable, we should try doing a little bit deeper
            if depth < maximumDepth and self.numberOfInflections < 3:
                worker = self.restrict(trainingData)
                try:
                    expandedSolution = worker.sketchJointSolution(depth + 1,
                                                                  auxiliaryHarness = True)
                except SynthesisTimeout: return solution.toFrontier()
                if not any( r.doesNothing() for r in expandedSolution.rules ) and \
                   expandedSolution.cost() <= solution.cost():
                    solution = expandedSolution
                    result.recordSolution(solution)
                    print "Better compression achieved by expanding to %d rules"%(depth + 1)
                    depth += 1
                    try: counterexample = self.findCounterexample(expandedSolution, trainingData)
                    except SynthesisTimeout: return result.lastSolutionIsFinal()
                    
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
            except SynthesisTimeout: return result.lastSolutionIsFinal()

            result.recordFinalFrontier(self.expandFrontier(solution, k))
            return result

    def computeSolutionScores(self,solution,invariant):
        # Compute the description length of everything
        degree = numberOfCPUs()
        if hasattr(self,'numberOfCPUs'): degree = self.numberOfCPUs
        descriptionLengths = parallelMap(degree,
                                         lambda x: self.inflectionsDescriptionLength(solution, x), self.data)
        everythingCost = sum(descriptionLengths)
        invariantCost = sum([ len(u) for u in solution.underlyingForms.values() ]) 
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
    

    def paretoFront(self, depth, k, temperature, useMorphology = False,
                    morphologicalCoefficient = 3):
        # no idea why we want this
        #self.maximumObservationLength += 1

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
            self.conditionOnStem_1a(rules, stems[i], prefixes, suffixes, self.data[i])
        # actually we want this
        #for r in rules: condition(Not(ruleDoesNothing(r)))

        stemCostExpression = sum([ wordLength(u) for u in stems ])
        stemCostVariable = unknownInteger(numberOfBits = 7)
        condition(stemCostVariable == stemCostExpression)
        minimize(stemCostExpression - 50)
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
                if len(solutions) < k/2 and False:
                    # This condition just says that it has to be a
                    # different trade-off. Gets things a little bit off of
                    # the front
                    condition(And([ruleCostVariable == rc,stemCostVariable == uc]) == 0)
                else:
                    # This condition says that it has to actually be on
                    # the pareto - a stronger constraint
                    condition(Or([ruleCostVariable < rc, stemCostVariable < uc]))

            try:
                output = self.solveSketch(minimizeBound = 31)
            except SynthesisFailure:
                print "Exiting Pareto procedure early due to unsatisfied"
                break
            except SynthesisTimeout:
                print "Exiting Pareto procedure early due to timeout"
                break

            s = Solution(suffixes = [ parseAffix(output, m) for m in suffixes ],
                         prefixes = [ parseAffix(output, m) for m in prefixes ],
                         rules = [ Rule.parse(self.bank, output, r) for r in rules ],
                         underlyingForms = {x: Morph.parse(self.bank, output, m)
                                            for x,m in zip(self.data, stems) }).withoutUselessRules()
            solutions.append(s)
            print s

            rc = sum([r.cost() for r in s.rules ] + [len(a)*morphologicalCoefficient for a in s.prefixes + s.suffixes ])
            uc = sum([len(u) for u in s.underlyingForms.values() ])
            rc = int(rc + 0.5)
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

    def lesionMorphologicalRules(self, solution):
        """Sometimes we end up learning to put the morphology into the rewrite
        rules, e.g. 0 > k/#_ or something like that. This will take a
        solution and try removing insertion/deletion rules whenever
        possible, keeping the underlying forms constant but being
        willing to modify the morphology.
        """
        rules = list(solution.rules)
        for r in list(solution.rules):
            if isinstance(r.focus, EmptySpecification) and isinstance(r.structuralChange, ConstantPhoneme) and \
               u'#' in unicode(r):
                print "Candidate for lesion",r
                candidateRules = [ r_ for r_ in rules if r_ != r ]
                print "the new rules would be",candidateRules
                Model.Global()
                prefixes = [ Morph.sample() for _ in xrange(self.numberOfInflections) ]
                suffixes = [ Morph.sample() for _ in xrange(self.numberOfInflections) ]
                self.conditionOnData([ r_.makeConstant(self.bank) for r_ in candidateRules ],
                                     [ solution.underlyingForms[x].makeConstant(self.bank)
                                       for x in self.data ],
                                     prefixes, suffixes,
                                     auxiliaryHarness=True)
                #minimize(sum(wordLength(m) for m in prefixes+suffixes ))
                try:
                    output = self.solveSketch()
                    print "Lesioning morphological rule", r
                    solution = Solution(prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ],
                                        suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ],
                                        underlyingForms = solution.underlyingForms,
                                        rules = candidateRules)
                    rules = solution.rules
                except SynthesisFailure:
                    print "Turns out that you cannot lesion",r
        return solution
            
        

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
        restriction.data = [ tuple( None if i == None else (i if isinstance(i,Morph) else Morph(tokenize(i)))
                               for i in Lex)
                             for Lex in newData ]
        return restriction


