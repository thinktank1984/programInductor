# -*- coding: utf-8 -*-

from utilities import *
from solution import *
from features import FeatureBank, tokenize
from rule import Rule,Guard,FeatureMatrix,EMPTYRULE
from morph import Morph
from sketchSyntax import Expression
from sketch import *
from supervised import solveTopSupervisedRules
from latex import latexMatrix
from UG import str2ug #FlatUG, ChomskyUG, FeatureUG, SkeletonUG, SkeletonFeatureUG


import random
import sys
import pickle
import math
from time import time

USEPYTHONRULES = False

class SynthesisFailure(Exception):
    pass

def sampleMorphWithLength(l):
    m = Morph.sample()
    condition(wordLength(m) == l)
    return m
        

class UnderlyingProblem():
    def __init__(self, data, depth, bank = None):
        self.depth = depth
        self.data = data
        self.bank = bank if bank != None else FeatureBank([ w for l in data for w in l  ])

        self.numberOfInflections = len(data[0])
        self.inflectionMatrix = [ [ self.bank.wordToMatrix(i) for i in Lex ] for Lex in data ]

        self.maximumObservationLength = max([ len(tokenize(w)) for l in data for w in l ])
        self.maximumMorphLength = max(10,self.maximumObservationLength - 2)

    def solveSketch(self):
        return solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength, showSource = False)

    def applyRule(self, r, u):
        if USEPYTHONRULES:# use the Python implementation of rules
            return Morph.fromMatrix(r.apply(u))
        else:
            Model.Global()
            result = Morph.sample()
            _r = r.makeDefinition(self.bank) #define("Rule", r.makeConstant(self.bank))
            fixStructuralChange(_r)
            condition(wordEqual(result,applyRule(_r,u.makeConstant(self.bank))))
            # IMPORTANT!
            # if the result is a more than we need to make sure that morphs can be big
            output = solveSketch(self.bank, self.maximumObservationLength, self.maximumObservationLength)
            if output != None:
                return Morph.parse(self.bank, output, result)
            else:
                print "WARNING: Gets rejected in applyRule. Falling back on Python implementation."
                print u
                print r
                printSketchFailure()
                # Weaker test
                Model.Global()
                condition(wordLength(applyRule(r.makeConstant(self.bank),u.makeConstant(self.bank))) > 0)
                if solveSketch(self.bank, self.maximumObservationLength, self.maximumObservationLength) == None:
                    print "WARNING: weaker test also fails"
                else:
                    print "WARNING: weaker test succeeds"
                return Morph.fromMatrix(r.apply(u))

    def sortDataByLength(self):
        # Sort the data by length. Break ties by remembering which one originally came first.
        dataTaggedWithLength = [ (sum(map(compose(len,tokenize),self.data[j])), j, self.data[j])
                                 for j in range(len(self.data)) ]
        self.data = [ d[2] for d in sorted(dataTaggedWithLength) ]


    def conditionOnStem(self, rules, stem, prefixes, suffixes, surfaces):
        """surfaces : list of elements, each of which is either a sketch expression or a APA string"""
        
        def buildUnderlyingForm(prefix, suffix):
            if isinstance(stem, Morph): # underlying form is fixed
                return (prefix + stem + suffix).makeConstant(self.bank)
            else: # underlying form is unknown
                return concatenate3(prefix, stem, suffix)
            
        prediction = [ applyRules(rules, buildUnderlyingForm(prefixes[i],suffixes[i]))
                     for i in range(len(surfaces)) ]
        for i in range(len(surfaces)):
            surface = surfaces[i]
            if not isinstance(surface,Expression):
                surface = makeConstantWord(self.bank, surface)
            condition(wordEqual(surface, prediction[i]))
    
    def conditionOnData(self, rules, stems, prefixes, suffixes):
        '''Conditions on inflection matrix. This also modifies the rules in place! Always call this after calculating the cost of the rules.'''
        for r in rules:
            condition(fixStructuralChange(r))
        for i in range(len(stems)):
            self.conditionOnStem(rules, stems[i], prefixes, suffixes, self.data[i])
    
    def solveUnderlyingForms(self, solution):
        '''Takes in a solution w/o underlying forms, and gives the one that has underlying forms'''
        if solution.underlyingForms != []:
            print "WARNING: solveUnderlyingForms: Called with solution that already has underlying forms"
            
        Model.Global()
        rules_ = [ define("Rule", r.makeConstant(self.bank)) for r in solution.rules ]
        prefixes_ = [ define("Word", p.makeConstant(self.bank)) for p in solution.prefixes ]
        suffixes_ = [ define("Word", s.makeConstant(self.bank)) for s in solution.suffixes ]
        stems = [ Morph.sample() for _ in self.inflectionMatrix ]
        self.conditionOnData(rules_, stems, prefixes_, suffixes_)

        for stem in stems:
            minimize(wordLength(stem))
        
        output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
        if not output:
            print "FATAL: Failed underlying form analysis"
            for observation in self.data:
                stem = self.verify(prefixes, suffixes, rules, observation)
                print "Verification of",observation
                print "\tstem =",stem
                if stem == False: print "\t(FAILURE)"
            raise SynthesisFailure("Failed at underlying form analysis.")

        us = [ Morph.parse(self.bank, output, s) for s in stems ]

        # The only purpose of this nested loop is to verify that there are no bugs
        for j in range(len(self.inflectionMatrix)):
            for i in range(self.numberOfInflections):
                u = solution.prefixes[i] + us[j] + solution.suffixes[i]
                for r in solution.rules:
                    #print "Applying",r,"to",u,"gives",r.apply(u),"aka",Morph.fromMatrix(r.apply(u))
                    u = self.applyRule(r,u)
                # print Morph.fromMatrix(u),"\n",Morph(tokenize(self.data[j][i]))
                if Morph(tokenize(self.data[j][i])) != u:
                    print "underlying:",solution.prefixes[i] + us[j] + solution.suffixes[i]
                    print Morph(tokenize(self.data[j][i])), "versus", u
                    print Morph(tokenize(self.data[j][i])).phonemes, "versus", u.phonemes
                    assert False

        return Solution(rules = solution.rules,
                        prefixes = solution.prefixes,
                        suffixes = solution.suffixes,
                        underlyingForms = us)

    def fastTopRules(self, solution, k):
        inputs = [ solution.prefixes[i] + solution.underlyingForms[j] + solution.suffixes[i]
                   for j in range(len(self.data))
                   for i in range(self.numberOfInflections) ]

        def f(xs, rs):
            if rs == []: return [[]]
            ys = [ Morph.fromMatrix(rs[0].apply(x)) for x in xs ]            
            alternatives = solveTopSupervisedRules(zip(xs,ys), k, rs[0])
            suffixes = f(ys, rs[1:])
            return [ [a] + s
                     for a in alternatives
                     for s in suffixes ]

        return [ Solution(prefixes = solution.prefixes,
                          suffixes = solution.suffixes,
                          underlyingForms = solution.underlyingForms,
                          rules = rs)
                 for rs in f(inputs, solution.rules) ]
        

    def solveTopRules(self, solution, k):
        '''Takes as input a "seed" solution, and expands it to k solutions with the same morphological cost'''
        solutions = [solution]
        
        for _ in range(k - 1):
            Model.Global()

            rules = [ Rule.sample() for _ in range(len(solution.rules)) ]
            for other in solutions:
                condition(And([ ruleEqual(r, o.makeConstant(self.bank))
                                for r, o in zip(rules, other.rules) ]) == 0)

            minimize(sum([ ruleCost(r) for r in rules ]))

            # Keep morphology variable! Just ensure it has the same cost
            prefixes = [ sampleMorphWithLength(len(p)) for p in solution.prefixes ]
            suffixes = [ sampleMorphWithLength(len(p)) for p in solution.suffixes ]
            stems = [ sampleMorphWithLength(len(p)) for p in solution.underlyingForms ]
            self.conditionOnData(rules, stems, prefixes, suffixes)
            
            output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
            if not output:
                print "Found %d/%d solutions."%(len(solutions),k)
                break
            solutions.append(Solution(suffixes = [ Morph.parse(self.bank, output, m) for m in suffixes ],
                                      prefixes = [ Morph.parse(self.bank, output, m) for m in prefixes ],
                                      rules = [ Rule.parse(self.bank, output, r) for r in rules ]))
        return solutions

    def findCounterexamples(self, solution, trainingData = []):
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
        Model.Global()

        stem = Morph.sample()

        # Make the morphology/phonology be a global definition
        prefixes = [ define("Word", p.makeConstant(self.bank)) for p in solution.prefixes ]
        suffixes = [ define("Word", s.makeConstant(self.bank)) for s in solution.suffixes ]
        rules = [ define("Rule", r.makeConstant(self.bank)) for r in solution.rules ]

        for r in rules: condition(fixStructuralChange(r))

        self.conditionOnStem(rules, stem, prefixes, suffixes, inflections)

        output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
        if output == None:
            return False
        else:
            return Morph.parse(self.bank, output, stem)

    def minimizeJointCost(self, rules, stems, prefixes, suffixes, costUpperBound = None):
        affixSize = sum([ wordLength(prefixes[j]) + wordLength(suffixes[j]) -
                          (len(self.inflectionMatrix[0][j]) - min(map(len, self.inflectionMatrix[0])) if self.numberOfInflections > 5 else 0)
                          for j in range(self.numberOfInflections) ])

        # We subtract a constant from the stems size in order to offset the cost
        # Should have no effect upon the final solution that we find,
        # but it lets sketch get away with having to deal with smaller numbers
        stemSize = sum([ wordLength(m)-
                         (min(map(len,self.inflectionMatrix[j])) if self.numberOfInflections > 1
                          else len(self.inflectionMatrix[j][0]) - 4)
                         for j,m in enumerate(stems) ])

        ruleSize = sum([ruleCost(r) for r in rules ])
        totalCost = define("int",ruleSize + stemSize + affixSize)
        if costUpperBound != None:
            print "conditioning upon total cost being less than",costUpperBound
            condition(totalCost < costUpperBound)
        minimize(totalCost)

        return totalCost

    def sketchJointSolution(self, canAddNewRules = False, costUpperBound = None, fixedRules = None):
        try:
            Model.Global()
            if fixedRules == None:
                rules = [ Rule.sample() for _ in range(self.depth) ]
            else:
                rules = [ r.makeDefinition(self.bank) for r in fixedRules ]
            stems = [ Morph.sample() for _ in self.inflectionMatrix ]
            prefixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]
            suffixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]

            self.minimizeJointCost(rules, stems, prefixes, suffixes, costUpperBound)

            self.conditionOnData(rules, stems, prefixes, suffixes)

            output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength, showSource = False)
            if not output:
                raise SynthesisFailure("Failed at morphological analysis.")

            solution = Solution(prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ],
                                suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ],
                                underlyingForms = [ Morph.parse(self.bank, output, s) for s in stems ],
                                rules = [ Rule.parse(self.bank, output, r) for r in rules ] if fixedRules == None else fixedRules)
            solution.showMorphologicalAnalysis()
            solution.showRules()
            return solution
        
        except SynthesisFailure:
            if canAddNewRules:
                self.depth += 1
                print "Expanding rule depth to %d"%self.depth
                return self.sketchJointSolution(canAddNewRules = canAddNewRules)
            else:
                return None


    def counterexampleSolution(self, k = 1, threshold = float('inf'), initialTrainingSize = 2):
        # Start out with the shortest examples
        self.sortDataByLength()
        if self.numberOfInflections == 1 or initialTrainingSize == 0:
            initialTrainingSize = len(self.data)
        trainingData = self.data[:initialTrainingSize]

        while True:
            print "CEGIS: Training data:"
            for r in trainingData:
                for i in r: print i,
                print ""

            solverTime = time() # time to sketch the solution
            # expand the rule set until we can fit the training data
            solution = UnderlyingProblem(trainingData, self.depth, self.bank).sketchJointSolution(canAddNewRules = True)
            self.depth = solution.depth() # update depth because it might have grown
            solverTime = time() - solverTime

            counterexample = self.findCounterexample(solution, trainingData)
            if counterexample != None:
                trainingData.append(counterexample)
                continue
            
            # we found a solution that had no counterexamples
            print "Final set of counterexamples:"
            print latexMatrix(trainingData)

            # When we expect it to be tractable, we should try doing a little bit deeper
            if self.depth < 3 and self.numberOfInflections < 3:
                slave = UnderlyingProblem(trainingData, self.depth + 1, self.bank)
                expandedSolution = slave.sketchJointSolution(costUpperBound = solution.adjustedCost)
                if expandedSolution.cost() <= solution.cost():
                    solution = expandedSolution
                    print "Better compression achieved by expanding to %d rules"%(self.depth + 1)
                    self.depth += 1
                    counterexample = self.findCounterexample(prefixes, suffixes, rules, trainingData)
                    if counterexample != None:
                        trainingData.append(counterexample)
                        print "Despite being better, there is a counterexample; continue CEGIS"
                        continue # do another round of counterexample guided synthesis
                    else:
                        print "Also, expanded rules have no counter examples."
                else:
                    print "Sticking with depth of %d"%(self.depth)
                    
            print "Final solutions:"
            print solution
            solution = self.solveUnderlyingForms(solution)

            # Do we have enough time in our budget to not be fast?
            if solverTime*k < threshold:
                solutions = self.solveTopRules(solution, k)
            else:
                print "Using the optimized top rules."
                solutions = self.fastTopRules(solution, k)

            return solutions


    def sketchChangeToSolution(self,solution, rules, costUpperBound = None):
        # costUpperBound: an upper bound on the cost; includes fancy index manipulations
        Model.Global()

        originalRules = list(rules) # save it for later
        
        rules = [ (rule.makeDefinition(self.bank) if rule != None else Rule.sample())
                  for rule in rules ]
        stems = [ Morph.sample() for _ in self.inflectionMatrix ]
        prefixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]
        suffixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]

        print "upper bound = ",costUpperBound
        self.minimizeJointCost(rules, stems, prefixes, suffixes, costUpperBound)
        self.conditionOnData(rules, stems, prefixes, suffixes)

        output = self.solveSketch()
        if output == None:
            print "\t(no modification possible)"
            raise SynthesisFailure("No satisfying modification possible.")

        loss = parseMinimalCostValue(output)
        print "\t(modification successful; loss = %d)"%loss
        if costUpperBound != None and False:
            print output
            print makeSketch(self.bank)
            assert False

        return Solution(prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ],
                        suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ],
                        underlyingForms = [ Morph.parse(self.bank, output, s) for s in stems ],
                        rules = [ (Rule.parse(self.bank, output, r) if rp == None else rp)
                                  for r,rp in zip(rules,originalRules) ],
                        adjustedCost = loss)

    def sketchIncrementalChange(self, solution, radius = 1):
        bestSolution = None

        # can we solve the problem without changing any existing rules but just by reordering them?
        for perm in everyPermutation(len(solution.rules), radius + 1):
            print "permutation =",perm
            permutedRules = [ solution.rules[j] for j in perm ]
            newSolution = self.sketchJointSolution(fixedRules = permutedRules)
            if newSolution == None: continue
            if bestSolution == None or newSolution.cost() < bestSolution.cost():
                bestSolution = newSolution

        # did we solve it just by reordering the rules?
        if bestSolution != None: return bestSolution
        
        # construct change vectors. These are None for new rule and a rule object otherwise.
        nr = len(solution.rules)
        # do not add a new rule
        ruleVectors = [ [ (r if k else None) for k,r in zip(v,solution.rules) ]
                        for v in everyBinaryVector(nr,nr - radius) ] 
        # add a new rule to the end
        ruleVectors += [ [ (r if k else None) for k,r in zip(v,solution.rules) ] + [None]
                         for v in everyBinaryVector(nr,nr - radius + 1) ] 
        # add a new rule to the beginning
        ruleVectors += [ [None] + [ (r if k else None) for k,r in zip(v,solution.rules) ]
                         for v in everyBinaryVector(nr,nr - radius + 1) ]
        
        for v in ruleVectors:
            try:
                s = self.sketchChangeToSolution(solution, v,
                                                costUpperBound = None if bestSolution == None else bestSolution.adjustedCost)
                if bestSolution == None or s.cost() < bestSolution.cost():
                    bestSolution = s
            except SynthesisFailure: pass
        if bestSolution:
            return bestSolution
        raise SynthesisFailure('incremental change')
        
        
    def incrementallySolve(self, stubborn = False, beam = 1):
        # start out with just the first example
        print "Starting out with explaining just the first two examples:"
        trainingData = self.data[:2]
        slave = UnderlyingProblem(trainingData, 1, self.bank)
        solution = slave.sketchJointSolution(canAddNewRules = True)

        radius = 1

        while True:
            haveCounterexample = False
            newExample = None
            for ce in self.findCounterexamples(solution, trainingData):
                haveCounterexample = True
                slave = UnderlyingProblem(trainingData + [ce], 0, self.bank)
                try:
                    solution = slave.sketchIncrementalChange(solution, radius)
                    print solution
                    newExample = ce
                    break
                except SynthesisFailure:
                    print "But, cannot incrementally change rules right now to accommodate that example."
                    # stubborn: insist on explaining earlier examples before explaining later examples
                    # so we want to break out of the loop over counterexamples
                    if stubborn: break
            if not haveCounterexample:
                print "No more counterexamples; done."
                return self.solveUnderlyingForms(solution)

            if newExample == None:
                print "I can't make any local changes to my rules to accommodate a counterexample."
                radius += 1
                print "Increasing search radius to %d"%radius
                if radius > 2:
                    print "I refuse to use a radius this big."
                    return None                    
            else:
                trainingData += [ce]
                print "Added the counterexample to the training data."
                print "Training data:"
                for t in trainingData:
                    print u"\t".join(t)
                print
                if radius > 1:
                    print "(radius set back to 1)"
                    radius = 1
                    print

                # Should we enumerate alternative hypotheses?
                if beam > 1:
                    worker = UnderlyingProblem(trainingData, 0, self.bank)
                    solutionScores = [(s.modelCost() + self.solutionDescriptionLength(s), s)
                                      for s in worker.solveTopRules(solution, beam) ]
                    print "Alternative solutions and their scores:"
                    for c,s in solutionScores:
                        print "COST = %d, SOLUTION = \n%s\n"%(c,str(s))

                    solution = min(solutionScores)[1]
                    
                    
                    
    def solutionDescriptionLength(self,solution,inflections = None):
        if inflections == None:
            return sum([self.solutionDescriptionLength(solution,i)
                        for i in self.data ])

        Model.Global()
        stem = Morph.sample()

        # Make the morphology/phonology be a global definition
        prefixes = [ define("Word", p.makeConstant(self.bank)) for p in solution.prefixes ]
        suffixes = [ define("Word", s.makeConstant(self.bank)) for s in solution.suffixes ]
        rules = [ define("Rule", r.makeConstant(self.bank)) for r in solution.rules ]

        for r in rules: condition(fixStructuralChange(r))

        predictions = [ applyRules(rules, concatenate3(prefixes[j],stem,suffixes[j]))
                        for j in range(self.numberOfInflections) ]

        cost = wordLength(stem)
        for j in range(self.numberOfInflections):
            m = Morph(inflections[j])
            cost = cost + Conditional(wordEqual(predictions[j], m.makeConstant(self.bank)),
                                      Constant(0),
                                      Constant(len(m)))

        minimize(cost)

        output = solveSketch(self.bank, self.maximumObservationLength, self.maximumObservationLength)
        if output == None:
            print "Fatal error: "
            print "Could not compute description length of:"
            print u"\t".join(inflections)
            print "For model:"
            print solution
            print "Bank:"
            print self.bank
            print "Solver output:"
            printSketchFailure()
            assert False
        return parseMinimalCostValue(output)
