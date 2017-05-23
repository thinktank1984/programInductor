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

from problems import underlyingProblems,interactingProblems
from countingProblems import CountingProblem

from multiprocessing import Pool
import random
import sys
import pickle
import argparse
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
            _r = define("Rule", r.makeConstant(self.bank))
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
        def applyRules(d):
            for r in rules: d = applyRule(r,d)
            return d
        def buildUnderlyingForm(prefix, suffix):
            if isinstance(stem, Morph): # underlying form is fixed
                return (prefix + stem + suffix).makeConstant(self.bank)
            else: # underlying form is unknown
                return concatenate3(prefix, stem, suffix)
            
        prediction = [ applyRules(buildUnderlyingForm(prefixes[i],suffixes[i]))
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
            raise SynthesisFailure("Failed at underlying form analysis.")

        us = [ Morph.parse(self.bank, output, s) for s in stems ]

        # The only purpose of this nested loop is to verify that there are no bugs
        for j in range(len(self.inflectionMatrix)):
            for i in range(self.numberOfInflections):
                u = prefixes[i] + us[j] + suffixes[i]
                for r in rules:
                    #print "Applying",r,"to",u,"gives",r.apply(u),"aka",Morph.fromMatrix(r.apply(u))
                    u = self.applyRule(r,u)
                # print Morph.fromMatrix(u),"\n",Morph(tokenize(self.data[j][i]))
                if Morph(tokenize(self.data[j][i])) != u:
                    print "underlying:",prefixes[i] + us[j] + suffixes[i]
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
                 for rs in f(inputs, existingRules) ]
        

    def solveTopRules(self, solution, k):
        '''Takes as input a "seed" solution, and expands it to k solutions with the same morphological cost'''
        solutions = [solution] if existingRules == None else [(prefixes, suffixes, existingRules)]
        
        for _ in range(k - 1):
            Model.Global()

            rules = [ Rule.sample() for _ in range(self.depth) ]
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
            if observation in trainingData:
                continue
            if not self.verify(solution, observation):
                print "COUNTEREXAMPLE:\t",
                for i in observation: print i,"\t",
                print ""
                yield observation

    def findCounterexample(self, solution, trainingData = []):
        # Returns the first counterexample or None if there are no counterexamples
        return next(self.findCounterexamples(solution, trainingData), None)

    def verify(self, solution, inflections):
        Model.Global()

        stem = Morph.sample()

        # Make the morphology/phonology be a global definition
        prefixes = [ define("Word", p.makeConstant(self.bank)) for p in solution.prefixes ]
        suffixes = [ define("Word", s.makeConstant(self.bank)) for s in solution.suffixes ]
        rules = [ define("Rule", r.makeConstant(self.bank)) for r in solution.rules ]

        for r in rules: condition(fixStructuralChange(r))

        self.conditionOnStem(rules, stem, prefixes, suffixes, inflections)

        output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
        return (output != None)

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
            condition(totalCost < costUpperBound)
        minimize(totalCost)

        return totalCost

    def sketchJointSolution(self, canAddNewRules = False, costUpperBound = None):
        try:
            Model.Global()
            rules = [ Rule.sample() for _ in range(self.depth) ]
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
                                rules = [ Rule.parse(self.bank, output, r) for r in rules ])
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

    def heldOutSolution(self, k = 1, threshold = float('inf'), initialTrainingSize = 2, testing = 0.0, inductiveBiases = []):
        if testing == 0.0:
            return self.counterexampleSolution(k, threshold, initialTrainingSize),None,None

        if self.numberOfInflections > 7: initialTrainingSize = 3

        trainingData,testingData = randomTestSplit(self.data, testing)
        slave = UnderlyingProblem(trainingData, self.depth, bank = self.bank)
        solutions = slave.counterexampleSolution(k,threshold,initialTrainingSize)

        accuracies, compressions = {}, {}
        for bias in inductiveBiases:
            print "Considering bias",bias
            ug = str2ug(bias)
            prefixes, suffixes, rules = max(solutions, key = lambda z: ug.logLikelihood(z[2]))
            accuracy,compression = self.accuracyAndCompression(prefixes, suffixes, rules, testingData)
            print "Average held out accuracy: ",accuracy
            print "Average held out compression:",compression
            print "As a test, trying to calculate on the original training data also:"
            print self.accuracyAndCompression(prefixes, suffixes, rules, trainingData)
            accuracies[bias] = accuracy
            compressions[bias] = compression
        return solutions, accuracies, compressions

    def accuracyAndCompression(self, prefixes, suffixes, rules, testingData):
        accuracy,compression = 0,0
        for inflections in testingData:
            a,c = self.inflectionAccuracy(prefixes, suffixes, rules, inflections)
            compression += c
            accuracy += a
        return accuracy/len(testingData), compression/float(len(testingData))

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
                expandedSolution = slave.sketchJointSolution(costUpperBound = solution.cost())
                if expandedSolution.cost() <= solution.cost():
                    solution = expandedSolution
                    print "Better compression achieved by expanding to %d rules"%(self.depth + 1)
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


    def inflectionAccuracy(self, prefixes, suffixes, rules, inflections):
        """inflections : list of APA strings. Returns (accuracy, descriptionLength)"""
        correctPredictions = 0
        encodingLength = sum([ len(Morph(i)) for i in inflections ])
        
        for testingIndex in range(len(inflections) + 1):
            # if testingIndex  = len(inflections), don't hold anything out
            # we do this to check description length
            trainingIndexes = [ j for j in range(len(inflections)) if j != testingIndex ]

            if len(inflections) == 1 and testingIndex != len(inflections):
                continue
                        
            Model.Global()
            
            stem = Morph.sample()
            minimize(wordLength(stem))

            # Make the morphology/phonology be a global definition
            prefixes_ = [ define("Word", p.makeConstant(self.bank)) for p in prefixes ]
            suffixes_ = [ define("Word", s.makeConstant(self.bank)) for s in suffixes ]
            rules_ = [ define("Rule", r.makeConstant(self.bank)) for r in rules ]
            for r in rules_: condition(fixStructuralChange(r))

            prediction = Morph.sample()
            surfaces = [ (s if j in trainingIndexes else prediction) for j,s in enumerate(inflections) ]
            
            self.conditionOnStem(rules_, stem, prefixes_, suffixes_, surfaces)

            # IMPORTANT!
            # Because we are modeling the prediction as a morph,
            # the maximum morph length most also be the maximum observation length
            output = solveSketch(self.bank, self.maximumObservationLength, self.maximumObservationLength)
            if not output or testingIndex == len(inflections):
                prediction = None
            else:
                prediction = Morph.parse(self.bank, output, prediction)
            if testingIndex < len(inflections): # testing ability to make new inflection
                if prediction == Morph(tokenize(inflections[testingIndex])):
                    correctPredictions += 1
                else:
                    print "I saw these inflections:","\t".join([s for j,s in enumerate(inflections)
                                                                if j != testingIndex])
                    print "I predicted ", prediction,"instead of", Morph(tokenize(inflections[testingIndex]))
            else: # checking compression
                if output:
                    encodingLength = len(Morph.parse(self.bank, output, stem))
        return correctPredictions/float(len(inflections)), encodingLength

    def sketchChangeToSolution(self,solution,keepRule, costUpperBound = None):
        Model.Global()

        rules = [ (rule.defineConstant(self.bank) if keep else Rule.sample())
                  for keep, rule in zip(keepRule,solution.rules + [None]) ]
        stems = [ Morph.sample() for _ in self.inflectionMatrix ]
        prefixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]
        suffixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]

        self.minimizeJointCost(rules, stems, prefixes, suffixes, costUpperBound)
        self.conditionOnData(rules, stems, prefixes, suffixes)

        output = self.solveSketch()
        if output == None: raise SynthesisFailure("No satisfying modification possible.")

        return Solution(prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ],
                        suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ],
                        underlyingForms = [ Morph.parse(self.bank, output, s) for s in stems ],
                        rules = [ Rule.parse(self.bank, output, r) for r in rules ])

    def sketchIncrementalChange(self, solution, radius = 1):
        bestSolution = None
        bestCost = None
        # construct change vectors
        nr = len(solution.rules)
        keepVectors = list(everyBinaryVector(nr,nr - radius))
        keepVectors += [ v + [False] for v in everyBinaryVector(nr,nr - radius + 1) ]

        for v in keepVectors: print v

        for v in keepVectors:
            try:
                s = self.sketchChangeToSolution(solution, v, bestCost)
                if bestCost == None or s.cost() < bestCost:
                    bestSolution = s
                    bestCost = s.cost()
            except:
                pass
        if bestSolution:
            return bestSolution
        raise SynthesisFailure('incremental change')
        
        
    def incrementallySolve(self):
        # start out with just the first example
        print "Starting out with explaining just the first two examples:"
        trainingData = self.data[:2]
        slave = UnderlyingProblem(trainingData, 1, self.bank)
        solution = slave.sketchJointSolution()

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
            if not haveCounterexample:
                print "No more counterexamples; done."
                break
            else:
                if newExample == None:
                    print "I can't make any local changes to my rules to accommodate a counterexample."
                    radius += 1
                    print "Increasing search radius to %d"%radius
                    if radius > 2:
                        print "I refuse to use a radius this big."
                        break
                    
                else:
                    trainingData += [ce]
                    print "Added the counterexample to the training data."
                    print "Training data:"
                    for t in trainingData:
                        print "\t".join(t)
                    print
                    if radius > 1:
                        print "(radius set back to 1)"
                        radius = 1
                        print 
                    
                    
        
        
        

            


def handleProblem(parameters):
    problemIndex = parameters['problemIndex']
    random.seed(parameters['seed'] + problemIndex)
        
    p = underlyingProblems[problemIndex - 1] if problemIndex < 50 else interactingProblems[problemIndex - 1 - 50]
    print p.description
    if problemIndex != 7:
        print latexMatrix(p.data)
    else:
        print CountingProblem(p.data, p.parameters).latex()

    startTime = time()

    ss = None # solutions to save out to the pickled file
    accuracy, compression = None, None
    if problemIndex == 7:
        ss = CountingProblem(p.data, p.parameters).topSolutions(parameters['top'])
    else:
        up = UnderlyingProblem(p.data, 1)
        up.incrementallySolve()
        #up.counterexampleSolution()
        
        assert False
        ss, accuracy, compression = up.heldOutSolution(parameters['top'],
                                                       parameters['threshold'],
                                                       testing = parameters['testing'],
                                                       inductiveBiases = parameters['universalGrammar'])
        ss = [rs for _,_,rs in ss ] # just save the rules

    print "Total time taken by problem %d: %f seconds"%(problemIndex, time() - startTime)
    
    if ss != None and parameters['top'] > 1 and parameters['testing'] == 0.0:
        pickle.dump(ss, open("pickles/matrix_"+str(problemIndex)+".p","wb"))
    if accuracy != None and compression != None:
        parameters['accuracy'] = accuracy
        parameters['compression'] = compression
        print parameters
        name = "%d_%s_%f_%d_%d"%(parameters['problemIndex'],
                                 "_".join(sorted(parameters['universalGrammar'])),
                                 parameters['testing'],
                                 parameters['top'],
                                 parameters['seed'])
        pickle.dump(parameters, open("testingAccuracy/%s.p"%name,"wb"))
        
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Solve jointly for morphology and phonology given surface inflected forms of lexemes')
    parser.add_argument('problem')
    parser.add_argument('-t','--top', default = 1, type = int)
    parser.add_argument('-f','--threshold', default = float('inf'), type = int)
    parser.add_argument('-m','--cores', default = 1, type = int)
    parser.add_argument('-s','--seed', default = '0', type = str)
    parser.add_argument('-H','--hold', default = '0.0', type = str)
    parser.add_argument('-u','--universal', default = 'flat',type = str)

    arguments = parser.parse_args()
    
    if arguments.problem == 'integration':
        problems = [1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    # Chapter five problems
                    51,
                    52,
                    53]
    else:
        problems = map(int,arguments.problem.split(','))

    parameters = [{'problemIndex': problemIndex,
                   'seed': seed,
                   'testing': testing,
                   'universalGrammar': arguments.universal.split(','),
                   'top': arguments.top,
                   'threshold': arguments.threshold}
                  for problemIndex in problems
                  for seed in map(int,arguments.seed.split(','))
                  for testing in map(float,arguments.hold.split(',')) ]
    print parameters
    
    if arguments.cores > 1:
        Pool(arguments.cores).map(handleProblem, parameters)
    else:
        map(handleProblem, parameters)
        
