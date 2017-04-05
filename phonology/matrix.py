# -*- coding: utf-8 -*-

from utilities import *

from features import FeatureBank, tokenize
from rule import Rule,Guard,FeatureMatrix
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

    def applyRule(self, r, u):
        if USEPYTHONRULES:# use the Python implementation of rules
            return Morph.fromMatrix(r.apply(u))
        else:
            Model.Global()
            result = Morph.sample()
            condition(wordEqual(result,applyRule(r.makeConstant(self.bank),u.makeConstant(self.bank))))
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
    
    def solveUnderlyingForms(self, prefixes, suffixes, rules):
        Model.Global()
        rules_ = [ define("Rule", r.makeConstant(self.bank)) for r in rules ]
        prefixes_ = [ define("Word", p.makeConstant(self.bank)) for p in prefixes ]
        suffixes_ = [ define("Word", s.makeConstant(self.bank)) for s in suffixes ]
        stems = [ Morph.sample() for _ in self.inflectionMatrix ]
        self.conditionOnData(rules_, stems, prefixes_, suffixes_)

        for stem in stems:
            minimize(wordLength(stem))
        
        output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
        if not output:
            raise SynthesisFailure("Failed at underlying form analysis.")

        us = [ Morph.parse(self.bank, output, s) for s in stems ]

        for j in range(len(self.inflectionMatrix)):
            for i in range(self.numberOfInflections):
                u = prefixes[i] + us[j] + suffixes[i]
                for r in rules:
                    #print "Applying",r,"to",u,"gives",r.apply(u),"aka",Morph.fromMatrix(r.apply(u))
                    u = self.applyRule(r,u) #r.apply(u)
                # print Morph.fromMatrix(u),"\n",Morph(tokenize(self.data[j][i]))
                if Morph(tokenize(self.data[j][i])) != u:
                    print "underlying:",prefixes[i] + us[j] + suffixes[i]
                    print Morph(tokenize(self.data[j][i])), "versus", u
                    print Morph(tokenize(self.data[j][i])).phonemes, "versus", u.phonemes
                    assert False
                # print "\n"
            # print "\n\n"

        return us

    def fastTopRules(self, prefixes, suffixes, underlyingForms, k, existingRules):
        #k = int(math.ceil((float(k)**(1.0/self.depth))))

        inputs = [ prefixes[i] + underlyingForms[j] + suffixes[i]
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

        return f(inputs, existingRules) #, key = lambda rs: - self.inductiveBias.logLikelihood(rs))
        

    def solveTopRules(self, prefixes, suffixes, underlyingForms, k, existingRules = None):
        solutions = [] if existingRules == None else [(prefixes, suffixes, existingRules)]
        
        for _ in range(k - (1 if existingRules else 0)):
            Model.Global()

            rules = [ Rule.sample() for _ in range(self.depth) ]
            for other in solutions:
                condition(And([ ruleEqual(r, o.makeConstant(self.bank))
                                for r, o in zip(rules, other[2]) ]) == 0)

            minimize(sum([ ruleCost(r) for r in rules ]))

            # Keep morphology variable! Just ensure it has the same cost
            prefixes_ = [ sampleMorphWithLength(len(p)) for p in prefixes ]
            suffixes_ = [ sampleMorphWithLength(len(p)) for p in suffixes ]
            stems_ = [ sampleMorphWithLength(len(p)) for p in underlyingForms ]
            self.conditionOnData(rules, stems_, prefixes_, suffixes_)
            
            output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
            if not output:
                print "Found %d rules."%len(solutions)
                break
            solutions.append(([ Morph.parse(self.bank, output, m) for m in suffixes_ ],
                              [ Morph.parse(self.bank, output, m) for m in prefixes_ ],
                              [ Rule.parse(self.bank, output, r) for r in rules ]))
        return solutions

    def findCounterexample(self, prefixes, suffixes, rules, trainingData = []):
        print "Beginning verification"
        for observation in self.data:
            if observation in trainingData:
                continue
            if not self.verify(prefixes, suffixes, rules, observation):
                print "COUNTEREXAMPLE:\t",
                for i in observation: print i,"\t",
                print ""
                return observation

        return None


    def verify(self, prefixes, suffixes, rules, inflections):
        Model.Global()

        stem = Morph.sample()

        # Make the morphology/phonology be a global definition
        prefixes = [ define("Word", p.makeConstant(self.bank)) for p in prefixes ]
        suffixes = [ define("Word", s.makeConstant(self.bank)) for s in suffixes ]
        rules = [ define("Rule", r.makeConstant(self.bank)) for r in rules ]

        for r in rules: condition(fixStructuralChange(r))

        self.conditionOnStem(rules, stem, prefixes, suffixes, inflections)

        output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength)
        return (output != None)

    @staticmethod
    def showMorphologicalAnalysis(prefixes, suffixes):
        print "Morphological analysis:"
        for i in range(len(prefixes)):
            print "Inflection %d:\t"%i,
            print prefixes[i],
            print "+ stem +",
            print suffixes[i]

    @staticmethod
    def showRules(rules):
        print "Phonological rules:"
        for r in rules: print r

    def sketchJointSolution(self, canAddNewRules = False):
        try:
            Model.Global()
            rules = [ Rule.sample() for _ in range(self.depth) ]
            stems = [ Morph.sample() for _ in self.inflectionMatrix ]
            # useful for debugging fifty one
            # for stem in stems: condition(wordLength(stem) == 3)
            prefixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]
            suffixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]
       
            affixSize = sum([ wordLength(prefixes[j]) + wordLength(suffixes[j]) -
                              (len(self.inflectionMatrix[0][j]) - min(map(len, self.inflectionMatrix[0])) if self.numberOfInflections > 5 else 0)
                              for j in range(self.numberOfInflections) ])
            # print "affixSize = ",affixSize
            # condition(wordLength(prefixes[0]) == 2)
            # condition(wordLength(suffixes[0]) == 2)

            # We subtract a constant from the stems size in order to offset the cost
            # Should have no effect upon the final solution that we find,
            # but it lets sketch get away with having to deal with smaller numbers
            stemSize = sum([ wordLength(m)-
                             (min(map(len,self.inflectionMatrix[j])) if self.numberOfInflections > 1
                              else len(self.inflectionMatrix[j][0]) - 3)
                             for j,m in enumerate(stems) ])
            
            ruleSize = sum([ruleCost(r) for r in rules ])
            minimize(ruleSize + stemSize + affixSize)
            if False: # testing for fifty one
                #[  ] ---> [ -highTone ] / [ +highTone ] [  ] _ 
                #[  ] ---> [ +highTone ] / [ +highTone ] [  ] _ [  ]
                r1 = Rule(FeatureMatrix([]), FeatureMatrix([(False,"highTone")]),
                          Guard('L',False,False,[FeatureMatrix([(True,"highTone")]),
                                                 FeatureMatrix([])]),
                          Guard('R',False,False,[]),
                          0)
                r2 = Rule(FeatureMatrix([]), FeatureMatrix([(True,"highTone")]),
                          Guard('L',False,False,[FeatureMatrix([(True,"highTone")]),
                                                 FeatureMatrix([])]),
                          Guard('R',False,False,[FeatureMatrix([])]),
                          0)
                print "I would like to see these rules:"
                print r1
                print r2
                if False: # using rule equal
                    condition(ruleEqual(rules[0],
                                        r1.makeConstant(self.bank)))
                    if len(rules) > 1:
                        condition(ruleEqual(rules[1],
                                        r2.makeConstant(self.bank)))
                else:
                    rules[0] = define("Rule",r1.makeConstant(self.bank))
                    if len(rules) > 1:
                        rules[1] = define("Rule",r2.makeConstant(self.bank))

            self.conditionOnData(rules, stems, prefixes, suffixes)

            output = solveSketch(self.bank, self.maximumObservationLength, self.maximumMorphLength, showSource = False)
            if not output:
                raise SynthesisFailure("Failed at morphological analysis.")
            prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ]
            suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ]
            stems = [ Morph.parse(self.bank, output, s) for s in stems ]
            rules = [ Rule.parse(self.bank, output, r) for r in rules ]
            UnderlyingProblem.showMorphologicalAnalysis(prefixes, suffixes)
            UnderlyingProblem.showRules(rules)
            return (prefixes, suffixes, stems, rules)
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
            prefixes, suffixes, stems, rules = UnderlyingProblem(trainingData, self.depth, self.bank).sketchJointSolution(canAddNewRules = True)
            self.depth = len(rules) # update depth because it might have grown
            solverTime = time() - solverTime

            counterexample = self.findCounterexample(prefixes, suffixes, rules, trainingData)
            if counterexample == None: # we found a solution that had no counterexamples
                print "Final set of counterexamples:"
                print latexMatrix(trainingData)
                print "Final solutions:"
                UnderlyingProblem.showMorphologicalAnalysis(prefixes, suffixes)
                underlyingForms = self.solveUnderlyingForms(prefixes, suffixes, rules)

                # Do we have enough time in our budget to not be fast?
                if solverTime*k < threshold:
                    solutions = self.solveTopRules(prefixes, suffixes, underlyingForms, k, rules)
                else:
                    print "Using the optimized top rules."
                    solutions = [ (prefixes, suffixes, rs)
                                  for rs in self.fastTopRules(prefixes, suffixes, underlyingForms, k, rules) ]
                    
                return solutions
            else:
                trainingData.append(counterexample)

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
                    #9, having problems
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
        
