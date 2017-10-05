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

def sampleMorphWithLength(l):
    m = Morph.sample()
    condition(wordLength(m) == l)
    return m
        

class UnderlyingProblem():
    def __init__(self, data, bank = None, useSyllables = False):
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
            for r in s.rules:
                y = self.applyRuleUsingSketch(r,x)
                print "Rewrites to %s using rule\t%s"%(y,r)
                x = y
            print 

    def applyRuleUsingSketch(self,r,u):
        '''u: morph; r: rule'''
        Model.Global()
        result = Morph.sample()
        _r = r.makeDefinition(self.bank)
        condition(wordEqual(result,applyRule(_r,u.makeConstant(self.bank), self.maximumObservationLength + 1)))
        try:
            output = self.solveSketch(self.bank)
        except SynthesisFailure:
            print "applyRuleUsingSketch: UNSATISFIABLE for %s %s"%(u,r)
            assert False
        except SynthesisTimeout:
            print "applyRuleUsingSketch: TIMEOUT for %s %s"%(u,r)
            assert False
        return Morph.parse(self.bank, output, result)
        
    
    def applyRule(self, r, u):
        ruleOutput = runForward(r.fst(self.bank),u.fst(self.bank))
        if ruleOutput == None: return None
        return Morph.fromFST(self.bank, ruleOutput)
                             

    def sortDataByLength(self):
        # Sort the data by length. Break ties by remembering which one originally came first.
        dataTaggedWithLength = [ (sum([ len(w) if w != None else 0 for w in self.data[j]]),
                                  j,
                                  self.data[j])
                                 for j in range(len(self.data)) ]
        self.data = [ d[2] for d in sorted(dataTaggedWithLength) ]


    def conditionOnStem(self, rules, stem, prefixes, suffixes, surfaces):
        """surfaces : list of numberOfInflections elements, each of which is a morph object"""
        assert self.numberOfInflections == len(surfaces)
        
        def buildUnderlyingForm(prefix, suffix):
            if isinstance(stem, Morph): # underlying form is fixed
                assert False # deprecated
                return (prefix + stem + suffix).makeConstant(self.bank)
            else: # underlying form is unknown
                return concatenate3(prefix, stem, suffix)
        
        for i,surface in enumerate(surfaces):
            if surface == None: continue
            
            prediction = applyRules(rules, buildUnderlyingForm(prefixes[i],suffixes[i]), len(surface) + 1)

            condition(wordEqual(surface.makeConstant(self.bank), prediction))
    
    def conditionOnData(self, rules, stems, prefixes, suffixes):
        '''Conditions on inflection matrix.'''
        for i in range(len(stems)):
            self.conditionOnStem(rules, stems[i], prefixes, suffixes, self.data[i])
    
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
        
        inputs = [ solution.prefixes[i] + solution.underlyingForms[j] + solution.suffixes[i]
                   for j in range(len(self.data))
                   for i in range(self.numberOfInflections) ]

        def f(xs, rs):
            if rs == []: return [[]]
            ys = [ self.applyRuleUsingSketch(rs[0],x)
                   for x in xs ]            
            alternatives = SupervisedProblem(zip(xs,ys)).fastTopK(k, rs[0])
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
        # EXPERIMENTAL: insert some UG!!!
        universal = [FeatureMatrix([(True,'vowel')]),
                     FeatureMatrix([(False,'vowel')]),
                     FeatureMatrix([(True,'voice')]),
                     FeatureMatrix([(False,'voice')]),
                     FeatureMatrix([(False,'sonorant')]),
                     Rule(FeatureMatrix([(False,'sonorant')]),
                          FeatureMatrix([(False,'voice')]),
                          Guard('L',False,False,[]),
                          Guard('R',True,False,[]),
                          0)]
        #sketchUniversalGrammar(universal,self.bank)

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
            if self.numberOfInflections > 5: # heuristic: adjust when there are at least five inflections
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

    def sketchJointSolution(self, depth, canAddNewRules = False, costUpperBound = None, fixedRules = None, fixedMorphology = None):
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

            self.conditionOnData(rules, stems, prefixes, suffixes)

            output = self.solveSketch()

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
                return self.sketchJointSolution(depth, canAddNewRules = canAddNewRules)
            else:
                return None
        # pass this exception onto the caller
        #except SynthesisTimeout:


    def counterexampleSolution(self, k = 1, threshold = float('inf'), initialTrainingSize = 2, fixedMorphology = None):
        # Start out with the shortest examples
        #self.sortDataByLength()
        if self.numberOfInflections == 1 or initialTrainingSize == 0:
            initialTrainingSize = len(self.data)
        trainingData = self.data[:initialTrainingSize]

        depth = 1

        solution = None

        while True:
            print "CEGIS: Training data:"
            for r in trainingData:
                for i in r: print i,
                print ""

            solverTime = time() # time to sketch the solution
            # expand the rule set until we can fit the training data
            try:
                solution = UnderlyingProblem(trainingData, self.bank).sketchJointSolution(depth, canAddNewRules = True, fixedMorphology = fixedMorphology)            
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
            if depth < 3 and self.numberOfInflections < 3:
                slave = UnderlyingProblem(trainingData, self.bank)
                try:
                    expandedSolution = slave.sketchJointSolution(depth + 1, fixedMorphology = fixedMorphology)
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

    def sketchCEGISChange(self,solution, rules):
        n = len(self.data)/5
        if n < 4: n = 4
        if n > 10: n = 10
        if n > len(self.data) - 2: n = len(self.data) - 2
        trainingData = random.sample(self.data[:-2], n) + self.data[-2:]

        newSolution = None
        while True:
            worker = UnderlyingProblem(trainingData, self.bank)
            newSolution = worker.sketchChangeToSolution(solution, rules, allTheData = self.data)
            if newSolution == None: return []
            print "CEGIS: About to find a counterexample to:\n",newSolution
            ce = self.findCounterexample(newSolution, trainingData)
            if ce == None:
                print "No counterexample so I am just returning best solution"
                newSolution.clearTransducers()
                newSolution.underlyingForms = None
                newSolution = self.solveUnderlyingForms(newSolution)
                print "Final CEGIS solution:\n%s"%(newSolution)
                return [newSolution]
            trainingData = trainingData + [ce]
        assert False
            
    def sketchChangeToSolution(self, solution, rules, allTheData = None):
        assert allTheData != None
        
        Model.Global()

        originalRules = list(rules) # save it for later

        if True:
            # Use the general-purpose sketch forward model implementation
            rules = [ (rule.makeDefinition(self.bank) if rule != None else Rule.sample())
                      for rule in rules ]
        else:
            # Compile each rule into its own special function
            rules = [ (compileRuleToSketch(self.bank,rule) if rule != None else Rule.sample())
                      for rule in rules ]
        stems = [ Morph.sample() for _ in self.data ]
        prefixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]
        suffixes = [ Morph.sample() for _ in range(self.numberOfInflections) ]

        # Should we hold the morphology fixed?
        fixedMorphologyThreshold = 10
        morphologicalCosts = []
        for j in range(self.numberOfInflections):
            # Do we have at least two examples for this particular inflection?
            # todo: this calculation is wrong - it should be based on the solution we are modifying
            inflectionExamples = len([ None for l in allTheData if l[j] != None ])
            
            if inflectionExamples >= fixedMorphologyThreshold:
                print "Fixing morphology of inflection %d to %s + %s"%(j,solution.prefixes[j],solution.suffixes[j])
                condition(wordEqual(prefixes[j], solution.prefixes[j].makeConstant(self.bank)))
                condition(wordEqual(suffixes[j], solution.suffixes[j].makeConstant(self.bank)))
                morphologicalCosts.append(len(solution.prefixes[j]) + \
                                          len(solution.suffixes[j]))
            else: morphologicalCosts.append(None)
            if inflectionExamples == 0:
                # Never seen this inflection: give it the empty morphology
                print "Clamping the morphology of inflection %d to be empty"%j
                condition(wordLength(prefixes[j]) == 0)
                condition(wordLength(suffixes[j]) == 0)

        # After underlyingFormLag examples passed an observation, we stop trying to modify the underlying form.
        # Set underlyingFormLag = infinity to disable this heuristic.
        underlyingFormLag = 10
        for stem, observation in zip(stems, self.data):
            observationIndex = allTheData.index(observation)
            if observationIndex < len(allTheData) - underlyingFormLag and observationIndex < len(solution.underlyingForms):
                oldStem = solution.underlyingForms[observationIndex]
                print "\t\t(clamping UR for observation %s to %s)"%(observation,oldStem)
                condition(wordEqual(stem, oldStem.makeConstant(self.bank)))

        # Only add in the cost of the new rules that we are synthesizing
        self.minimizeJointCost([ r for r,o in zip(rules,originalRules) if o == None],
                               stems, prefixes, suffixes,
                               morphologicalCosts = morphologicalCosts)
        self.conditionOnData(rules, stems, prefixes, suffixes)

        try:
            output = self.solveSketch()#minimizeBound = 50)
        except SynthesisFailure,SynthesisTimeout:
            print "\t(no modification possible)"
            # Because these are executed in parallel, do not throw an exception
            return None
        loss = parseMinimalCostValue(output)
        if loss == None:
            print "WARNING: None loss"
            print output
            printLastSketchOutput()
            print makeSketchSkeleton()
            assert False

        newSolution = Solution(prefixes = [ Morph.parse(self.bank, output, p) for p in prefixes ],
                               suffixes = [ Morph.parse(self.bank, output, s) for s in suffixes ],
                               underlyingForms = [ Morph.parse(self.bank, output, s) for s in stems ],
                               rules = [ (Rule.parse(self.bank, output, r) if rp == None else rp)
                                                 for r,rp in zip(rules,originalRules) ],
                               adjustedCost = loss)
        print "\t(modification successful; loss = %s, solution = \n%s\t)"%(loss,
                                                                           indent("\n".join(map(str,newSolution.rules))))

        #newSolution.verifyRuleCompilation(self.bank,self.data)
        flushEverything()
        return newSolution.withoutUselessRules()

    def sketchIncrementalChange(self, solution, radius = 1, k = 1):
        # This is the actual sequence of radii that we go through
        # We start out with a radius of at least 2 so that we can add a rule and revise an old rule
        def radiiSequence(sequenceIndex):
            assert sequenceIndex > 0
            if sequenceIndex == 1: return [1,2]
            else: return [sequenceIndex + 1]
        ruleVectors = everyEditSequence(solution.rules, radiiSequence(radius))

        # A cap the maximum number of rules that we are willing to consider
        ruleVectors = [ ruleVector for ruleVector in ruleVectors if len(ruleVector) <= 6 ]

        print "# parallel sketch jobs:",len(ruleVectors)

        # Ensure output is nicely ordered
        flushEverything()

        # parallel computation involves pushing the solution through a pickle
        # so make sure you do not pickle any transducers
        solution.clearTransducers()
        Rule.clearSavedTransducers()
        
        # Figure out how many CPUs we want to use.
        # if the solution we are modifying has lots of rules use fewer
        # This is because more rules means that each sketch invocation uses more memory;
        if len(solution.rules) > 3: desiredNumberOfCPUs = 20
        else: desiredNumberOfCPUs = 35
        allSolutions = Pool(min(desiredNumberOfCPUs,numberOfCPUs())).map(lambda v: self.sketchCEGISChange(solution,v), ruleVectors)
        allSolutions = [ s for ss in allSolutions for s in ss ]
        if allSolutions == []: raise SynthesisFailure('incremental change')
        return sorted(allSolutions,key = lambda s: s.cost())

    def incrementallySolve(self, beam = 1,windowSize = 2,eager = False,saveProgressTo = None,loadProgressFrom = None):
        print "Using beam width of",beam
        print "Using window size of ",windowSize

        if loadProgressFrom == None:        
            initialTrainingSize = windowSize
            print "Starting out with explaining just the first %d examples:"%initialTrainingSize
            trainingData = self.data[:initialTrainingSize]
            worker = UnderlyingProblem(trainingData, self.bank)
            solution = worker.sketchJointSolution(1,canAddNewRules = True)
            j = initialTrainingSize
        else:
            (j,trainingData,solution) = loadPickle(loadProgressFrom)
            print " [+] Loaded progress from %s"%loadProgressFrom
            print "Solution =\n%s"%solution
            if len(solution.prefixes) < len(self.data[0]):
                print " [?] WARNING: Missing morphology for some inflections, padding with empty"
                solution.prefixes += [Morph([])]*(len(self.data[0]) - len(solution.prefixes))
                solution.suffixes += [Morph([])]*(len(self.data[0]) - len(solution.suffixes))
                assert len(solution.prefixes) == len(solution.suffixes)
            elif len(solution.prefixes) > len(self.data[0]):
                print " [-] FATAL: Solution has more inflections and than data set???"
                assert False
                
        # Maintain the invariant: the first j examples have been explained
        while j < len(self.data):
            # Can we explain the jth example?
            if self.verify(solution, self.data[j]):
                j += 1
                continue

            trainingData = self.data[:j]

            print "Next data points to explain: "
            window = self.data[j:j + windowSize]
            print u"\n".join([ u'\t~\t'.join(map(unicode,w)) for w in window ]) 

            radius = 1
            while True:
                # Prevent the accumulation of a large number of temporary files
                # These can easily grow into the gigabytes and I have disk quotas
                deleteTemporarySketchFiles()
                try:
                    worker = UnderlyingProblem(trainingData + window, self.bank)
                    solutions = worker.sketchIncrementalChange(solution, radius, k = beam)
                    assert solutions != []
                    # see which of the solutions is best overall
                    # different metrics of "best overall",
                    # depending upon which set of examples you compute the description length
                    
                    solutionScores = [self.computeSolutionScores(s, trainingData + window)
                                      for s in solutions ]
                    print "Alternative solutions and their scores:"
                    for scoreDictionary in solutionScores:
                        print "COST = %d + (%d everything, %d invariant) = (%d, %d). SOLUTION = \n%s\n"%(
                            scoreDictionary['modelCost'],
                            scoreDictionary['everythingCost'],
                            scoreDictionary['invariantCost'],
                            scoreDictionary['modelCost'] + scoreDictionary['everythingCost'],
                            scoreDictionary['modelCost'] + scoreDictionary['invariantCost'],
                            scoreDictionary['solution'])
                    if eager: costRanking = ['everythingCost','invariantCost']
                    else:     costRanking = ['invariantCost','everythingCost']
                    print "Picking the model with the best cost as ordered by:",' > '.join(costRanking)
                    solutionScores = [ tuple([ scores[k] + scores['modelCost'] for k in costRanking ] + [scores['solution']])
                                      for scores in solutionScores ]
                    solutionScores = min(solutionScores)
                    newSolution = solutionScores[-1]
                    newJointScore = solutionScores[0]
                    
                    print " [+] Best new solution (cost = %d):"%(newJointScore)
                    print newSolution

                    # Make sure that all of the previously explained data points are still explained
                    for alreadyExplained in self.data[:j+windowSize]:
                        if not self.verify(newSolution, alreadyExplained):
                            print "But that solution cannot explain an earlier data point, namely:"
                            print u'\t~\t'.join(map(unicode,alreadyExplained))
                            print "This should be impossible with the new incremental CEGIS"
                            assert False
                except SynthesisFailure:
                    print "No incremental modification within radius of size %d"%radius
                    radius += 1
                    print "Increasing search radius to %d"%radius
                    if radius > 3:
                        print "I refuse to use a radius this big."
                        return None
                    continue # retreat back to the loop over different radii

                # Successfully explained a new data item

                # Update both the training data and solution
                solution = newSolution
                j += windowSize
                
                break # break out the loop over different radius sizes

            if saveProgressTo != None:
                print " [+] Saving progress to %s"%saveProgressTo
                dumpPickle((j,None,solution.clearTransducers()),saveProgressTo)            

        return solution

    def computeSolutionScores(self,solution,invariant):
        # Compute the description length of everything
        descriptionLengths = Pool(numberOfCPUs()).map(lambda x: self.inflectionsDescriptionLength(solution, x), self.data)
        everythingCost = sum(descriptionLengths)
        invariantCost = sum([ len(u) for u in solution.underlyingForms ]) 
        return {'solution': solution,
                'modelCost': solution.modelCost(),
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
            assert actualCosts == (rc,uc)
            (rc,uc) = actualCosts
            print "Actual costs:",actualCosts
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


if __name__ == '__main__':
    from parseSPE import parseSolution
    from problems import sevenProblems

    s = parseSolution(sevenProblems[1].solutions[0])
    solver = UnderlyingProblem(sevenProblems[1].data)

    solver.debugSolution(s,Morph(tokenize(u"koš^yil^y")))
    
