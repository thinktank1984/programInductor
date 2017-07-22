# -*- coding: utf-8 -*-

from compileRuleToSketch import compileRuleToSketch
from utilities import *
from solution import *
from features import FeatureBank, tokenize
from rule import * # Rule,Guard,FeatureMatrix,EMPTYRULE
from morph import Morph
from sketchSyntax import Expression,makeSketchSkeleton
from sketch import *
from supervised import solveTopSupervisedRules
from latex import latexMatrix
from UG import str2ug #FlatUG, ChomskyUG, FeatureUG, SkeletonUG, SkeletonFeatureUG

from pathos.multiprocessing import ProcessingPool as Pool
import random
import sys
import pickle
import math
from time import time

class SynthesisFailure(Exception):
    pass

def sampleMorphWithLength(l):
    m = Morph.sample()
    condition(wordLength(m) == l)
    return m
        

class UnderlyingProblem():
    def __init__(self, data, depth, bank = None):
        self.depth = depth
        self.bank = bank if bank != None else FeatureBank([ w for l in data for w in l if w != None ])

        self.numberOfInflections = len(data[0])
        for d in data: assert len(d) == self.numberOfInflections
        
        # wrap the data in Morph objects if it isn't already
        self.data = [ [ None if i == None else (i if isinstance(i,Morph) else Morph(tokenize(i)))
                        for i in Lex] for Lex in data ]

        self.maximumObservationLength = max([ len(w) for l in self.data for w in l if w != None ])
        self.maximumMorphLength = max(10,self.maximumObservationLength - 2)

    def solveSketch(self, minimizeBound = 31):
        return solveSketch(self.bank, self.maximumObservationLength + 1, self.maximumMorphLength, showSource = False, minimizeBound = minimizeBound)

    def applyRuleUsingSketch(self,r,u):
        '''u: morph; r: rule'''
        Model.Global()
        result = Morph.sample()
        _r = r.makeDefinition(self.bank)
        condition(wordEqual(result,applyRule(_r,u.makeConstant(self.bank), self.maximumObservationLength + 1)))
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
            condition(wordLength(applyRule(r.makeConstant(self.bank),u.makeConstant(self.bank),self.maximumObservationLength)) > 0)
            if solveSketch(self.bank, self.maximumObservationLength, self.maximumObservationLength) == None:
                print "WARNING: weaker test also fails"
            else:
                print "WARNING: weaker test succeeds"
            return Morph.fromMatrix(r.apply(u))

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
        
        surfaceLengths = [ 0 if s == None else len(s)
                           for s in surfaces ]
        prediction = [ applyRules(rules, buildUnderlyingForm(prefixes[i],suffixes[i]), surfaceLengths[i] + 1)
                     for i in range(len(surfaces)) ]
        for i in range(len(surfaces)):
            surface = surfaces[i]
            if surface == None: continue
            if not isinstance(surface,Expression):
                surface = surface.makeConstant(self.bank)
            condition(wordEqual(surface, prediction[i]))
    
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
        if maximumNumberOfSolutions != None:
            # enforce k^d < maximumNumberOfSolutions
            # k < maximumNumberOfSolutions**(1/d)
            k = int(min(k,maximumNumberOfSolutions**(1.0/k)))
        
        inputs = [ solution.prefixes[i] + solution.underlyingForms[j] + solution.suffixes[i]
                   for j in range(len(self.data))
                   for i in range(self.numberOfInflections) ]

        def f(xs, rs):
            if rs == []: return [[]]
            ys = [ self.applyRule(rs[0],x) #Morph.fromMatrix(rs[0].apply(x))
                   for x in xs ]            
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

            # Keep morphology variable! Just ensure it has the same cost
            prefixes = [ sampleMorphWithLength(len(p)) for p in solution.prefixes ]
            suffixes = [ sampleMorphWithLength(len(p)) for p in solution.suffixes ]
            stems = [ Morph.sample() for p in solution.underlyingForms ]
            
            self.conditionOnData(rules, stems, prefixes, suffixes)
            self.minimizeJointCost(rules, stems, prefixes, suffixes)
            
            output = self.solveSketch()
            if not output:
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

    def sketchJointSolution(self, canAddNewRules = False, costUpperBound = None, fixedRules = None, fixedMorphology = None):
        try:
            Model.Global()
            if fixedRules == None:
                rules = [ Rule.sample() for _ in range(self.depth) ]
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

            self.minimizeJointCost(rules, stems, prefixes, suffixes, costUpperBound)

            self.conditionOnData(rules, stems, prefixes, suffixes)

            output = self.solveSketch()
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


    def counterexampleSolution(self, k = 1, threshold = float('inf'), initialTrainingSize = 2, fixedMorphology = None):
        # Start out with the shortest examples
        #self.sortDataByLength()
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
            solution = UnderlyingProblem(trainingData, self.depth, self.bank).sketchJointSolution(canAddNewRules = True, fixedMorphology = fixedMorphology)
            self.depth = solution.depth() # update depth because it might have grown
            solverTime = time() - solverTime

            counterexample = self.findCounterexample(solution, trainingData)
            if counterexample != None:
                trainingData.append(counterexample)
                continue
            
            # we found a solution that had no counterexamples
            #print "Final set of counterexamples:"
            #print latexMatrix(trainingData)

            # When we expect it to be tractable, we should try doing a little bit deeper
            if self.depth < 3 and self.numberOfInflections < 3:
                slave = UnderlyingProblem(trainingData, self.depth + 1, self.bank)
                expandedSolution = slave.sketchJointSolution(fixedMorphology = fixedMorphology)
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


    def sketchChangeToSolution(self,solution, rules, k = 1):
        # if we are not changing any of the rules just enumerate one solution
        if not any([ r == None for r in rules ]): k = 1
        
        Model.Global()

        originalRules = list(rules) # save it for later
        solutionsSoFar = [] # how many of the K requested solutions have we found

        if False:
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
        fixedMorphologyThreshold = 6
        morphologicalCosts = []
        for j in range(self.numberOfInflections):
            # Do we have at least two examples for this particular inflection?
            inflectionExamples = len([ None for l in self.data if l[j] != None ])
            if inflectionExamples >= fixedMorphologyThreshold:
                print "Fixing morphology of inflection %d to %s + %s"%(j,solution.prefixes[j],solution.suffixes[j])
                condition(wordEqual(prefixes[j], solution.prefixes[j].makeConstant(self.bank)))
                condition(wordEqual(suffixes[j], solution.suffixes[j].makeConstant(self.bank)))
                morphologicalCosts.append(len(solution.prefixes[j]) + len(solution.suffixes[j]))
            else: morphologicalCosts.append(None)
            if inflectionExamples == 0:
                # Never seen this inflection: give it the empty morphology
                print "Clamping the morphology of inflection %d to be empty"%j
                condition(wordLength(prefixes[j]) == 0)
                condition(wordLength(suffixes[j]) == 0)

        # this piece of code will also hold the underlying forms fixed
        if False and len(solution.underlyingForms) > 3:
            for stemVariable,oldValue in zip(stems,solution.underlyingForms):
                condition(wordEqual(stemVariable, oldValue.makeConstant(self.bank)))

        # Only add in the cost of the new rules that we are synthesizing
        self.minimizeJointCost([ r for r,o in zip(rules,originalRules) if o == None],
                               stems, prefixes, suffixes,
                               morphologicalCosts = morphologicalCosts)
        self.conditionOnData(rules, stems, prefixes, suffixes)

        for _ in range(k):
            # Condition on it being a different solution
            for other in solutionsSoFar:
                condition(And([ ruleEqual(r,o.makeConstant(self.bank))
                                for r,o,v in zip(rules, other.rules, originalRules)
                                if v == None ]) == 0)
            output = self.solveSketch()#minimizeBound = 50)
            if output == None:
                print "\t(no modification possible: got %d solutions)"%(len(solutionsSoFar))
                # Because these are executed in parallel, do not throw an exception
                break
            loss = parseMinimalCostValue(output)
            if loss == None:
                print "WARNING: None loss"
                print output
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
                                                                             

            solutionsSoFar.append(newSolution)
            solutionsSoFar[-1].verifyRuleCompilation(self.bank,self.data)
        flushEverything()
        return [ s.withoutUselessRules() for s in solutionsSoFar ]

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
        if len(solution.rules) > 4: desiredNumberOfCPUs = 20
        else: desiredNumberOfCPUs = 35
        allSolutions = Pool(min(desiredNumberOfCPUs,numberOfCPUs())).map(lambda v: self.sketchChangeToSolution(solution,v,k), ruleVectors)
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
            worker = UnderlyingProblem(trainingData, 1, self.bank)
            solution = worker.sketchJointSolution(canAddNewRules = True)
            j = initialTrainingSize
        else:
            (j,trainingData,solution) = loadPickle(loadProgressFrom)
            print " [+] Loaded progress from %s"%loadProgressFrom
            print "Solution =\n%s"%solution
        
        # Maintain the invariant: the first j examples have been explained
        while j < len(self.data):
            # Can we explain the jth example?
            if self.verify(solution, self.data[j]):
                j += 1
                continue

            print "Next data points to explain: "
            window = self.data[j:j + windowSize]
            print u"\n".join([ u'\t~\t'.join(map(unicode,w)) for w in window ]) 

            radius = 1
            while True:
                try:
                    worker = UnderlyingProblem(trainingData + window, 0, self.bank)
                    solutions = worker.sketchIncrementalChange(solution, radius, k = beam)
                    assert solutions != []
                    # see which of the solutions is best overall
                    # different metrics of "best overall",
                    # depending upon which set of examples you compute the description length
                    
                    solutionScores = [self.computeSolutionScores(s,self.data[:j + windowSize],trainingData + window)
                                      for s in solutions ]
                    print "Alternative solutions and their scores:"
                    for scoreDictionary in solutionScores:
                        print "COST = %d + (%d everything, %d invariant, %d training) = (%d, %d, %d). SOLUTION = \n%s\n"%(scoreDictionary['modelCost'],
                                                                                                                          scoreDictionary['everythingCost'],
                                                                                                                          scoreDictionary['invariantCost'],
                                                                                                                          scoreDictionary['trainingCost'],
                                                                                                                          scoreDictionary['modelCost'] + scoreDictionary['everythingCost'],
                                                                                                                          scoreDictionary['modelCost'] + scoreDictionary['invariantCost'],
                                                                                                                          scoreDictionary['modelCost'] + scoreDictionary['trainingCost'],
                                                                                                                          scoreDictionary['solution'])
                    if eager: costRanking = ['everythingCost','invariantCost','trainingCost']
                    else:     costRanking = ['invariantCost','everythingCost','trainingCost']
                    print "Picking the model with the best cost as ordered by:",' > '.join(costRanking)
                    solutionScores = [ tuple([ scores[k] + scores['modelCost'] for k in costRanking ] + [scores['solution']])
                                      for scores in solutionScores ]
                    solutionScores = min(solutionScores)
                    newSolution = solutionScores[-1]
                    newJointScore = solutionScores[0]
                    
                    print " [+] Best new solution (cost = %d):"%(newJointScore)
                    print newSolution

                    # Make sure that all of the previously explained data points are still explained
                    # These "regressions" triggered the regressed test case being added to the training data
                    haveRegression = False
                    for alreadyExplained in self.data[:j+windowSize]:
                        if not self.verify(newSolution, alreadyExplained):
                            print "But that solution cannot explain an earlier data point, namely:"
                            print u'\t~\t'.join(map(unicode,alreadyExplained))
                            if alreadyExplained in trainingData or alreadyExplained in window:
                                self.illustrateFatalIncrementalError(newSolution,
                                                                     alreadyExplained,
                                                                     newSolution.underlyingForms[(trainingData+window).index(alreadyExplained)])
                                assert False
                            else:
                                # Incorporate at most one regression into the training data
                                if not haveRegression:
                                    trainingData.append(alreadyExplained)
                            haveRegression = True
                    if haveRegression:
                        continue
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
                trainingData += window
                solution = newSolution
                j += windowSize
                
                break # break out the loop over different radius sizes

            if saveProgressTo != None:
                print " [+] Saving progress to %s"%saveProgressTo
                dumpPickle((j,trainingData,solution.clearTransducers()),saveProgressTo)
            

        return solution

    def computeSolutionScores(self,solution,invariant,training):
        # Compute the description length of everything
        descriptionLengths = [ self.inflectionsDescriptionLength(solution, x) for x in self.data ]
        everythingCost = sum(descriptionLengths)
        invariantCost = sum([ descriptionLengths[j] for j,x in enumerate(self.data) if x in invariant ])
        trainingCost = sum([ descriptionLengths[j] for j,x in enumerate(self.data) if x in training ])
        return {'solution': solution,
                'modelCost': solution.modelCost(),
                'everythingCost': everythingCost,
                'invariantCost': invariantCost,
                'trainingCost': trainingCost}

    def illustrateFatalIncrementalError(self,newSolution,alreadyExplained,ur):
        print " [-] FATAL: Already in training data!"
        # Try transducing the underlying form using each inflection individually
        for i in range(self.numberOfInflections):
            print "Transducing just the %d inflection:"%(i+1)
            justThisInflection = [None]*i + [alreadyExplained[i]] + [None]*(self.numberOfInflections - i - 1)
            print newSolution.transduceUnderlyingForm(self.bank, justThisInflection)

        # Illustrate the derivation
        print "UR =",ur
        for i in range(self.numberOfInflections):
            print "Inflection",i
            surface = newSolution.prefixes[i] + ur + newSolution.suffixes[i]
            print "Using sketch rules:"
            for r in newSolution.rules:
                newSurface = self.applyRuleUsingSketch(r,surface)
                print "%s > %s"%(surface,newSurface)
                surface = newSurface
            print "Using transducer rules:"
            surface = newSolution.prefixes[i] + ur + newSolution.suffixes[i]
            for r in newSolution.rules:
                newSurface = self.applyRule(r,surface) if surface != None else None
                print "%s > %s"%(surface,newSurface)
                surface = newSurface

        # saved to disk for further dissection
        temporaryName = makeTemporaryFile('.p')
        newSolution.clearTransducers()
        dumpPickle((newSolution,alreadyExplained), temporaryName)
        print " [-] Saved (solution, inflections) to %s"%temporaryName
    
    def solutionDescriptionLength(self,solution,data = None):
        if data == None: data = self.data
        if getVerbosity() > 3:
            print "Calculating description length of:"
            print solution
        return sum([self.inflectionsDescriptionLength(solution,i)
                    for i in data ])
        
    def inflectionsDescriptionLength(self, solution, inflections):
        ur = solution.transduceUnderlyingForm(self.bank, inflections)
        if getVerbosity() > 3:
            print "Transducing UR of:",u"\t".join(map(unicode,inflections))
            print "\tUR = ",ur
        if ur != None:
            return len(ur)
        else:
            # Dumb noise model
            if False:
                return sum([ len(s) for s in inflections if s != None ])
            else:
                # Smart noise model
                # todo: we could also incorporate the morphology here if we wanted to
                subsequenceLength = multiLCS([ s.phonemes for s in inflections if s != None ])
                return sum([ len(s) - subsequenceLength for s in inflections if s != None ])
    

    def paretoFront(self, k, temperature, useMorphology = False):
        assert self.numberOfInflections == 1
        self.maximumObservationLength += 1

        def affix():
            if useMorphology: return Morph.sample()
            else: return Morph([]).makeConstant(self.bank)
        def parseAffix(output, morph):
            if useMorphology: return Morph.parse(self.bank, output, morph)
            else: return Morph([])
            
        Model.Global()
        rules = [ Rule.sample() for _ in range(self.depth) ]

        stems = [ Morph.sample() for _ in self.data ]
        prefixes = [ affix() for _ in range(self.numberOfInflections) ]
        suffixes = [ affix() for _ in range(self.numberOfInflections) ]

        for i in range(len(stems)):
            self.conditionOnStem(rules, stems[i], prefixes, suffixes, self.data[i])

        stemCostExpression = sum([ wordLength(u) for u in stems ] + [ wordLength(u) for u in suffixes ] + [ wordLength(u) for u in prefixes ])
        stemCostVariable = unknownInteger()
        condition(stemCostVariable == stemCostExpression)
        minimize(stemCostExpression)
        ruleCostExpression = sum([ ruleCost(r) for r in rules ])
        ruleCostVariable = unknownInteger()
        condition(ruleCostVariable == ruleCostExpression)
        if len(rules) > 0:
            minimize(ruleCostExpression)

        solutions = []
        solutionCosts = []
        for _ in range(k):
            # Excludes solutions we have already found
            for rc,uc in solutionCosts:
                condition(And([ruleCostVariable == rc,stemCostVariable == uc]) == 0)

            output = self.solveSketch(minimizeBound = 64)

            if output == None: break

            s = Solution(suffixes = [ parseAffix(output, m) for m in suffixes ],
                         prefixes = [ parseAffix(output, m) for m in prefixes ],
                         rules = [ Rule.parse(self.bank, output, r) for r in rules ],
                         underlyingForms = [ Morph.parse(self.bank, output, m) for m in stems ])
            solutions.append(s)
            print s

            rc = sum([r.cost() for r in s.rules ])
            uc = sum([len(u) for u in s.prefixes+s.suffixes+s.underlyingForms ])
            print "Costs:",(rc,uc)
            actualCosts = (parseInteger(output, ruleCostVariable), parseInteger(output, stemCostVariable))
            assert actualCosts == (rc,uc)
            (rc,uc) = actualCosts
            solutionCosts.append((rc,uc))

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


    def randomSampleSolver(self, N = 30, lower = 5, upper = 8):
        '''N: number of random samples.
        lower: lower bound on the size of the random samples.
        upper: upper bound on the size of the random samples.'''
        
        # Figure out the morphology from the first few examples
        preliminarySolution = UnderlyingProblem(self.data[:4], 1, self.bank).sketchJointSolution(canAddNewRules = True)
        print "Sticking with that morphology from here on out..."
        
        # construct random subsets
        subsets = []
        print "%d random subsets of the training data:"%N
        for _ in range(N):
            size = choice(range(lower, upper + 1))
            startingPoint = choice(range(len(self.data) - size))        
            endingPoint = startingPoint + size
            subsets.append(self.data[startingPoint:endingPoint])
            print "SUBSET:"
            print u"\n".join([ u'~'.join(map(unicode,x)) for x in subsets[-1] ])

        nc = min(N,numberOfCPUs())
        solutions = Pool(nc).map(lambda subset: UnderlyingProblem(subset, 1, self.bank).counterexampleSolution(k = 10, fixedMorphology = preliminarySolution),subsets)
        for ss,subset in zip(solutions, subsets):
            print " [+] Random sample solver: Training data:"
            print u"\n".join([u"\t".join(map(unicode,xs)) for xs in subset ])
            print " Solutions:"
            for s in ss: print s
            print 
            
        # Coalesce all of the rules found by the random samples
        coalescedRules = []
        for ss in solutions:
            for s in ss:
                for r in s.rules:
                    if not any([ unicode(r) == unicode(rp) for rp in coalescedRules ]):
                        coalescedRules.append(r)

        print "Here are all of the %d unique rules that were were discovered from the random samples:"%len(coalescedRules)
        for r in coalescedRules:
            print r
