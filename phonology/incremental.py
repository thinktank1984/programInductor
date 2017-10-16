# -*- coding: utf-8 -*-

from matrix import *

from pathos.multiprocessing import ProcessingPool as Pool
from time import time
import random
from sketch import setGlobalTimeout
import traceback


class IncrementalSolver(UnderlyingProblem):
    def __init__(self, data, window, bank = None):
        UnderlyingProblem.__init__(self, data, bank = bank)
        self.windowSize = window

    def sketchChangeToSolution(self, solution, rules, allTheData = None):
        assert allTheData != None
        
        Model.Global()

        originalRules = list(rules) # save it for later

        rules = [ (rule.makeDefinition(self.bank) if rule != None else Rule.sample())
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

    def sketchCEGISChange(self,solution, rules):
        n = len(self.data)/5
        if n < 4: n = 4
        if n > 10: n = 10
        if n > len(self.data) - self.windowSize: n = len(self.data) - self.windowSize
        trainingData = random.sample(self.data[:-self.windowSize], n) + self.data[-self.windowSize:]

        newSolution = None
        while True:
            worker = IncrementalSolver(trainingData, self.windowSize, self.bank)
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

    def sketchIncrementalChange(self, solution, radius = 1):
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
    

    def incrementallySolve(self, saveProgressTo = None,loadProgressFrom = None):

        if loadProgressFrom == None:        
            initialTrainingSize = self.windowSize
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
            window = self.data[j:j + self.windowSize]
            print u"\n".join([ u'\t~\t'.join(map(unicode,w)) for w in window ]) 

            radius = 1
            while True:
                # Prevent the accumulation of a large number of temporary files
                # These can easily grow into the gigabytes and I have disk quotas
                deleteTemporarySketchFiles()
                try:
                    worker = IncrementalSolver(trainingData + window, self.windowSize, self.bank)
                    solutions = worker.sketchIncrementalChange(solution, radius)
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
                    for alreadyExplained in self.data[:j+self.windowSize]:
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
                j += self.windowSize
                
                break # break out the loop over different radius sizes

            if saveProgressTo != None:
                print " [+] Saving progress to %s"%saveProgressTo
                dumpPickle((j,None,solution.clearTransducers()),saveProgressTo)            

        return solution
