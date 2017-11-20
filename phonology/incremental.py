# -*- coding: utf-8 -*-

from matrix import *
import utilities

from time import time
import random
from sketchSyntax import auxiliaryCondition
import traceback

def everyEditSequence(sequence, radii, allowSubsumption = True, maximumLength = None):
    '''Handy utility which is at the core of incremental solving.
    The idea is that we want to enumerate every way that the original sequence could be edited.
    Edits include adding new elements to the sequence, substituting existing elements with None, and exchanging existing elements of the sequence.
    
    radii: This is a list of how many edits we are allowed to make. example: [1,2] means we can make either one or two edits.
    allowSubsumption: Are we allowed to have two edits which subsume each other? 
    returns: a list of sequences, which might have None in them. None means a new unknown sequence element.'''
    
    def _everySequenceEdit(r):
        # radius larger than sequence
        if r > len(sequence): return [[None]*r]
        # radius zero
        if r < 1: return [list(range(len(sequence)))]

        edits = []
        for s in _everySequenceEdit(r - 1):
            # Should we consider adding a new thing to the sequence?
            if len(s) == len(sequence) and (maximumLength == None or len(sequence) < maximumLength):
                edits += [ s[:j] + [None] + s[j:] for j in range(len(s) + 1) ]
            # Consider doing over any one element of the sequence
            edits += [ s[:j] + [None] + s[j+1:] for j in range(len(s)) ]
            # Consider swapping elements
            edits += [ [ (s[i] if k == j else (s[j] if k == i else s[k])) for k in range(len(s)) ]
                       for j in range(len(s) - 1)
                       for i in range(j,len(s)) ]
        return edits

    # remove duplicates
    candidates = set([ tuple(s)
                       for radius in radii
                       for s in _everySequenceEdit(radius) ] )
    # remove things that came from an earlier radius
    for smallerRadius in range(min(radii)):
        candidates -= set([ tuple(s) for s in _everySequenceEdit(smallerRadius) ])
    # some of the edit sequences might subsume other ones, eg [None,1,None] subsumes [0,1,None]
    # we want to not include things that are subsumed by other things

    def subsumes(moreGeneral, moreSpecific):
        # Does there exist a substitution of None's that converts general to specific?
        # Importantly, we are allowed to substitute None for the empty sequence
        if len(moreGeneral) == 0: return len(moreSpecific) == 0
        if len(moreSpecific) == 0: return all(x == None for x in moreSpecific)
        g = moreGeneral[0]
        s = moreSpecific[0]
        return (g == None and subsumes(moreGeneral[1:],moreSpecific)) or \
            ((s == g or g == None) and subsumes(moreGeneral[1:],moreSpecific[1:]))
        if not len(moreGeneral) == len(moreSpecific): return False
        for g,s in zip(moreGeneral,moreSpecific):
            if g != None and s != g: return False
        #print "%s is strictly more general than %s"%(moreGeneral,moreSpecific)
        return True

    # disabling subsumption removal
    if not allowSubsumption:
        removedSubsumption = [ s
                               for s in candidates 
                               if not any([ subsumes(t,s) for t in candidates if t != s ]) ]
    else: removedSubsumption = candidates

    # Order them by expected difficulty
    removedSubsumption = sorted(removedSubsumption,
                                key = lambda x: (len([y for y in x if y == None]), # How many new things are there
                                                 len(x),
                                                 x))
        
    # reindex into the input sequence
    return [ [ (None if j == None else sequence[j]) for j in s ]
             for s in removedSubsumption ]

class IncrementalSolver(UnderlyingProblem):
    def __init__(self, data, window, bank = None, UG = None, numberOfCPUs = None, maximumNumberOfRules = 5):
        UnderlyingProblem.__init__(self, data, bank = bank, UG = UG)
        self.numberOfCPUs = numberOfCPUs if numberOfCPUs != None else utilities.numberOfCPUs()/2

        self.maximumNumberOfRules = maximumNumberOfRules
        self.windowSize = window

        self.fixedMorphologyThreshold = 10
        self.fixedUnderlyingFormThreshold = 10
        self.fixedMorphology = None

    def sketchChangeToSolution(self, solution, rules, allTheData = None):
        assert allTheData != None
        
        Model.Global()

        originalRules = list(rules) # save it for later

        isNewRule = [ r == None for r in rules ]
        rules = [ (rule.makeDefinition(self.bank) if rule != None else Rule.sample())
                  for rule in rules ]
        prefixes = []
        suffixes = []

        # Calculate the random variables for the morphology
        # Some of these will be constant if we have seen them enough
        
        morphologicalCosts = []
        for j in range(self.numberOfInflections):
            # Do we have at least fixedMorphologyThreshold examples for this particular inflection?
            inflectionExamples = len([ None for l in allTheData if l[j] != None ])
            
            if inflectionExamples >= self.fixedMorphologyThreshold or self.fixedMorphology != None:
                if self.fixedMorphology:
                    assert self.fixedMorphology.prefixes[j] == solution.prefixes[j]
                    assert self.fixedMorphology.suffixes[j] == solution.suffixes[j]
                print "Fixing morphology of inflection %d to %s + %s"%(j,solution.prefixes[j],solution.suffixes[j])
                morphologicalCosts.append(len(solution.prefixes[j]) + \
                                          len(solution.suffixes[j]))
                prefixes.append(solution.prefixes[j].makeDefinition(self.bank))
                suffixes.append(solution.suffixes[j].makeDefinition(self.bank))
            else:
                morphologicalCosts.append(None)
                prefixes.append(Morph.sample())
                suffixes.append(Morph.sample())
            if inflectionExamples == 0:
                # Never seen this inflection: give it the empty morphology
                print "Clamping the morphology of inflection %d to be empty"%j
                condition(wordLength(prefixes[j]) == 0)
                condition(wordLength(suffixes[j]) == 0)

        # Construct the random variables for the stems
        # After fixedUnderlyingFormThreshold examples passed an observation, we stop trying to modify the underlying form.
        # Set fixedUnderlyingFormThreshold = infinity to disable this heuristic.
        observationsWithFixedUnderlyingForms = [ o for j,o in enumerate(allTheData)
                                                 if j < len(allTheData) - self.fixedUnderlyingFormThreshold \
                                                 and j < len(solution.underlyingForms) ]
        for j,observation in enumerate(observationsWithFixedUnderlyingForms):
            # we need to also take into account the length of these auxiliary things because they aren't necessarily in self.data
            if max([ len(o) for o in observation if o != None ]) > self.maximumObservationLength: continue
            
            print "\t\t(clamping UR for observation %s to %s)"%(observation,solution.underlyingForms[j])
            stem = solution.underlyingForms[j].makeConstant(self.bank)
            for i,o in enumerate(observation):
                if o == None: continue
                phonologicalInput = concatenate3(prefixes[i],stem,suffixes[i])
                auxiliaryCondition(wordEqual(o.makeConstant(self.bank),
                                             applyRules(rules, phonologicalInput,
                                                        wordLength(prefixes[i]) + wordLength(stem),
                                                        len(o) + 1)))
                # SUBTLE: We can't actually assume that it does nothing if it is a "new" rule
                # This is because sometimes we go back and revise old rules
                # ,doNothing = isNewRule)))
        stems = [ Morph.sample() for observation in self.data
                  if not (observation in observationsWithFixedUnderlyingForms) ]
        dataToConditionOn = [ d for d in self.data
                              if not (d in observationsWithFixedUnderlyingForms)]
            
        # Only add in the cost of the new rules that we are synthesizing
        self.minimizeJointCost([ r for r,o in zip(rules,originalRules) if o == None],
                               stems, prefixes, suffixes,
                               morphologicalCosts = morphologicalCosts)
        self.conditionOnData(rules, stems, prefixes, suffixes,
                             observations = dataToConditionOn,
                             auxiliaryHarness = True)

        try:
            output = self.solveSketch()
        except SynthesisFailure:
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

        newSolution = Solution(prefixes = [ Morph.parse(self.bank, output, p) \
                                            if morphologicalCosts[j] == None \
                                            else solution.prefixes[j] \
                                            for j,p in enumerate(prefixes) ],
                               suffixes = [ Morph.parse(self.bank, output, s) \
                                            if morphologicalCosts[j] == None \
                                            else solution.suffixes[j] \
                                            for j,s in enumerate(suffixes) ],
                               rules = [ Rule.parse(self.bank, output, r) if rp == None else rp
                                         for r,rp in zip(rules,originalRules) ],
                               adjustedCost = loss)
        print "\t(modification successful; loss = %s, solution = \n%s\t)"%(loss,
                                                                           indent("\n".join(map(str,newSolution.rules))))

        flushEverything()
        return newSolution.withoutUselessRules()

    def sketchCEGISChange(self,solution, rules):
        n = len(self.data)/5
        if n < 4: n = 4
        if n > 10: n = 10
        if n > len(self.data) - self.windowSize: n = len(self.data) - self.windowSize
        trainingData = random.sample(self.data[:-self.windowSize], n) + self.data[-self.windowSize:]

        newSolution = None
        try: # catch timeout exception
            
            while True:
                worker = self.restrict(trainingData)
                newSolution = worker.sketchChangeToSolution(solution, rules, allTheData = self.data)
                if newSolution == None: return None
                print "CEGIS: About to find a counterexample to:\n",newSolution
                ce = self.findCounterexample(newSolution, trainingData)
                if ce == None:
                    print "No counterexample so I am just returning best solution"
                    newSolution.clearTransducers()
                    newSolution.underlyingForms = None
                    newSolution = self.solveUnderlyingForms(newSolution)
                    print "Final CEGIS solution:\n%s"%(newSolution)
                    return newSolution
                trainingData = trainingData + [ce]
                
        except SynthesisTimeout: return None

    def sketchIncrementalChange(self, solution, radius = 1):
        # This is the actual sequence of radii that we go through
        # We start out with a radius of at least 2 so that we can add a rule and revise an old rule
        def radiiSequence(sequenceIndex):
            assert sequenceIndex > 0
            if sequenceIndex == 1: return [1,2]
            else: return [sequenceIndex + 1]
        ruleVectors = everyEditSequence(solution.rules, radiiSequence(radius),
                                        allowSubsumption = False,
                                        maximumLength = self.maximumNumberOfRules)

        print "# parallel sketch jobs:",len(ruleVectors)

        # Ensure output is nicely ordered
        flushEverything()

        allSolutions = parallelMap(self.numberOfCPUs,
                                   lambda v: self.sketchCEGISChange(solution,v),
                                   ruleVectors)
        allSolutions = [ s for s in allSolutions if s != None ]
        if allSolutions == []:
            if exhaustedGlobalTimeout(): raise SynthesisTimeout()
            else: raise SynthesisFailure('incremental change')
        return sorted(allSolutions,key = lambda s: s.cost())
    

    def incrementallySolve(self, saveProgressTo = None,loadProgressFrom = None,fixedMorphology = None):
        if loadProgressFrom == None:        
            initialTrainingSize = self.windowSize
            print "Starting out with explaining just the first %d examples:"%initialTrainingSize
            trainingData = self.data[:initialTrainingSize]
            worker = self.restrict(trainingData)#, self.bank)
            solution = worker.sketchJointSolution(1,canAddNewRules = True,
                                                  auxiliaryHarness = True,
                                                  fixedMorphology = fixedMorphology)
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
            try:
                if self.verify(solution, self.data[j]):
                    j += 1
                    continue
            except SynthesisTimeout: return [solution]

            trainingData = self.data[:j]

            print "Next data points to explain: "
            window = self.data[j:j + self.windowSize]
            print u"\n".join([ u'\t~\t'.join(map(unicode,w)) for w in window ]) 

            radius = 1
            while True:
                # Prevent the accumulation of a large number of temporary files
                # These can easily grow into the gigabytes and I have disk quotas
                # deleteTemporarySketchFiles()
                try:
                    worker = self.restrict(trainingData + window)
                    solutions = worker.sketchIncrementalChange(solution, radius)
                    assert solutions != []
                    # see which of the solutions is best overall
                    # different metrics of "best overall",
                    # depending upon which set of examples you compute the description length
                    
                    solutionScores = [self.computeSolutionScores(s, trainingData + window)
                                      for s in solutions ]
                    print "Alternative solutions and their scores:"
                    for scoreDictionary in solutionScores:
                        print "COST = %.2f + (%d everything, %d invariant) = (%.2f, %.2f). SOLUTION = \n%s\n"%(
                            scoreDictionary['modelCost'],
                            scoreDictionary['everythingCost'],
                            scoreDictionary['invariantCost'],
                            scoreDictionary['modelCost'] + scoreDictionary['everythingCost'],
                            scoreDictionary['modelCost'] + scoreDictionary['invariantCost'],
                            scoreDictionary['solution'])
                    eager = False
                    if eager: costRanking = ['everythingCost','invariantCost']
                    else:     costRanking = ['invariantCost','everythingCost']
                    print "Picking the model with the best cost as ordered by:",' > '.join(costRanking)
                    solutionScores = [ tuple([ scores[k] + scores['modelCost'] for k in costRanking ] + [scores['solution']])
                                      for scores in solutionScores ]
                    solutionScores = min(solutionScores)
                    newSolution = solutionScores[-1]
                    newJointScore = solutionScores[0]
                    
                    print " [+] Best new solution (cost = %.2f):"%(newJointScore)
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
                    if radius > 1:
                        print "I refuse to use a radius this big."
                        self.windowSize -= 1
                        if self.windowSize > 0:
                            print "Decreased window size to"%self.windowSize
                            break # break out of the loop over different radius sizes
                        
                        print "Can't shrink the window anymore so I'm just going to return"
                        return [solution]
                    continue # retreat back to the loop over different radii
                except SynthesisTimeout: return [solution]

                # Successfully explained a new data item

                # Update both the training data and solution
                solution = newSolution
                j += self.windowSize
                
                break # break out the loop over different radius sizes

            if saveProgressTo != None:
                print " [+] Saving progress to %s"%saveProgressTo
                dumpPickle((j,None,solution.clearTransducers()),saveProgressTo)            

        return solution
