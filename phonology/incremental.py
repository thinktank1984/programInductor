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
                # Consider either appending or prepending
                edits += [ [None] + s, s + [None]]
                #edits += [ s[:j] + [None] + s[j:] for j in range(len(s) + 1) ]
            # Consider doing over any one element of the sequence
            edits += [ s[:j] + [None] + s[j+1:] for j in range(len(s)) ]
            # Consider swapping elements
            edits += [ [ (s[i] if k == j else (s[j] if k == i else s[k])) for k in range(len(s)) ]
                       for j in range(len(s) - 1)
                       for i in range(j,len(s))
                       if s[j] != None and s[i] != None ]
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
    def __init__(self, data, window, bank = None, UG = None, numberOfCPUs = None, maximumNumberOfRules = 6, fixedMorphology = None, maximumRadius = 2):
        UnderlyingProblem.__init__(self, data, bank = bank, UG = UG, fixedMorphology = fixedMorphology)
        self.numberOfCPUs = numberOfCPUs if numberOfCPUs != None else \
                            int(math.ceil(utilities.numberOfCPUs()*0.75))

        self.maximumNumberOfRules = maximumNumberOfRules
        self.maximumRadius = maximumRadius
        
        totalNumberOfWords = sum( x != None for i in self.data for x in i )
        wordsPerDataPoint = float(totalNumberOfWords)/len(self.data)
        
        if window == None:
            # Adaptively set the window size
            if wordsPerDataPoint <= 1.0: window = 6
            elif wordsPerDataPoint <= 3.0: window = 3
            elif wordsPerDataPoint <= 4.0: window = 2
            else: window = 1
            print "Incremental solver has adaptively set the window size to",window
        self.windowSize = window

        if wordsPerDataPoint >= 12:
            self.fixedMorphologyThreshold = 1
        else:
            self.fixedMorphologyThreshold = 10
        self.fixedUnderlyingFormThreshold = 10

        # Map from (surface1, ..., surface_I) to ur
        self.fixedUnderlyingForms = {}

        # After we have seen a rule be around for at least 2 times in
        # a row we keep it forever
        self.ruleFreezingThreshold = 2
        self.frozenRules = set([])
        # Map from rule to how many times in a row we have seen it lately
        self.ruleHistory = {}

    def solveUnderlyingForms(self, solution):
        '''Takes in a solution w/o underlying forms, and gives the one that has underlying forms.
        Unlike the implementation in Matrix, reuses fixed underlying forms to be more time efficient'''
        if len(solution.underlyingForms) != 0 and getVerbosity() > 0:
            print "WARNING: solveUnderlyingForms: Called with solution that already has underlying forms"

        return Solution(rules = solution.rules,
                        prefixes = solution.prefixes,
                        suffixes = solution.suffixes,
                        underlyingForms = \
                        dict(self.fixedUnderlyingForms,
                             **{ inflections: solution.transduceUnderlyingForm(self.bank, inflections)
                                 for inflections in self.data if not (inflections in self.fixedUnderlyingForms) }))

    def updateFixedMorphology(self, solution, trainingData):
        fixed = []
        for j in range(self.numberOfInflections):
            # Do we have at least fixedMorphologyThreshold examples for this particular inflection?
            inflectionExamples = sum( ss[j] != None for ss in solution.underlyingForms.keys() )
            if inflectionExamples >= self.fixedMorphologyThreshold:
                fixed.append((solution.prefixes[j], solution.suffixes[j]))
                print "Fixing morphology of inflection %d to %s + stem + %s"%(j,
                                                                              solution.prefixes[j],
                                                                              solution.suffixes[j])
                if self.fixedMorphology[j] != None:
                    assert self.fixedMorphology[j][0] == solution.prefixes[j]
                    assert self.fixedMorphology[j][1] == solution.suffixes[j]
            # Otherwise was it already fixed?
            elif self.fixedMorphology[j] != None: fixed.append(self.fixedMorphology[j])
            else: fixed.append(None)
        self.fixedMorphology = fixed

    def updateFixedUnderlyingForms(self, solution):
        numberToFix = max(0, len(solution.underlyingForms) - self.fixedUnderlyingFormThreshold)
        self.fixedUnderlyingForms = {x: solution.underlyingForms[x]
                                     for x in self.data[:numberToFix]}
        for x in self.data:
            if x in self.fixedUnderlyingForms:
                print "\t\t(clamping UR for observation %s to %s)"%(x,self.fixedUnderlyingForms[x])

    def updateFrozenRules(self, newSolution):
        for r in newSolution.rules:
            self.ruleHistory[r] = self.ruleHistory.get(r,0) + 1
        toRemove = [ r for r in self.ruleHistory.keys() if r not in newSolution.rules ]
        for r in toRemove:
            del self.ruleHistory[r]

        for r,f in self.ruleHistory.iteritems():
            if f >= self.ruleFreezingThreshold and r not in self.frozenRules:
                print "Permanently freezing the rule",r
                self.frozenRules.add(r)

    def guessUnderlyingForms(self, stems):
        dataToConditionOn = [ d for d in self.data
                              if not (d in self.fixedUnderlyingForms)]
        prefixes = [ None if inflection is None else inflection[0] for inflection in self.fixedMorphology ]
        suffixes = [ None if inflection is None else inflection[1] for inflection in self.fixedMorphology ]
        assert len(dataToConditionOn) == len(stems)
        for x,stem in zip(dataToConditionOn, stems):
            self.constrainUnderlyingRepresentation(stem, prefixes, suffixes, x)

    def sketchChangeToSolution(self, solution, rules):
        Model.Global()

        originalRules = list(rules) # save it for later

        rules = [ (rule.makeDefinition(self.bank) if rule != None else Rule.sample())
                  for rule in rules ]

        prefixes = []
        suffixes = []

        # Calculate the random variables for the morphology
        # Some of these will be constant if we have seen them enough
        
        morphologicalCosts = []
        for j in range(self.numberOfInflections):
            if self.fixedMorphology[j] != None:
                assert self.fixedMorphology[j][0] == solution.prefixes[j]
                assert self.fixedMorphology[j][1] == solution.suffixes[j]
                
                morphologicalCosts.append(len(solution.prefixes[j]) + \
                                          len(solution.suffixes[j]))
                prefixes.append(solution.prefixes[j].makeDefinition(self.bank))
                suffixes.append(solution.suffixes[j].makeDefinition(self.bank))
            else:
                morphologicalCosts.append(None)
                prefixes.append(Morph.sample())
                suffixes.append(Morph.sample())
            if all(l[j] == None for l in self.data + self.fixedUnderlyingForms.values()) \
               and self.fixedMorphology[j] == None:
                # Never seen this inflection: give it the empty morphology
                print "Clamping the morphology of inflection %d to be empty"%j
                condition(wordLength(prefixes[j]) == 0)
                condition(wordLength(suffixes[j]) == 0)

        # Construct the random variables for the stems
        # After fixedUnderlyingFormThreshold examples passed an observation, we stop trying to modify the underlying form.
        # Set fixedUnderlyingFormThreshold = infinity to disable this heuristic.
        for observation,stem in self.fixedUnderlyingForms.iteritems():
            # we need to also take into account the length of these auxiliary things because they aren't necessarily in self.data
            if max( len(o) for o in observation if o != None ) > self.maximumObservationLength: continue
            
            stem = stem.makeConstant(self.bank)
            for i,o in enumerate(observation):
                if o == None: continue
                phonologicalInput = concatenate3(prefixes[i],stem,suffixes[i])
                auxiliaryCondition(wordEqual(o.makeConstant(self.bank),
                                             applyRules(rules, phonologicalInput,
                                                        wordLength(prefixes[i]) + wordLength(stem),
                                                        len(o) + 1)))
        stems = [ Morph.sample() for observation in self.data
                  if not (observation in self.fixedUnderlyingForms) ]
        dataToConditionOn = [ d for d in self.data
                              if not (d in self.fixedUnderlyingForms)]
            
        # Only add in the cost of the new rules that we are synthesizing
        self.minimizeJointCost([ r for r,o in zip(rules,originalRules) if o == None],
                               stems, prefixes, suffixes,
                               morphologicalCosts = morphologicalCosts)
        self.conditionOnData(rules, stems, prefixes, suffixes,
                             observations = dataToConditionOn,
                             auxiliaryHarness = True)
        self.guessUnderlyingForms(stems)

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
                                         for r,rp in zip(rules,originalRules) ])
        print "\t(modification successful; loss = %s, solution = \n%s\t)"%(loss,
                                                                           indent("\n".join(map(str,newSolution.rules))))

        flushEverything()
        return newSolution.withoutUselessRules()

    def sketchCEGISChange(self, solution, rules):
        windowData = self.data[-self.windowSize:]
        fixedData = self.data[:len(self.fixedUnderlyingForms)]
        remainingData = self.data[len(self.fixedUnderlyingForms):-self.windowSize]
        n = min(10,len(remainingData))
        trainingData = random.sample(remainingData, n) + windowData

        newSolution = None
        try: # catch timeout exception
            
            while True:
                worker = self.restrict(trainingData)
                newSolution = worker.sketchChangeToSolution(solution, rules)
                if newSolution == None: return None
                print "CEGIS: About to find a counterexample to:\n",newSolution
                ce = self.findCounterexample(newSolution, trainingData)
                if ce == None:
                    print "No counterexample so I am just returning best solution"
                    newSolution.underlyingForms = {}
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
        ruleVectors = [ vector
                        for vector in ruleVectors
                        if all(f in vector for f in self.frozenRules )]
        
        if len(solution.rules) <= 2:
            # This is generally tractable when there is  < 3 rules
            ruleVectors.append(solution.rules + [None,None])

        print "# parallel sketch jobs:",len(ruleVectors)
        print "# data points not in window or fixed:",len(self.data) - self.windowSize - len(self.fixedUnderlyingForms)

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
    

    def incrementallySolve(self, saveProgressTo = None,loadProgressFrom = None,k = 1):
        if loadProgressFrom == None:        
            initialTrainingSize = self.windowSize
            print "Starting out with explaining just the first %d examples:"%initialTrainingSize
            trainingData = self.data[:initialTrainingSize]
            print u"\n".join(u"\t~\t".join(map(unicode,w)) for w in trainingData)
            worker = self.restrict(trainingData)
            solution = worker.sketchJointSolution(1,canAddNewRules = True,
                                                  auxiliaryHarness = True)
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
                if self.verify(solution,self.data[j]):
                    j += 1
                    continue
            except SynthesisTimeout: return solution.toFrontier()

            trainingData = self.data[:j]

            print "Next data points to explain: "
            window = self.data[j:j + self.windowSize]
            print u"\n".join([ u'\t~\t'.join(map(unicode,w)) for w in window ]) 

            # Fix the morphology/stems/rules that we are certain about
            self.updateFixedMorphology(solution, trainingData)
            self.updateFixedUnderlyingForms(solution)
            self.updateFrozenRules(solution)

            radius = 1
            while True:
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
                    if radius > self.maximumRadius:
                        print "I refuse to use a radius this big."
                        self.windowSize -= 1
                        if self.windowSize > 0:
                            print "Decreased window size to %s"%self.windowSize
                            radius = 1
                            break # break out of the loop over different radius sizes
                        
                        print "Can't shrink the window anymore so I'm just going to return"
                        return solution.toFrontier()
                    continue # retreat back to the loop over different radii
                except SynthesisTimeout: return solution.toFrontier()

                # Successfully explained a new data item

                # Update both the training data and solution
                solution = newSolution
                j += self.windowSize
                
                break # break out the loop over different radius sizes

            if saveProgressTo != None:
                print " [+] Saving progress to %s"%saveProgressTo
                dumpPickle((j,None,solution),saveProgressTo)            

        return self.expandFrontier(self.solveUnderlyingForms(solution), k)
