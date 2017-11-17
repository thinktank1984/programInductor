from problems import underlyingProblems,interactingProblems,sevenProblems,nineProblems
from countingProblems import CountingProblem
from utilities import *
from parseSPE import parseSolution

from fragmentGrammar import FragmentGrammar
from matrix import *
from randomSampleSolver import RandomSampleSolver
from incremental import IncrementalSolver

import argparse
from multiprocessing import Pool
import sys
import io

def heldOutSolution(data, k = 1, threshold = float('inf'), initialTrainingSize = 2, testing = 0.0, inductiveBiases = []):
    numberOfInflections = len(data[0])
    if numberOfInflections > 7: initialTrainingSize = 3
    bank = UnderlyingProblem(data).bank

    trainingData,testingData = randomTestSplit(data, testing)
    slave = UnderlyingProblem(trainingData, bank = bank)
    
    solutions = slave.counterexampleSolution(k,threshold,initialTrainingSize)

    accuracies, compressions = {}, {}
    for bias in inductiveBiases:
        print "Considering bias",bias
        ug = str2ug(bias)
        solution = max(solutions, key = lambda z: ug.logLikelihood(z.rules))
        accuracy,compression = accuracyAndCompression(solution, testingData, bank)
        print "Average held out accuracy: ",accuracy
        print "Average held out compression:",compression
        print "As a test, trying to calculate on the original training data also:"
        print accuracyAndCompression(solution, trainingData)
        accuracies[bias] = accuracy
        compressions[bias] = compression
    return solutions, accuracies, compressions

def accuracyAndCompression(solution, testingData, bank):
    accuracy,compression = 0,0
    for inflections in testingData:
        a,c = inflectionAccuracy(solution, inflections, bank)
        compression += c
        accuracy += a
    return accuracy/len(testingData), compression/float(len(testingData))

def inflectionAccuracy(solution, inflections, bank):
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
        prefixes = [ define("Word", p.makeConstant(bank)) for p in solution.prefixes ]
        suffixes = [ define("Word", s.makeConstant(bank)) for s in solution.suffixes ]
        rules = [ define("Rule", r.makeConstant(bank)) for r in solution.rules ]
        for r in rules: condition(fixStructuralChange(r))

        prediction = Morph.sample()
        surfaces = [ (s if j in trainingIndexes else prediction) for j,s in enumerate(inflections) ]

        worker = UnderlyingProblem([inflections], bank)
        worker.conditionOnStem(rules, stem, prefixes, suffixes, surfaces)

        # IMPORTANT!
        # Because we are modeling the prediction as a morph,
        # the maximum morph length most also be the maximum observation length
        output = worker.solveSketch()
        if not output or testingIndex == len(inflections):
            prediction = None
        else:
            prediction = Morph.parse(bank, output, prediction)
        if testingIndex < len(inflections): # testing ability to make new inflection
            if prediction == Morph(tokenize(inflections[testingIndex])):
                correctPredictions += 1
            else:
                print "I saw these inflections:","\t".join([s for j,s in enumerate(inflections)
                                                            if j != testingIndex])
                print "I predicted ", prediction,"instead of", Morph(tokenize(inflections[testingIndex]))
        else: # checking compression
            if output:
                encodingLength = len(Morph.parse(bank, output, stem))
    return correctPredictions/float(len(inflections)), encodingLength

def handleProblem(parameters):
    problemIndex = parameters['problemIndex']
    random.seed(parameters['seed'] + problemIndex)

    if problemIndex < 50:
        p = underlyingProblems[problemIndex - 1]
    elif str(problemIndex)[0] == '5':
        p = interactingProblems[int(str(problemIndex)[1:]) - 1]
    elif str(problemIndex)[0] == '7':
        p = sevenProblems[int(str(problemIndex)[1:]) - 1]
    elif str(problemIndex)[0] == '9':
        p = nineProblems[int(str(problemIndex)[1:]) - 1]

    if parameters['redirect']:
        redirectName = "multicore_output/%d"%problemIndex
        print "Redirecting output for problem %d to %s"%(problemIndex,redirectName)
        (oldOutput,oldErrors) = (sys.stdout,sys.stderr)
        handle = io.open(redirectName,'w',encoding = 'utf-8')#.character_encoding)
        sys.stdout = handle
        #        sys.stderr = handle

    if parameters['restrict'] != None:
        print "(Restricting problem data to interval: %d -- %d)"%(parameters['restrict'][0],parameters['restrict'][1])
        p.data = p.data[parameters['restrict'][0] : parameters['restrict'][1]]
    
    print p.description
    if problemIndex != 7:
        print u"\n".join([ u"\t".join(map(unicode,inflections)) for inflections in p.data ])
    else:
        print CountingProblem(p.data, p.parameters).latex()

    if parameters['universalGrammar'] != None:
        if not os.path.exists(parameters['universalGrammar']):
            print "Fatal error: Cannot find universal grammar",parameters['universalGrammar']
            assert False
        ug = FragmentGrammar.load(parameters['universalGrammar'])
        print "Loaded %s:\n%s"%(parameters['universalGrammar'],ug)
    else: ug = None

    startTime = time()

    ss = None # solutions to save out to the pickled file
    accuracy, compression = None, None
    
    if problemIndex == 7:
        if parameters['task'] != 'frontier':
            ss = CountingProblem(p.data, p.parameters).topSolutions(parameters['top'])
        else:
            f = str(problemIndex) + ".p"
            seed = os.path.join(parameters['restore'], f)
            if not os.path.exists(seed):
                print "Skipping frontier job %d, because I can't find %s"%(problemIndex,seed)
                sys.exit(0)
                
            seed = loadPickle(seed)
            assert isinstance(seed,list)
            assert len(seed) == 1
            frontier = CountingProblem(p.data, p.parameters).solveFrontiers(seed, k = parameters['top'])
            dumpPickle(frontier, os.path.join(parameters['save'], f))
            sys.exit(0)
            
    else:
        if parameters['testing'] == 0.0:
            if parameters['task'] == 'stochastic':
                UnderlyingProblem(p.data).stochasticSearch(20, parameters['beam'])
            elif parameters['task'] == 'debug':
                for s in p.solutions:
                    s = parseSolution(s)
                    UnderlyingProblem(p.data).debugSolution(s,Morph(tokenize(parameters['debug'])))
            elif parameters['task'] == 'verify':
                for s in p.solutions:
                    s = parseSolution(s)
                    print "verifying:"
                    print s
                    b = UnderlyingProblem(p.data).bank
                    for r in s.rules:
                        print "Explaining rule: ",r
                        r.explain(b)
                    UnderlyingProblem(p.data).illustrateSolution(s)
                ss = []
            elif parameters['task'] == 'ransac':
                RandomSampleSolver(p.data, parameters['timeout']*60*60, 10, 25, UG = ug, dummy = parameters['dummy']).\
                    solve(numberOfWorkers = parameters['cores'],
                          numberOfSamples = parameters['samples'])
                assert False
            elif parameters['task'] == 'incremental':
                ss = IncrementalSolver(p.data,parameters['window'],UG = ug).\
                     incrementallySolve(saveProgressTo = parameters['save'],
                                        loadProgressFrom = parameters['restore'])
            elif parameters['task'] == 'CEGIS':
                ss = UnderlyingProblem(p.data, UG = ug).counterexampleSolution(k = parameters['top'],
                                                                               threshold = parameters['threshold'])
            elif parameters['task'] == 'exact':
                ss = UnderlyingProblem(p.data).sketchJointSolution(1, canAddNewRules = True)
            elif parameters['task'] == 'frontier':
                f = str(problemIndex) + ".p"
                seed = os.path.join(parameters['restore'], f)
                if not os.path.exists(seed):
                    print "Skipping frontier job %d, because I can't find %s"%(problemIndex,seed)
                    sys.exit(0)
                seed = loadPickle(seed)
                assert isinstance(seed,list)
                assert len(seed) == 1
                worker = UnderlyingProblem(p.data)
                seed = worker.solveUnderlyingForms(seed[0])
                frontier = worker.solveFrontiers(seed, k = parameters['top'])
                dumpPickle(frontier, os.path.join(parameters['save'], f))
                sys.exit(0)
                
        else:
            ss, accuracy, compression = heldOutSolution(p.data,
                                                        parameters['top'],
                                                        parameters['threshold'],
                                                        testing = parameters['testing'],
                                                        inductiveBiases = parameters['universalGrammar'])
        if not isinstance(ss,list): ss = [ss]
        

    print "Total time taken by problem %d: %f seconds"%(problemIndex, time() - startTime)

    if parameters['redirect']:
        sys.stdout,sys.stderr = oldOutput,oldErrors
        handle.close()

    if parameters['pickleDirectory'] != None:
        fullPath = os.path.join(parameters['pickleDirectory'], str(problemIndex) + ".p")
        if not (ss != None and parameters['testing'] == 0.0):
            print "Exporting to %s, in spite of weird parameter settings"%fullPath
        if not os.path.exists(parameters['pickleDirectory']):
            os.mkdir(parameters['pickleDirectory'])
        dumpPickle(ss, fullPath)
    if accuracy != None and compression != None:
        parameters['accuracy'] = accuracy
        parameters['compression'] = compression
        print parameters
        name = "%d_%s_%f_%d_%d"%(parameters['problemIndex'],
                                 "_".join(sorted(parameters['universalGrammar'])),
                                 parameters['testing'],
                                 parameters['top'],
                                 parameters['seed'])
        dumpPickle(parameters, "testingAccuracy/%s.p"%name)
        
                




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Solve jointly for morphology and phonology given surface inflected forms of lexemes')
    parser.add_argument('problem')
    parser.add_argument('task',
                        choices = ["CEGIS","incremental","ransac","stochastic","exact",
                                   "debug","verify","frontier"],
                        default = "CEGIS",
                        type = str,
                        help = "The task you are asking the driver to initiate.")
    parser.add_argument('-t','--top', default = 1, type = int)
    parser.add_argument('-f','--threshold', default = float('inf'), type = int)
    parser.add_argument('-m','--cores', default = 1, type = int)
    parser.add_argument('--timeout', default = 1.0, type = float,
                        help = 'timeout for ransac solver. can be a real number. measured in hours.')
    parser.add_argument('--serial', default = False, action = 'store_true',
                        help = 'Run the incremental solver in serial mode (no parallelism)')
    parser.add_argument('--dummy', default = False, action = 'store_true',
                        help = 'Dont actually run the solver for ransac')
    parser.add_argument('-s','--seed', default = '0', type = str)
    parser.add_argument('-H','--hold', default = '0.0', type = str)
    parser.add_argument('-u','--universal', default = None, type = str)
    parser.add_argument('--window', default = 2, type = int)
    parser.add_argument('--save', default = None, type = str)
    parser.add_argument('--restore', default = None, type = str)
    parser.add_argument('--debug', default = None, type = unicode)
    parser.add_argument('--restrict', default = None, type = str)
    parser.add_argument('--samples', default = 30, type = int)
    parser.add_argument('--eager', default = False, action = 'store_true')
    parser.add_argument('--beam',default = 1,type = int)
    parser.add_argument('--pickleDirectory',default = None,type = str)
    parser.add_argument('-V','--verbosity', default = 0, type = int)

    arguments = parser.parse_args()
    setVerbosity(arguments.verbosity)
    
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
                    13,
                    14,
                    15,
                    # Chapter five problems
                    51,
                    52,
                    53]
    else:
        problems = map(int,arguments.problem.split(','))

    if arguments.restrict != None:        
        restriction = tuple(map(int,arguments.restrict.split(":")))
        if len(restriction) == 1:
            if arguments.restrict.startswith(":"):
                restriction = [0,restriction[0]]
            elif arguments.restrict.endswith(":"):
                restriction = [restriction[0],99999]
            else:
                assert False, ("Invalid restriction expression:"+arguments.restrict)
        arguments.restrict = restriction

    parameters = [{'problemIndex': problemIndex,
                   'seed': seed,
                   'testing': testing,
                   'universalGrammar': arguments.universal,
                   'top': arguments.top,
                   'task': arguments.task,
                   'threshold': arguments.threshold,
                   'redirect': False,
                   'window': arguments.window,
                   'debug': arguments.debug,
                   'save': arguments.save,
                   'restore': arguments.restore,
                   "restrict": arguments.restrict,
                   'eager': arguments.eager,
                   'cores': arguments.cores,
                   'beam': arguments.beam,
                   'timeout': arguments.timeout,
                   'pickleDirectory': arguments.pickleDirectory,
                   'serial': arguments.serial,
                   'samples': arguments.samples,
                   'dummy': arguments.dummy,
                   }
                  for problemIndex in problems
                  for seed in map(int,arguments.seed.split(','))
                  for testing in map(float,arguments.hold.split(',')) ]
    print parameters
    
    if arguments.cores > 1 and arguments.problem == 'integration':
        Pool(arguments.cores).map(handleProblem, parameters)
    else:
        map(handleProblem, parameters)
        
