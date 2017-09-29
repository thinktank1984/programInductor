from problems import underlyingProblems,interactingProblems,sevenProblems,nineProblems
from countingProblems import CountingProblem
from utilities import *
from parseSPE import parseSolution

from matrix import *
from randomSampleSolver import RandomSampleSolver

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
    
    print p.description
    if problemIndex != 7:
        print u"\n".join([ u"\t".join(map(unicode,inflections)) for inflections in p.data ])
    else:
        print CountingProblem(p.data, p.parameters).latex()

    startTime = time()

    ss = None # solutions to save out to the pickled file
    accuracy, compression = None, None
    
    if problemIndex == 7:
        ss = CountingProblem(p.data, p.parameters).topSolutions(parameters['top'])
    else:
        if parameters['testing'] == 0.0:
            if parameters['stochastic']:
                UnderlyingProblem(p.data).stochasticSearch(20, parameters['beam'])
            elif parameters['verify']:
                for s in p.solutions:
                    s = parseSolution(s)
                    print "verifying:"
                    print s
                    list(UnderlyingProblem(p.data).findCounterexamples(s))
                ss = []
            elif parameters['randomSample']:
                RandomSampleSolver(p.data, parameters['timeout']*60*60, 5, 15).solve(numberOfWorkers = parameters['cores'])
                assert False
            elif parameters['incremental']:
                ss = UnderlyingProblem(p.data).incrementallySolve(windowSize = parameters['window'],
                                                                  beam = parameters['beam'],
                                                                  eager = parameters['eager'],
                                                                  saveProgressTo = parameters['save'],
                                                                  loadProgressFrom = parameters['restore'])
            else:
                ss = UnderlyingProblem(p.data).counterexampleSolution()
            print "ss = "
            print ss
        else:
            ss, accuracy, compression = heldOutSolution(p.data,
                                                        parameters['top'],
                                                        parameters['threshold'],
                                                        testing = parameters['testing'],
                                                        inductiveBiases = parameters['universalGrammar'])
        if not isinstance(ss,list): ss = [ss]
        ss = [s.rules for s in ss ] # just save the rules

    print "Total time taken by problem %d: %f seconds"%(problemIndex, time() - startTime)

    if parameters['redirect']:
        sys.stdout,sys.stderr = oldOutput,oldErrors
        handle.close()
    
    if ss != None and parameters['top'] > 1 and parameters['testing'] == 0.0:
        dumpPickle(ss, "pickles/matrix_"+str(problemIndex)+".p")
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
    parser.add_argument('-t','--top', default = 1, type = int)
    parser.add_argument('-f','--threshold', default = float('inf'), type = int)
    parser.add_argument('-m','--cores', default = 1, type = int)
    parser.add_argument('--timeout', default = 1.0, type = float,
                        help = 'timeout for ransac solver. can be a real number. measured in hours.')
    parser.add_argument('-s','--seed', default = '0', type = str)
    parser.add_argument('-H','--hold', default = '0.0', type = str)
    parser.add_argument('-u','--universal', default = 'flat',type = str)
    parser.add_argument('--window', default = 2, type = int)
    parser.add_argument('--save', default = None, type = str)
    parser.add_argument('--restore', default = None, type = str)
    parser.add_argument('--eager', default = False, action = 'store_true')
    parser.add_argument('--randomSample', default = False, action = 'store_true',
                        help = 'ransac style solver')
    parser.add_argument('--stochastic', default = False, action = 'store_true')
    parser.add_argument('--incremental', default = False, action = 'store_true')
    parser.add_argument('--beam',default = 1,type = int)
    parser.add_argument('--verify',action = 'store_true')
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
                    # Chapter five problems
                    51,
                    52,
                    53]
    else:
        problems = map(int,arguments.problem.split(','))

    parameters = [{'problemIndex': problemIndex,
                   'seed': seed,
                   'testing': testing,
                   'randomSample': arguments.randomSample,
                   'universalGrammar': arguments.universal.split(','),
                   'top': arguments.top,
                   'verify': arguments.verify,
                   'threshold': arguments.threshold,
                   'redirect': False,
                   'incremental': arguments.incremental,
                   'window': arguments.window,
                   'save': arguments.save,
                   'restore': arguments.restore,
                   'eager': arguments.eager,
                   'beam': arguments.beam,
                   'stochastic': arguments.stochastic,
                   'timeout': arguments.timeout,
                   }
                  for problemIndex in problems
                  for seed in map(int,arguments.seed.split(','))
                  for testing in map(float,arguments.hold.split(',')) ]
    print parameters
    
    if arguments.cores > 1 and False:
        Pool(arguments.cores).map(handleProblem, parameters)
    else:
        map(handleProblem, parameters)
        
