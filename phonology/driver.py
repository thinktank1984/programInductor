from sketch import disableFeatures, disableClean
from features import switchFeatures
from problems import MATRIXPROBLEMS
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

from command_server import start_server

def handleProblem(parameters):
    problemIndex = parameters['problemIndex']
    random.seed(parameters['seed'] + problemIndex)

    p = MATRIXPROBLEMS[problemIndex]

    if parameters['restrict'] != None:
        print "(Restricting problem data to interval: %d -- %d)"%(parameters['restrict'][0],parameters['restrict'][1])
        restriction = p.data[parameters['restrict'][0] : parameters['restrict'][1]]
    else: restriction = p.data
        
    print p.description
    isCountingProblem = isinstance(p.parameters, list) \
                        and all( isinstance(parameter,int) for parameter in p.parameters  )
    if not isCountingProblem:
        print u"\n".join([ u"\t".join(map(unicode,inflections)) for inflections in restriction ])
    else:
        print CountingProblem(p.data, p.parameters).latex()

    if parameters['universalGrammar'] != None:
        assert parameters['universalGrammar'].endswith('.p')
        universalGrammarPath = parameters['universalGrammar']
            
        if not os.path.exists(universalGrammarPath):
            print "Fatal error: Cannot find universal grammar",universalGrammarPath
            assert False
            
        ug = FragmentGrammar.load(universalGrammarPath)
        print "Loaded %s:\n%s"%(universalGrammarPath,ug)
    else: ug = None

    startTime = time()

    ss = None # solutions to save out to the pickled file
    accuracy, compression = None, None
    
    if isCountingProblem:
        problem = CountingProblem(p.data, p.parameters)
        parameters['task'] = 'exact'
    else:
        problem = UnderlyingProblem(p.data, UG = ug).restrict(restriction)
    
    if parameters['task'] == 'debug':
        for s in p.solutions:
            s = parseSolution(s)
            problem.debugSolution(s,Morph(tokenize(parameters['debug'])))
        sys.exit(0)
    elif parameters['task'] == 'verify':
        for s in p.solutions:
            s = parseSolution(s)
            print "verifying:"
            print s
            b = UnderlyingProblem(p.data).bank
            for r in s.rules:
                print "Explaining rule: ",r
                r.explain(b)
            problem.illustrateSolution(s)
        sys.exit(0)
        
    elif parameters['task'] == 'ransac':
        RandomSampleSolver(p.data, parameters['timeout']*60*60, 10, 25, UG = ug, dummy = parameters['dummy']).\
            restrict(restriction).\
            solve(numberOfWorkers = parameters['cores'],
                  numberOfSamples = parameters['samples'])
        sys.exit(0)
        
    elif parameters['task'] == 'incremental':
        ss = IncrementalSolver(p.data,parameters['window'],UG = ug,
                               problemName = str(problemIndex),
                               numberOfCPUs = 1 if parameters['serial'] else None).\
             restrict(restriction).\
             incrementallySolve(resume = parameters['resume'],                                
                                k = parameters['top'])
    elif parameters['task'] == 'CEGIS':
        ss = problem.counterexampleSolution(k = parameters['top'])
    elif parameters['task'] == 'exact':
        s = problem.sketchJointSolution(1, canAddNewRules = True)
        ss = problem.expandFrontier(s, parameters['top'])
    elif parameters['task'] == 'frontier':
        f = str(problemIndex) + ".p"
        seed = os.path.join(parameters['restore'], f)
        if not os.path.exists(seed):
            print "Skipping frontier job %d, because I can't find %s"%(problemIndex,seed)
            sys.exit(0)
        seed = loadPickle(seed)
        assert isinstance(seed,Frontier)
        worker = problem
        seed = worker.solveUnderlyingForms(seed[0])
        frontier = worker.solveFrontiers(seed, k = parameters['top'])
        dumpPickle(frontier, os.path.join(parameters['save'], f))
        sys.exit(0)

    assert isinstance(ss,Frontier)
    print ss

    print "Total time taken by problem %d: %f seconds"%(problemIndex, time() - startTime)

    if parameters['pickleDirectory'] != None:
        fullPath = os.path.join(parameters['pickleDirectory'], "matrix_" + str(problemIndex) + ".p")
        if not os.path.exists(parameters['pickleDirectory']):
            os.mkdir(parameters['pickleDirectory'])
        dumpPickle(ss, fullPath)
        print "Exported frontier to",fullPath
        
                


def paretoFrontier(problemIndex):
    p = MATRIXPROBLEMS[problemIndex]
    print p.description
    random.seed(0)
    data = randomlyPermute(p.data)
    print "\n".join(map(str,data))
    if arguments.restrict:
        print "(Restricting problem data to interval: %d -- %d)"%(arguments.restrict[0],
                                                                  arguments.restrict[1])
        data = data[arguments.restrict[0] : arguments.restrict[1]]
        print "\n".join(map(str,data))
    p = UnderlyingProblem(data)
    paretoFront = p.paretoFront(3, 20, 1,
                                useMorphology=True)
    if arguments.pickleDirectory is not None:
       path = arguments.pickleDirectory + "/" + str(problemIndex) + "_paretoFrontier.p"
       dumpPickle(paretoFront, path)
       print "Exported Pareto frontier to",path
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Solve jointly for morphology and phonology given surface inflected forms of lexemes')
    parser.add_argument('problem')
    parser.add_argument('task',
                        choices = ["CEGIS","incremental","ransac","exact",
                                   "debug","verify","frontier","pareto"],
                        default = "CEGIS",
                        type = str,
                        help = "The task you are asking the driver to initiate.")
    parser.add_argument('--features',
                        choices = ["none","sophisticated","simple"],
                        default = "sophisticated",
                        type = str,
                        help = "What features the solver allowed to use")
    parser.add_argument('-t','--top', default = 1, type = int)
    parser.add_argument('-m','--cores', default = 1, type = int)
    parser.add_argument('--timeout', default = 1.0, type = float,
                        help = 'timeout for ransac solver. can be a real number. measured in hours.')
    parser.add_argument('--serial', default = False, action = 'store_true',
                        help = 'Run the incremental solver in serial mode (no parallelism)')
    parser.add_argument('--disableClean', default = False, action = 'store_true',
                        help = 'disable kleene star')
    parser.add_argument('--resume', default = False, action = 'store_true',
                        help = 'Resume the incremental solver from the last checkpoint')
    parser.add_argument('--dummy', default = False, action = 'store_true',
                        help = 'Dont actually run the solver for ransac')
    parser.add_argument('-s','--seed', default = '0', type = str)
    parser.add_argument('-u','--universal', default = None, type = str)
    parser.add_argument('--window', default = None, type = int)
    parser.add_argument('--save', default = None, type = str)
    parser.add_argument('--restore', default = None, type = str)
    parser.add_argument('--debug', default = None, type = lambda s: unicode(s,'utf8'))
    parser.add_argument('--restrict', default = None, type = str)
    parser.add_argument('--samples', default = 30, type = int)
    parser.add_argument('--pickleDirectory',default = None,type = str)
    parser.add_argument('-V','--verbosity', default = 0, type = int)

    arguments = parser.parse_args()
    setVerbosity(arguments.verbosity)
    
    if arguments.features == "none":
        disableFeatures()
    else:
        print "Using the `%s` feature set"%(arguments.features)
        switchFeatures(arguments.features)
        
    if arguments.disableClean:
        disableClean()
    
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
        restriction = tuple(map(int,[offset for offset in arguments.restrict.split(":") if offset != '']))
        if len(restriction) == 1:
            if arguments.restrict.startswith(":"):
                restriction = [0,restriction[0]]
            elif arguments.restrict.endswith(":"):
                restriction = [restriction[0],99999]
            else:
                assert False, ("Invalid restriction expression:"+arguments.restrict)
        arguments.restrict = restriction

    start_server(arguments.cores)

    if arguments.task == "pareto":
        # a quick hack
        assert len(problems) == 1
        paretoFrontier(problems[0])
        sys.exit(0)
        

    parameters = [{'problemIndex': problemIndex,
                   'seed': seed,
                   'universalGrammar': arguments.universal,
                   'top': arguments.top,
                   'task': arguments.task,
                   'window': arguments.window,
                   'resume': arguments.resume,
                   'debug': arguments.debug,
                   'save': arguments.save,
                   'restore': arguments.restore,
                   "restrict": arguments.restrict,
                   'cores': arguments.cores,
                   'timeout': arguments.timeout,
                   'pickleDirectory': arguments.pickleDirectory,
                   'serial': arguments.serial,
                   'samples': arguments.samples,
                   'dummy': arguments.dummy,
                   }
                  for problemIndex in problems
                  for seed in map(int,arguments.seed.split(',')) ]
    print parameters
    
    if arguments.cores > 1 and arguments.problem == 'integration':
        Pool(arguments.cores).map(handleProblem, parameters)
    else:
        map(handleProblem, parameters)
        
