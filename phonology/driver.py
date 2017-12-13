from problems import underlyingProblems,interactingProblems,sevenProblems,nineProblems,MATRIXPROBLEMS
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
        restriction = p.data[parameters['restrict'][0] : parameters['restrict'][1]]
    else: restriction = p.data
        
    print p.description
    if problemIndex != 7:
        print u"\n".join([ u"\t".join(map(unicode,inflections)) for inflections in restriction ])
    else:
        print CountingProblem(p.data, p.parameters).latex()

    if parameters['universalGrammar'] != None:
        assert parameters['universalGrammar'].endswith('.p')
        universalGrammarPath = parameters['universalGrammar']
        if parameters['curriculum']:
            index = MATRIXPROBLEMS.index(p)
            universalGrammarPath = universalGrammarPath[:-2] + "_curriculum" + str(index) + ".p"
            
        if not os.path.exists(universalGrammarPath):
            print "Fatal error: Cannot find universal grammar",universalGrammarPath
            assert False
            
        ug = FragmentGrammar.load(universalGrammarPath)
        print "Loaded %s:\n%s"%(universalGrammarPath,ug)
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
        problem = UnderlyingProblem(p.data, UG = ug).restrict(restriction)
        if parameters['task'] == 'stochastic':
            problem.stochasticSearch(20, parameters['beam'])
        elif parameters['task'] == 'debug':
            for s in p.solutions:
                s = parseSolution(s)
                problem.debugSolution(s,Morph(tokenize(parameters['debug'])))
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
            ss = []
        elif parameters['task'] == 'ransac':
            RandomSampleSolver(p.data, parameters['timeout']*60*60, 10, 25, UG = ug, dummy = parameters['dummy']).\
                restrict(restriction).\
                solve(numberOfWorkers = parameters['cores'],
                      numberOfSamples = parameters['samples'])
            assert False
        elif parameters['task'] == 'incremental':
            ss = IncrementalSolver(p.data,parameters['window'],UG = ug,
                                   numberOfCPUs = 1 if parameters['serial'] else None).\
                 restrict(restriction).\
                 incrementallySolve(saveProgressTo = parameters['save'],
                                    loadProgressFrom = parameters['restore'])
        elif parameters['task'] == 'CEGIS':
            ss = problem.counterexampleSolution(k = parameters['top'],
                                                threshold = parameters['threshold'])
        elif parameters['task'] == 'exact':
            ss = problem.sketchJointSolution(1, canAddNewRules = True)
        elif parameters['task'] == 'frontier':
            f = str(problemIndex) + ".p"
            seed = os.path.join(parameters['restore'], f)
            if not os.path.exists(seed):
                print "Skipping frontier job %d, because I can't find %s"%(problemIndex,seed)
                sys.exit(0)
            seed = loadPickle(seed)
            assert isinstance(seed,list)
            assert len(seed) == 1
            worker = problem
            seed = worker.solveUnderlyingForms(seed[0])
            frontier = worker.solveFrontiers(seed, k = parameters['top'])
            dumpPickle(frontier, os.path.join(parameters['save'], f))
            sys.exit(0)

        if not isinstance(ss,list): ss = [ss]
        

    print "Total time taken by problem %d: %f seconds"%(problemIndex, time() - startTime)

    if parameters['redirect']:
        sys.stdout,sys.stderr = oldOutput,oldErrors
        handle.close()

    if parameters['pickleDirectory'] != None:
        fullPath = os.path.join(parameters['pickleDirectory'], "matrix_" + str(problemIndex) + ".p")
        if not os.path.exists(parameters['pickleDirectory']):
            os.mkdir(parameters['pickleDirectory'])
        dumpPickle(ss, fullPath)
        
                




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
    parser.add_argument('-u','--universal', default = None, type = str)
    parser.add_argument('--window', default = None, type = int)
    parser.add_argument('--save', default = None, type = str)
    parser.add_argument('--restore', default = None, type = str)
    parser.add_argument('--debug', default = None, type = lambda s: unicode(s,'utf8'))
    parser.add_argument('--restrict', default = None, type = str)
    parser.add_argument('--samples', default = 30, type = int)
    parser.add_argument('--curriculum', default = False, action = 'store_true',
                        help = "Only use in conjunction with universal grammar. Specifies that the loaded UG should be the one calculated from previous problems. see the curriculum option in UG.py")
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
        restriction = tuple(map(int,[offset for offset in arguments.restrict.split(":") if offset != '']))
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
                   'universalGrammar': arguments.universal,
                   'top': arguments.top,
                   'curriculum': arguments.curriculum,
                   'task': arguments.task,
                   'threshold': arguments.threshold,
                   'redirect': False,
                   'window': arguments.window,
                   'debug': arguments.debug,
                   'save': arguments.save,
                   'restore': arguments.restore,
                   "restrict": arguments.restrict,
                   'cores': arguments.cores,
                   'beam': arguments.beam,
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
        
