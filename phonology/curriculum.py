import os
import sys

def numberOfCPUs():
    import multiprocessing
    return multiprocessing.cpu_count()


def flushEverything():
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "Curriculum solving of phonology problems. Calls out to UG.py and driver.py")
    parser.add_argument("startingIndex",
                        type=int,
                        help="Which problem to start out solving. NOT 0-indexed (e.g. problem 1 is the first)")
    parser.add_argument("endingIndex",
                        type=int)
    parser.add_argument("ug",
                        choices=["empirical","ground","none"],
                        help="What kind of universal grammars to use. empirical: estimate from solutions found to previous problems. ground: estimate using textbook solutions. none: do not use universal grammar.")
    parser.add_argument("--CPUs",
                        type=int,
                        default=None)
    
    arguments = parser.parse_args()

    CPUs = arguments.CPUs or numberOfCPUs()
    print("Using %d CPUs"%CPUs)

    os.system("python command_server.py %d &"%CPUs)

    if arguments.ug == "ground":
        print "Precomputing ground-truth universal grammars..."
        for j in xrange(arguments.startingIndex, arguments.endingIndex+1):
            os.system("pypy UG.py fromGroundTruth --CPUs %d --problems %d --export universalGrammars/groundTruth_%d.p"%(arguments.CPUs, j, j))

    for j in xrange(arguments.startingIndex, arguments.endingIndex+1):
        print("Solving problem %d"%j)

        if j == 1 or arguments.ug == "none":
            u = ""
        elif arguments.ug == "empirical":
            u = "--universal universalGrammars/empirical_%d.p"%(j - 1)
        elif arguments.ug == "ground":
            u = "--universal universalGrammars/groundTruth_%d.p"%(j - 1)
        else: assert False

        command = "python driver.py %s incremental --top 100 %s --pickleDirectory frontierPickles/"%(j,u)
        print
        print "\tCURRICULUM: Solving problem %d by issuing the command:"%j
        print "\t\t",command
        flushEverything()
        os.system(command)

        if arguments.ug == "empirical":
            command = "pypy UG.py fromFrontiers --CPUs %d --problems %d --export universalGrammars/empirical_%d.p"%(CPUs, j, j)
            print
            print "Re- estimating universal grammar by executing:"
            print command
            os.system(command)
            
            
