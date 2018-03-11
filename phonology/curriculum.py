import os
import sys

def flushEverything():
    sys.stdout.flush()
    sys.stderr.flush()


starting = 0
ending = 10
if len(sys.argv) > 1:
    starting = int(sys.argv[1])
    if len(sys.argv) > 2:
        ending = int(sys.argv[2])
        
for j in range(starting, ending):
    print "Solving problem #",j
    if j == 0:
        ug = ""
    else:
        ug = "--universal universalGrammars/empirical_%d.p"%j
    command = "python driver.py %s incremental --top 100 %s --pickleDirectory frontierPickles/"%(j,ug)
    print 
    print "\tCURRICULUM: Solving the next problem by issuing the command:"
    print "\t\t",command
    print
    flushEverything()
    os.system(command)

    command = "pypy UG.py fromFrontiers --problems %d --export universalGrammars/empirical_%d.p"%(j+1, j+1)
    print 
    print "\tCURRICULUM: Reestimating a universal grammar by issuing the command:"
    print "\t\t",command
    flushEverything()
    os.system(command)
    print 
