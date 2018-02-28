import os

for j in range(10):
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
    os.system(command)

    command = "pypy UG.py fromFrontiers --problems %d --export universalGrammars/empirical_%d.p"%(j+1, j+1)
    print 
    print "\tCURRICULUM: Reestimating a universal grammar by issuing the command:"
    print "\t\t",command
    os.system(command)
    print 
