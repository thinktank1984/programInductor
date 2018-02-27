import os

for j in range(10):
    print "Solving problem #",j
    if j == 0:
        ug = ""
    else:
        ug = "--universal universalGrammars/empirical_%d.p"%j
    os.system("python driver.py %d incremental --top 100 %s --pickleDirectory frontierPickles/"%ug)
    print "Reestimating a universal grammar..."
    os.system("pypy UG.py fromFrontiers --problems %d --export universalGrammars/empirical_%d.p"%(j+1, j+1))
