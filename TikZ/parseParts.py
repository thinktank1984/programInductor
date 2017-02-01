from render import render,animateMatrices
from language import *
from random import random,choice,seed
from time import time

t = str(time())
seed(t)
print "seed:",t

originalProgram = Sequence.sample()
#originalProgram = Rectangle((1,1),(2,4))
print "TARGET:"
print originalProgram
print " ==  ==  == "
target = render(str(originalProgram), showImage = False, yieldsPixels = True)
#animateMatrices([target])

actual = Sequence.sample()

history = {}
def fastPixels(program):
    program = str(program)
    if program in history: return history[program]
    x = render(program, showImage = False, yieldsPixels = True)
    history[program] = x
    return x

def distance(x,y):
    d = (x - y)**2
    return d.sum()
proposals = []
def error(program):
    global proposals
    p = fastPixels(program)
#    proposals.append(p)
    return distance(p, target)

startTime = time()
best = error(actual)


for _ in range(1000):
    child = actual.mutate()
    childFitness = error(child)
    if childFitness < best or random() < 0.0:
        best = childFitness
        actual = child
        print "error:",best
        proposals.append(fastPixels(child))
        print actual
    if best == 0.0:
        break

print "Total renders:",len(history),"in",(time() - startTime),"seconds"
if best > 0.0:
    print "FAILEDTOFINDASOLUTION"
print error(actual)
print error(originalProgram)

#animateMatrices(proposals)
