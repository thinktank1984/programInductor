from render import render,animateMatrices
from language import *
from random import random,choice,seed
from time import time

t = str(time())
seed(t)
print "seed:",t

#render([str(Sequence.sample()) for _ in range(100) ],showImage = False,yieldsPixels = True)

originalProgram = Sequence.sample()
#originalProgram = Rectangle((1,1),(2,4))
print "TARGET:"
print originalProgram
print " ==  ==  == "
target = render([str(originalProgram)], showImage = False, yieldsPixels = True)[0]
#animateMatrices([target])

actual = Sequence.sample()

history = {}
def fastPixels(programs):
    programs = map(str,programs)
    toRender = list(set([p for p in programs if not (p in history) ]))
    if toRender != []:
        renders = render(toRender, yieldsPixels = True)
        for r,p in zip(renders, toRender):
            history[p] = r
    return [history[p] for p in programs ]

def distance(x,y):
    d = (x - y)**2
    return d.sum()
def error(programs):
    return [distance(p, target) for p in fastPixels(programs)] 

startTime = time()
best = error([actual])[0]

BRANCHINGFACTOR = 100
proposals = []

for generation in range(10):
    print "GENERATION:",generation
    children = [ actual.mutate() for _ in range(BRANCHINGFACTOR) ]
    childFitness = error(children)
    (childFitness,child) = min(zip(childFitness,children))
    if childFitness < best or random() < 0.0:
        best = childFitness
        actual = child
        print "error:",best
        proposals.append(fastPixels([child])[0])
        print actual
    if best == 0.0:
        break

print "Total renders:",len(history),"in",(time() - startTime),"seconds"
if best > 0.0:
    print "FAILEDTOFINDASOLUTION"
print error([actual])[0]
print error([originalProgram])[0]

#animateMatrices(proposals)
