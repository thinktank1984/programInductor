# -*- coding: utf-8 -*-



from problems import underlyingProblems, interactingProblems, alternationProblems

import re
import math
import numpy as np
import matplotlib.pyplot as plot

def getRulesFromComment(problem):
    return [ l for l in problem.description.split("\n") if '--->' in l ]
def getFeaturesFromComment(problem):
    return [ f for r in getRulesFromComment(problem) for f in re.findall('[\-\+]([a-zA-Z]+)',r) ]

features = [ f
             for problem in underlyingProblems + interactingProblems + alternationProblems
             for f in getFeaturesFromComment(problem) ]
frequencies = list(reversed(sorted(list(set([ (len([y for y in features if y == x ]), x) for x in features ])))))
for c,f in frequencies:
    print f,c

# make a histogram of which features were popular
x = range(len(frequencies))
plot.bar(x, [float(c)/len(features) for c,f in frequencies ])
plot.xticks(x, [f for c,f in frequencies ], rotation = 'vertical')

plot.ylabel('Probability')

plot.show()

