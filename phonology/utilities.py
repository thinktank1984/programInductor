# -*- coding: utf-8 -*-
import tempfile
import sys
import pickle
import math
import random
import itertools

def compose(f,g):
    return lambda x: f(g(x))
def mergeCounts(m,n):
    c = {}
    for f in set(m.keys() + n.keys()):
        c[f] = m.get(f,0.0) + n.get(f,0.0)
    return c
def isFinite(x):
    return not (math.isnan(x) or math.isinf(x))
def lse(x,y):
    if not isFinite(x): return y
    if not isFinite(y): return x
    
    if x > y:
        return x + math.log(1 + math.exp(y - x))
    else:
        return y + math.log(1 + math.exp(x - y))
def lseList(l):
    a = l[0]
    for x in l[1:]: a = lse(a,x)
    return a
def normalizeLogDistribution(d):
    z = lseList([w for w,_ in d ])
    return [(w-z,x) for w,x in d ]


def randomTestSplit(data,ratio):
    """ratio: what fraction is testing data. returns training,test"""
    testingSize = min(len(data) - 1, int(round(len(data)*ratio)))
    trainingSize = len(data) - testingSize
    shuffledData = list(data)
    random.shuffle(shuffledData)
    training, test = shuffledData[:trainingSize], shuffledData[trainingSize:]
    return [ x for x in data if x in training ], [ x for x in data if x in test ]

def flatten(xss):
    return [ x for xs in xss for x in xs ]

def randomlyRemoveOne(x):
    t = random.choice(x)
    return [ y for y in x if t != y ]

def everyBinaryVector(l,w):
    if l == 0:
        if w == 0: yield []
    elif w > -1:
        for v in everyBinaryVector(l - 1,w):
            yield [False] + v
        for v in everyBinaryVector(l - 1,w - 1):
            yield [True] + v

def everyPermutation(l,r):
    # every permutation of 0 -- (l-1)
    # each permutation is constrained to exchange exactly r elements
    assert r > 1
    for exchangedElements in itertools.combinations(range(l),r):
        for perm in itertools.permutations(exchangedElements):
            # every element has to be mapped to a new one
            if any([ p == e for p,e in zip(list(perm),list(exchangedElements)) ]): continue

            returnValue = list(range(l))
            for p,e in zip(list(perm),list(exchangedElements)):
                returnValue[e] = p
            yield returnValue

def everyEditSequence(sequence, radii):
    def _everySequenceEdit(r):
        # radius zero
        if r < 1: return [list(range(len(sequence)))]

        edits = []
        for s in _everySequenceEdit(r - 1):
            # Should we consider adding a new thing to the sequence?
            if len(s) == len(sequence):
                edits += [ s[:j] + [None] + s[j:] for j in range(len(s) + 1) ]
            # Consider doing over any one element of the sequence
            edits += [ s[:j] + [None] + s[j+1:] for j in range(len(s)) ]
            # Consider swapping elements
            edits += [ [ (s[i] if k == j else (s[j] if k == i else s[k])) for k in range(len(s)) ]
                       for j in range(len(s) - 1)
                       for i in range(j,len(s)) ]
        return edits

    # remove duplicates
    candidates = set([ tuple(s)
                       for radius in radii
                       for s in _everySequenceEdit(radius) ] )
    # remove things that came from an earlier radius
    for smallerRadius in range(min(radii)):
        candidates -= set([ tuple(s) for s in _everySequenceEdit(smallerRadius) ])
    # some of the edit sequences might subsume other ones, eg [None,1,None] subsumes [0,1,None]
    # we want to not include things that are subsumed by other things

    def subsumes(moreGeneral, moreSpecific):
        if not len(moreGeneral) == len(moreSpecific): return False
        for g,s in zip(moreGeneral,moreSpecific):
            if g != None and s != g: return False
        #print "%s is strictly more general than %s"%(moreGeneral,moreSpecific)
        return True

    # disabling subsumption removal
    removedSubsumption = [ s
                           for s in candidates ]
                           # if not any([ subsumes(t,s) for t in candidates if t != s ]) ]
    
    # reindex into the input sequence
    return [ [ (None if j == None else sequence[j]) for j in s ]
             for s in removedSubsumption ]

def dumpPickle(o,f):
    with open(f,'wb') as handle:
        pickle.dump(o,handle)
def loadPickle(f):
    with open(f,'rb') as handle:
        o = pickle.load(handle)
    return o
def flushEverything():
    sys.stdout.flush()
    sys.stderr.flush()
def makeTemporaryFile(suffix, d = '.'):
    fd = tempfile.NamedTemporaryFile(mode = 'w',suffix = suffix,delete = False,dir = d)
    fd.write('')
    fd.close()
    return fd.name



VERBOSITYLEVEL = 0
def getVerbosity():
    global VERBOSITYLEVEL
    return VERBOSITYLEVEL
def setVerbosity(v):
    global VERBOSITYLEVEL
    VERBOSITYLEVEL = v

def sampleGeometric(p):
    if random.random() < p: return 0
    return 1 + sampleGeometric(p)


def numberOfCPUs():
    import multiprocessing
    return multiprocessing.cpu_count()

def indent(s):
    return '\t' + s.replace('\n','\n\t')

