# -*- coding: utf-8 -*-

import math
import random

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

def everyBinaryVector(l,w):
    if l == 0:
        if w == 0: yield []
    elif w > -1:
        for v in everyBinaryVector(l - 1,w):
            yield [False] + v
        for v in everyBinaryVector(l - 1,w - 1):
            yield [True] + v
