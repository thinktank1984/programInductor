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
def lse(x,y):
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
    data = list(data)
    random.shuffle(data)
    return data[:trainingSize], data[trainingSize:]
