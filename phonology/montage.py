from problems import *
from textbook_problems import *

from utilities import *
import pickle
import os
import sys

import numpy as np

import matplotlib.pyplot as plot


class Bars():
    def __init__(self, problem, universal, *baselines):
        self.problem = problem
        self.baselines = baselines
        self.universal = universal
        self.name = "%s (%s)"%(self.problem.languageName, self.problem.source)
        print self.name, any( b is None for b in baselines), self.universal is None
        #self.name = self.problem.languageName

    @property
    def alternation(self):
        return self.problem.parameters and "alternations" in self.problem.parameters



    @property
    def language(self): return self.problem.languageName

    def universalHeight(self):
        if self.alternation: return 1.
        if self.problem.key == "Odden_2.4_Tibetan": return 1.
        if self.universal is None: return 0.
        n = len(self.problem.data)
        return float(len(self.universal.finalFrontier.underlyingForms))/n

    def baselineHeight(self, b):
        if self.alternation: return 1.
        assert not self.alternation
        if self.problem.key == "Odden_2.4_Tibetan": return 1.
        b = self.baselines[b]
        if b is None: return 0.
        n = len(self.problem.data)
        return float(len(b.finalFrontier.underlyingForms))/n

    def __str__(self):
        return "Bars(%s,%f)"%(self.name, self.universalHeight())

    def __repr__(self):
        return str(self)
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "Graphs the the system on all of the languages")
    
    arguments = parser.parse_args()

    baselinePath_1 = ["experimentOutputs/%s_incremental_disableClean=False_features=sophisticated_geometry=False.p"]
    baselinePath_2 = ["experimentOutputs/%s_CEGIS_disableClean=False_features=sophisticated.p"]
    universalPath = ["experimentOutputs/%s_incremental_disableClean=False_features=sophisticated_geometry=True_ug.p",
                     "experimentOutputs/%s_incremental_disableClean=False_features=sophisticated_geometry=True.p"]
    bars = []
    for name, problem in Problem.named.iteritems():
        if "Kevin" in name: continue

        if name == "Odden_2.4_Tibetan":
            bl = "experimentOutputs/Odden_2.4_Tibetan_exact_disableClean=False_features=sophisticated.p"
            ul = "experimentOutputs/Odden_2.4_Tibetan_exact_disableClean=False_features=sophisticated_geometry=True.p"
            if os.path.exists(bl): bl = loadPickle(bl)
            else: bl = None
            bl_1 = bl
            bl_2 = bl
            if os.path.exists(ul): ul = loadPickle(ul)
            else: ul = None
        elif problem.parameters and "alternations" in problem.parameters:
            if os.path.exists("experimentOutputs/alternation/%s.p"%name):
                p = loadPickle("experimentOutputs/alternation/%s.p"%name)
            else:
                p = None
            bl_1 = p
            bl_2 = p
            ul = p
            if p is None:
                print "Missing alternation",name
        else:
            bl_1 = None
            for b in baselinePath_1:
                if os.path.exists(b%name):
                    bl_1 = loadPickle(b%name)
                    break
            bl_2 = None
            for b in baselinePath_2:
                if os.path.exists(b%name):
                    bl_2 = loadPickle(b%name)
                    break
            ul = None
            for u in universalPath:
                if os.path.exists(u%name):
                    ul = loadPickle(u%name)
        
        bars.append(Bars(problem,ul,bl_1,bl_2))

    bars.sort(key=lambda b: (not b.alternation, -b.universalHeight()))

    if False:
        for n,b in enumerate(bars):
            if b.alternation: b.name = b.problem.languageName + "*"
            else:
                if sum(b.language == o.language for o in bars if not b.alternation ) > 1:
                    i = sum(b.language == o.language for o in bars[:n + 1]
                            if not b.alternation)
                    b.name = b.language + " (" + "I"*i + ")"
                else:
                    b.name = b.language


    columns = 3
    f, axes = plot.subplots(1,columns)
    # partition into columns
    partitions = partitionEvenly(bars,columns)
    #f.yticks(rotation=45)
    for bs,a in zip(partitions,axes):
        bs.reverse()
        
        W = 0.4
        ys = np.arange((len(bs)))
        
        a.barh(ys,
               [b.universalHeight() for b in bs ],
               W,
               color='b'*len(bs))
#               tick_label=[b.name for b in bs ])
        a.barh(ys + W,
               [b.baselineHeight(1) for b in bs ],
               W,
               color='g')
        a.set(yticks=ys + W,
              yticklabels=[b.name for b in bs ])

    plot.show()

