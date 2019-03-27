from problems import *
from textbook_problems import *

from utilities import *
import pickle
import os
import sys

import matplotlib.pyplot as plot


class Bars():
    def __init__(self, problem, baseline, universal):
        self.problem = problem
        self.baseline = baseline
        self.universal = universal
        self.name = "%s (%s)"%(self.problem.languageName, self.problem.source)
        print self.name, self.baseline is None, self.universal is None

    @property
    def alternation(self):
        return self.problem.parameters and "alternations" in self.problem.parameters

    def universalHeight(self):
        if self.alternation: return 1.
        if self.problem.key == "Odden_2.4_Tibetan": return 1.
        if self.universal is None: return 0.
        n = len(self.problem.data)
        return float(len(self.universal.finalFrontier.underlyingForms))/n
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "Graphs the the system on all of the languages")
    
    arguments = parser.parse_args()

    baselinePath = ["experimentOutputs/%s_incremental_disableClean=False_features=sophisticated_geometry=False.p"]
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
            if os.path.exists(ul): ul = loadPickle(ul)
            else: ul = None
        elif problem.parameters and "alternations" in problem.parameters:
            if os.path.exists("experimentOutputs/alternation/%s.p"%name):
                p = loadPickle("experimentOutputs/alternation/%s.p"%name)
            else:
                p = None
            bl = p
            ul = p
            if p is None:
                print "Missing alternation",name
        else:
            bl = None
            for b in baselinePath:
                if os.path.exists(b%name):
                    bl = loadPickle(b%name)
                    break
            ul = None
            for u in universalPath:
                if os.path.exists(u%name):
                    ul = loadPickle(u%name)
        
        bars.append(Bars(problem,bl,ul))

    columns = 3
    f, axes = plot.subplots(1,columns)
    # partition into columns
    partitions = partitionEvenly(bars,columns)
    #f.yticks(rotation=45)
    for bs,a in zip(partitions,axes):

        a.barh(range(len(bs)),
               [b.universalHeight() for b in bs ],
               tick_label=[b.name for b in bs ])
    plot.show()

