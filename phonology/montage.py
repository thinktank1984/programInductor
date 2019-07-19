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
        print self.name, "Missing baseline?", any( b is None for b in baselines), "Missing full model?", self.universal is None

        self.Nadia = None # do not have a Nadia solution
        if self.alternation:
            if os.path.exists("CSV/"+self.problem.key+".output"):
                with open("CSV/"+self.problem.key+".output","r") as handle:
                    for ln in handle:
                        if "Successfully discovered rule" in ln:
                            self.Nadia = 1.
                            break
                        if "Could not discover rule" in ln:
                            self.Nadia = 0.
                            break
                assert self.Nadia is not None
        else:
            fn1 = "CSV/"+self.problem.key+".output"
            fn2 = "CSV/"+self.problem.key+"_cc0.output"
            if os.path.exists(fn1) and os.path.exists(fn2):
                for fn in [fn1,fn2]:
                    with open(fn,"r") as handle:
                        for ln in handle:
                            if "Successful" in ln:
                                self.Nadia = 1.
                                return
                            if "dictionary could not be made bigger" in ln:
                                self.Nadia = None
                                return
                self.Nadia = 0.
                
        #self.name = self.problem.languageName

    @property
    def alternation(self):
        return self.problem.parameters and "alternations" in self.problem.parameters

    @property
    def numberOfBars(self):
        if self.alternation: return 1
        return int(len(self.universal) > 0) + sum(b is not None for b in self.baselines)

    @property
    def language(self): return self.problem.languageName

    def universalTime(self):
        if not self.alternation and self.universal:
            print self.name, "solved in", min(r.solutionSequence[-1][1] for r in self.universal), "seconds"
        

    def universalHeight(self):
        if self.alternation: return 1.
        if len(self.universal) == 0: return 0.
        n = len(self.problem.data)
        if "Somali" in self.language and False:
            u = self.universal[0]
            print(n)
            print(u.finalFrontier.MAP())
            print(len(u.finalFrontier.underlyingForms))
        return float(max(len(u.finalFrontier.underlyingForms) for u in self.universal))/n

    def NadiaHeight(self):
        if self.Nadia is None: return 0.
        if self.Nadia == 0.: return 0.02
        if self.Nadia == 1.: return 1.
        assert False

    def baselineHeight(self, b):
        if self.alternation: return 1.
        assert not self.alternation
        b = self.baselines[b]
        if b is None: return 0.
        n = len(self.problem.data)
        return max(float(len(b.finalFrontier.underlyingForms))/n,0.02)

    def __str__(self):
        return "Bars(%s,%f)"%(self.name, self.universalHeight())

    def __repr__(self):
        return str(self)
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "Graphs the the system on all of the languages")
    parser.add_argument("--final","-f",action='store_true',default=False)
    arguments = parser.parse_args()

    baselinePath_1 = ["experimentOutputs/%s_CEGIS_disableClean=True_features=none.p"]
    baselinePath_2 = ["experimentOutputs/%s_CEGIS_disableClean=False_features=sophisticated_geometry=True.p"]
    universalPath = ["experimentOutputs/%s_incremental_disableClean=False_features=sophisticated_geometry=True_ug.p",
                     "experimentOutputs/%s_incremental_disableClean=False_features=sophisticated_geometry=True.p"]
    bars = []

    for name, problem in Problem.named.iteritems():
        if problem.supervised: continue
        
        if "Kevin" in name: continue

        # if name == "Odden_2.4_Tibetan":
        #     bl = "experimentOutputs/Odden_2.4_Tibetan_exact_disableClean=False_features=sophisticated.p"
        #     ul = "experimentOutputs/Odden_2.4_Tibetan_exact_disableClean=False_features=sophisticated_geometry=True.p"
        #     if os.path.exists(bl): bl = loadPickle(bl)
        #     else: bl = None
        #     bl_1 = bl
        #     bl_2 = bl
        #     if os.path.exists(ul): ul = [loadPickle(ul)]
        #     else: ul = []
        # el
        if problem.parameters and "alternations" in problem.parameters:
            if os.path.exists("experimentOutputs/alternation/%s.p"%name):
                p = loadPickle("experimentOutputs/alternation/%s.p"%name)
            else:
                p = None
            bl_1 = p
            bl_2 = p
            ul = [p]
            if p is None:
                print "Missing alternation",name
                
        else:
            bl_1 = None
            for b in baselinePath_1:
                if os.path.exists(b%name):
                    bl_1 = loadPickle(b%name)
                    print "Loaded",b%name
                    break
                else:
                    print "Failed to load",b%name
            bl_2 = None
            for b in baselinePath_2:
                if os.path.exists(b%name):
                    print "Loaded",b%name
                    bl_2 = loadPickle(b%name)
                    break
                else:
                    print "Failed to load",b%name
            if bl_2 is None: print "MISSINGCEGIS",name
            ul = []
            for u in universalPath:
                if os.path.exists(u%name):
                    ul.append(loadPickle(u%name))
        
        bars.append(Bars(problem,ul,bl_1,bl_2))

    # for b in bars:
    #     b.universalTime()
    # assert False

    bars.sort(key=lambda b: (not b.alternation, -b.universalHeight(), -(b.baselineHeight(1) - b.baselineHeight(0))))

    if arguments.final:
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
    number_of_baselines = 3
    for bs,a in zip(partitions,axes):
        bs.reverse()
        
        W = (1 - 0.2)/(number_of_baselines + 1)
        ys = np.arange((len(bs)))
        
        a.barh(ys,
               [b.universalHeight() for b in bs ],
               W,
               color='b'*len(bs))
        a.barh(ys + W,
               [b.baselineHeight(1) for b in bs ],
               W,
               color='g')
        a.barh(ys + W*2,
               [b.baselineHeight(0) for b in bs ],
               W,
               color='y')
        a.barh(ys + W*3,
               [b.NadiaHeight() for b in bs ],
               W,
               color='k')
        a.set(yticks=ys + 2*W,
              yticklabels=[b.name for b in bs ])

    if not arguments.final or True:
        plot.show()
    else:
        plot.savefig("/tmp/language_montage.png")

