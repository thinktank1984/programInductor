from problems import *
from textbook_problems import *

from utilities import *
import pickle
import os
import sys

import numpy as np

from matplotlib.lines import Line2D
import matplotlib.pyplot as plot

#NADIANAME = "Barke et al. 2019"
NADIANAME = "phonosynth (2019)"

class Bars():
    def __init__(self, problem, universal, fragment, *baselines):
        self.fragment = fragment
        self.problem = problem
        self.baselines = baselines
        self.universal = universal
        self.name = "%s (%s)"%(self.language, self.problem.source)
        print self.name, "Missing baseline?", any( b is None for b in baselines), "Missing full model?", self.universal is None

        self.Nadia = None # do not have a Nadia solution
        self.Nadia_cc0 = None # the column cost zero Nadia solution
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
            if os.path.exists(fn1):
                with open(fn1,"r") as handle:
                    content = handle.read()
                    if "Successful" in content: self.Nadia = 1.
                    else: self.Nadia = 0.
                    assert "could not be made bigger" not in content
            if os.path.exists(fn2):
                with open(fn2,"r") as handle:
                    content = handle.read()
                    if "Successful" in content: self.Nadia_cc0 = 1.
                    else: self.Nadia_cc0 = 0.
                    assert "could not be made bigger" not in content

    @property
    def alternation(self):
        return self.problem.parameters and "alternations" in self.problem.parameters

    @property
    def numberOfBars(self):
        return int(len(self.universal) > 0) + sum(b is not None for b in self.baselines)

    @property
    def language(self):
        if "Ukrainian" in self.problem.languageName: return "Ukrainian"
        return self.problem.languageName.replace(u" (Cuzco dialect)","")#self.problem.languageName

    def universalTime(self):
        if not self.alternation and self.universal:
            print self.name, "solved in", min(r.solutionSequence[-1][1] for r in self.universal), "seconds"
        

    def universalHeight(self):
        if self.alternation: return 1.
        if len(self.universal) == 0: return 0.
        n = len(self.problem.data)
        return float(max(len(u.finalFrontier.underlyingForms) for u in self.universal))/n

    def averageBaselineHeight(self):
        return (self.NadiaHeight() + sum(self.baselineHeight(b) for b in range(len(self.baselines)) ))/(len(self.baselines)+1)

    def NadiaHeight(self,cc0=None):
        if cc0 is None: return max(self.NadiaHeight(True), self.NadiaHeight(False))
        
        if cc0: Nadia = self.Nadia_cc0
        else: Nadia = self.Nadia
        if Nadia is None: return 0.
        if Nadia == 0.: return 0.02
        if Nadia == 1.: return 1.
        assert False

    def baselineHeight(self, b):
        if self.alternation:
            if b >= len(self.baselines) or self.baselines is None: return 0.
            if self.baselines[b] == "FAILURE": return 0.02
            else: return 1.
        b = self.baselines[b]
        if b is None: return 0.
        n = len(self.problem.data)
        return max(float(len(b.finalFrontier.underlyingForms))/n,0.02)

    def fragmentHeight(self):
        if self.alternation:
            return 0.
        b = self.fragment
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
    parser.add_argument("--universal","-u",action='store_true',default=False)
    parser.add_argument("--columns","-c",type=int,default=3)
    arguments = parser.parse_args()

    baselinePaths = ["experimentOutputs/%s_CEGIS_disableClean=False_features=sophisticated_geometry=True.p",
                     "experimentOutputs/%s_CEGIS_disableClean=False_features=simple.p",
                     "experimentOutputs/%s_CEGIS_disableClean=True_features=none.p"                     ]
    alternationBaselines = ["experimentOutputs/alternation/%s.p", # CEGIS, whatever
                            "experimentOutputs/alternation/%s_simple.p",
                            "experimentOutputs/alternation/%s_ablation.p"]
    universalPath = ["experimentOutputs/%s_incremental_disableClean=False_features=sophisticated_geometry=True.p"]
    fragmentPath = "experimentOutputs/%s_incremental_disableClean=False_features=sophisticated_geometry=True_ug.p"
    bars = []

    for name, problem in Problem.named.iteritems():
        if problem.supervised: continue
        
        if "Kevin" in name: continue

        baselines = []
        universals = []
        fragment = None
        if problem.parameters and "alternations" in problem.parameters:
            if os.path.exists("experimentOutputs/alternation/%s.p"%name):
                universals.append(loadPickle("experimentOutputs/alternation/%s.p"%name))
            else:
                universals.append(None)
                print "Missing alternation",name

            for pathTemplate in alternationBaselines:
                if os.path.exists(pathTemplate%name):
                    bl = loadPickle(pathTemplate%name)
                    if bl is None: bl = "FAILURE"
                else:
                    bl = None
                baselines.append(bl)
        else:
            for pathTemplate in baselinePaths:
                if os.path.exists(pathTemplate%name):
                    bl = loadPickle(pathTemplate%name)
                    print "Loaded", pathTemplate%name
                else:
                    bl = None
                    print "Missing baseline",pathTemplate%name
                baselines.append(bl)
            
            for u in universalPath:
                if os.path.exists(u%name):
                    universals.append(loadPickle(u%name))

            if os.path.exists(fragmentPath%name):
                fragment = loadPickle(fragmentPath%name)
        
        bars.append(Bars(problem,universals,fragment,*baselines))

    # for b in bars:
    #     b.universalTime()
    # assert False

    bars.sort(key=lambda b: (b.fragment is not None,not b.alternation, -b.universalHeight(), -(b.averageBaselineHeight())))

    if arguments.final:
        for n,b in enumerate(bars):
            if b.alternation and False: b.name = b.problem.languageName + "*"
            else:
                if sum(b.language == o.language for o in bars if not b.alternation ) > 1:
                    i = sum(b.language == o.language for o in bars[:n + 1]
                            if not b.alternation)
                    b.name = b.language # + " (" + "I"*i + ")"
                else:
                    b.name = b.language
            b.name = b.name.replace(" (Cuzco dialect)","")


    columns = arguments.columns
    if arguments.universal:
        bars = [b for b in bars if b.fragment is not None]
        for b in bars:
            print(b.name,b.fragmentHeight(),b.universalHeight())
        ys = np.arange(len(bars))
        W = (1 - 0.2)/2
        colors = [("before learning fragment grammar","#bc5090"),
                  ("with learned fragment grammar","#003f5c")]
        plot.bar(ys - W/2, [b.universalHeight() for b in bars],W,color=colors[0][1])
        plot.bar(ys + W/2, [b.fragmentHeight() for b in bars],W,color=colors[1][1])
        plot.gca().set(xticks=ys - W,
                       xticklabels=[b.name for b in bars ])
        plot.xticks(rotation=45)
        plot.ylabel("% problem solved")
        plot.gca().spines['right'].set_visible(False)
        plot.gca().spines['top'].set_visible(False)

        geometryAverage = sum([b.universalHeight() for b in bars])/len(bars)
        fragmentAverage = sum([b.fragmentHeight() for b in bars])/len(bars)
        print(geometryAverage,fragmentAverage,fragmentAverage/geometryAverage)


        
        plot.legend([Line2D([0],[0],color=c,lw=4)
                     for _,c in colors],
                    [n for n,_ in colors ],
                    ncol=2,
                    loc='lower center',
                    bbox_to_anchor=(0.5,-1))
        plot.show()
#         top=0.88,
# bottom=0.7,
# left=0.165,
# right=0.85,
# hspace=0.2,
# wspace=0.2
        sys.exit()
    ##003f5c
#58508d
#bc5090
#ff6361
#ffa600

        
    f, axes = plot.subplots(1,columns)
    # partition into columns
    partitions = partitionEvenly(bars,columns)
    #f.yticks(rotation=45)
    colors = [("ours (full)", "b"),
              ("ours (CEGIS)", "mediumslateblue"),
              ("ours (simple features)", "purple"),
              ("-representation", "teal"),
#              ("FG","cyan"),
              (NADIANAME, "gold")]
    number_of_baselines = len(colors) - 1
    colormap = dict(colors)
    for pi,(bs,a) in enumerate(zip(partitions,axes)):
        bs.reverse()
        
        W = (1 - 0.2)/(len(colors))
        ys = np.arange((len(bs)))
        
        a.barh(ys + W*number_of_baselines,
               [b.universalHeight() for b in bs ],
               W,
               color=colormap["ours (full)"])
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        
        for bi,(name,c) in enumerate(colors[1:-1]):
            a.barh(ys + W*(len(colors) - 2 - bi),
                   [b.fragmentHeight() if name == "FG" else b.baselineHeight(bi)
                    for b in bs ],
                   W,
                   color=c)

        a.barh(ys,
               [b.NadiaHeight() for b in bs ],
               W,
               color=colormap[NADIANAME])

        print "names",[b.name for b in bs ]

        a.set(yticks=ys + 2*W,
              yticklabels=[b.name for b in bs ])
        if pi == int(columns/2):
            a.set_xlabel('% data covered')

    

    f.legend([Line2D([0],[0],color=c,lw=4)
                 for _,c in colors],
                [n for n,_ in colors ],
                ncol=len(colors),
             loc='lower center')
    print len(bars),"data sets"
    print len({b.language for b in bars }),"distinct languages"


    if not arguments.final or True:
        plot.show()
    else:
        plot.savefig("/tmp/language_montage.png")

    print({b.language for b in bars })

