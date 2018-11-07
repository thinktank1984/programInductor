from problems import MATRIXPROBLEMS, alternationProblems

from utilities import *
import pickle
import os
import sys

import matplotlib.pyplot as plot
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "Graphs the output of curriculum.py")
    parser.add_argument("startingIndex",
                        type=int,
                        help="Which problem to start out solving. 0-indexed")
    parser.add_argument("endingIndex",
                        type=int)
    parser.add_argument("--empirical",
                        default=False, action='store_true')
    parser.add_argument("--ground",
                        default=False, action='store_true')
    parser.add_argument("--none",
                        default=False, action='store_true')
    
    arguments = parser.parse_args()
    assert arguments.empirical or arguments.ground or arguments.none

    def getPickleDirectory(ug):
        if ug == "ground":
            pickleDirectory = "frontierPickles/groundUniversal"
        elif ug == "none":
            pickleDirectory = "frontierPickles/noUniversal"
        elif ug == "empiricalUniversal":
            pickleDirectory = "frontierPickles/empiricalUniversal"
        else: assert False
        return pickleDirectory

    universals = []
    if arguments.empirical: universals.append("empirical")
    if arguments.ground: universals.append("ground")
    if arguments.none: universals.append("none")

    labeledUniversal = universals[len(universals)/2]

    color = {None: 'b',
             "empirical": 'r',
             "ground": 'g',
             "none": 'k'}

    covers = []
    for ap in alternationProblems:
        language = ap.languageName
        if ' (' in language:
            language = language[:language.index(' (')]
        covers.append((None,language + '*', 1.))

    for j in range(arguments.startingIndex, arguments.endingIndex+1):
        for u in universals:
            fn = '%s/matrix_%d.p'%(getPickleDirectory(u),j)
            try:
                with open(fn,'rb') as handle:
                    solution = pickle.load(handle)
                    covered = len(solution.underlyingForms)
            except IOError:
                print "WARNING: Could not load",fn
                covered = 0
            total = len(MATRIXPROBLEMS[j].data)
            language = MATRIXPROBLEMS[j].languageName

            # Tibetan counting is weird
            if language == 'Tibetan': covered = total

            covers.append((u,language,covered/float(total)))
            print fn, language, covered/float(total)
    plot.figure(figsize=(5,10))
    plot.yticks(rotation=45)
    plot.barh(range(len(covers)),
             [c for _,l,c in covers ],
             tick_label=[l if u is None or u == labeledUniversal else ""
                         for u,l,c in covers ],
              color=[color[u] for u,l,c in covers ])
    plot.xlabel('% data covered by rules')

    # if arguments.ug == "ground":
    #     plot.title("Learned UG (supervised)")
    # elif arguments.ug == "none":
    #     plot.title("No UG")
    # elif arguments.ug == "empirical":
    #     plot.title("Learned UG (unsupervised)")
    plot.tight_layout()
    plot.savefig("/tmp/visualize.png")

