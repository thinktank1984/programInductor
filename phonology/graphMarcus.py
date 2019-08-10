from Marcus import *
from fragmentGrammar import *
from utilities import *

from collections import defaultdict

import os
import random

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plot

BANK = None 

transductionCashFile = ".MarcusTransductionTable.p"

TRANSDUCTIONTABLE = {}
def cachedTransduction(s, word):
    global TRANSDUCTIONTABLE, BANK
    key = (s.withoutStems(),
           word)
    if key in TRANSDUCTIONTABLE:
        return TRANSDUCTIONTABLE[key]
    TRANSDUCTIONTABLE[key] = s.transduceUnderlyingForm(BANK,(Morph(word),))
    return TRANSDUCTIONTABLE[key]

def posteriorPredictiveLikelihood(posterior, testWord):
    b = FeatureBank([ s
                      for ss in posterior[0].underlyingForms.keys()
                      for s in ss ] + [Morph(testWord), u"-"])
    
    # compute posterior probabilities for each element in posterior
    logPosterior = []
    for solution in posterior:
        getEmptyFragmentGrammar().numberOfFeatures = len(b.features)
        getEmptyFragmentGrammar().numberOfPhonemes = len(b.phonemes)
        lp = -sum(len(stem) * math.log(len(b.phonemes))
                  for stem in solution.prefixes + solution.suffixes + solution.underlyingForms.values() ) + \
                      sum(getEmptyFragmentGrammar().ruleLogLikelihood(r)[0]
                       for r in solution.rules)/10
        logPosterior.append(lp)

    z = lseList(logPosterior)
    logPosterior = [lp - z for lp in logPosterior ]

    if True:
        print "Posterior (test word = %s)"%testWord
        for s,lp in sorted(zip(posterior, logPosterior),
                           key=lambda(s,p):-p):
            print lp
            print s
            print

    stems = [cachedTransduction(s, testWord) for s in posterior]

    logMarginal = float('-inf')
    for lp, stem in zip(logPosterior, stems):
        if stem is None: continue

        logMarginal = lse(logMarginal, lp - len(stem)*math.log(len(b.phonemes)))

    return logMarginal    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Analyze and plot Marcus patterns on held out data')
    pairings = ["%s,%s"%(x,y)
                for x in ["aba","aab","abb","abx","aax","Chinese"]
                for y in ["aba","aab","abb","abx","aax","Chinese"]
                if x != y]
    parser.add_argument('testCases',
                        choices = pairings,
                        type = str,
                        nargs='+')
    parser.add_argument('-n','--number', default = 4, type = int)
    parser.add_argument('-s','--samples', default=1, type=int)
    parser.add_argument('--sigmoid', default=False, type=float,
                        help="Plot sigmoid(log-odds/T), --sigmoid T ")
    parser.add_argument('-j','--jitter', default=0., type=float)
    parser.add_argument('--export', default = None, type = str)

    random.seed(0)

    if os.path.exists(transductionCashFile):
        print "Loading transductions from",transductionCashFile
        TRANSDUCTIONTABLE = loadPickle(transductionCashFile)
    
    arguments = parser.parse_args()

    arguments.testCases = [tuple(tc.split(","))
                           for tc in arguments.testCases ]
    
    withSyllables = {} # map from (training distribution, n_examples) to [solution]
    withoutSyllables = {} # map from (training distribution, n_examples) to [solution]
    consistentExamples = {} # map from (consistent, inconsistent) distribution to consistent test examples
    inconsistentExamples = {} # map from (consistent, inconsistent) to inconsistent test examples
    jobs = set() # {(solution, word)}
    everyWord = set() # set of every word ever
    for consistent, inconsistent in arguments.testCases:
        allTrainingExamples = set()
        X = None
        for n in range(1,arguments.number+1):
            withSyllables[consistent,n] = loadPickle("paretoFrontier/%s%d.p"%(consistent,n)).values()
            withoutSyllables[consistent,n] = loadPickle("paretoFrontier/%s%d_noSyllable.p"%(consistent,n)).values()
        
            # Control for training data
            surfaces1 = set(withSyllables[consistent,n][0].underlyingForms.keys())
            surfaces2 = set(withoutSyllables[consistent,n][0].underlyingForms.keys())
            for surface in surfaces1|surfaces2:
                allTrainingExamples.add(surface[0])
            assert len(surfaces1^surfaces2) == 0

            # Figure out what X is
            if consistent in {'abx','aax'}:
                X = u"".join(withSyllables[consistent,n][0].underlyingForms.keys()[0][0].phonemes[-2:])
            elif consistent == 'axa':
                X = u"".join(withSyllables[consistent,n][0].underlyingForms.keys()[0][0].phonemes[2:][:2])
        print consistent, inconsistent, "\tX=",X
    
        sampling = {'aba': sampleABA,
                    'abb': sampleABB,
                    'abx': lambda ne: sampleABX(ne,X=X),
                    'aax': lambda ne: sampleAAX(ne,X=X),
                    'axa': lambda ne: sampleAXA(ne,X=X),
                    'aab': sampleAAB,
                    }    
        random.seed(0)
        testConsistent = []
        while len(testConsistent) < arguments.samples:
            w = Morph(sampling[consistent](1)[0])
            if w not in allTrainingExamples: testConsistent.append(w)

        testInconsistent = []
        while len(testInconsistent) < arguments.samples:
            w = Morph(sampling[inconsistent](1)[0])
            if w not in allTrainingExamples: testInconsistent.append(w)

        for n in range(1,arguments.number+1):
            for s in withSyllables[consistent,n] + withoutSyllables[consistent,n]:
                for surfaces in s.underlyingForms.keys():
                    for surface in surfaces: everyWord.add(surface)
                for test in testConsistent + testInconsistent:
                    everyWord.add(Morph(test))
                    jobs.add((s.withoutStems(), test))

        consistentExamples[consistent,inconsistent] = testConsistent
        inconsistentExamples[consistent,inconsistent] = testInconsistent

    BANK = FeatureBank([u"-"] + list(everyWord))

    # Batched likelihood calculation in parallel
    transductions = \
                    lightweightParallelMap(numberOfCPUs(), 
                                           lambda (s,w): (s,w,s.transduceUnderlyingForm(BANK,(w,))),
                                           list(jobs - set(TRANSDUCTIONTABLE.keys())))
    for solution, word, transduction in transductions:
        TRANSDUCTIONTABLE[(solution, word)] = transduction

    if len(transductions) > 0:
        print "Updating cached transductions in",transductionCashFile
        temporaryFile = makeTemporaryFile(".p")
        dumpPickle(TRANSDUCTIONTABLE, temporaryFile)
        os.system("mv %s %s"%(temporaryFile, transductionCashFile))
    

    plot.figure()
    xs = range(0,arguments.number+1)
    COLORS = ["r","g","b","cyan"]

    for color, (consistent, inconsistent) in zip(COLORS,arguments.testCases):
        for syllables in [withSyllables, withoutSyllables]:
            ys = [0.5] if sigmoid else [0]
            deviations = [0]
            for n in range(1,arguments.number+1):
                # with syllables after n examples
                consistentLikelihoods = \
                 [posteriorPredictiveLikelihood(syllables[consistent,n], e)
                 for e in consistentExamples[consistent, inconsistent] ]
                inconsistentLikelihoods = \
                 [posteriorPredictiveLikelihood(syllables[consistent,n], e)
                 for e in inconsistentExamples[consistent, inconsistent] ]
                y = average(consistentLikelihoods) - average(inconsistentLikelihoods)
                if arguments.sigmoid:
                    y = sigmoid(y/arguments.sigmoid)
                    s = standardDeviation([sigmoid((c - i)/arguments.sigmoid)
                                           for c in consistentLikelihoods
                                           for i in inconsistentLikelihoods ])
                else:
                    s = standardDeviation([c - i
                                           for c in consistentLikelihoods
                                           for i in inconsistentLikelihoods ])

                ys.append(y)

                deviations.append(s)
            jxs = [x + arguments.jitter*(2*(random.random() - 0.5))
                   for x in xs ]
            plot.errorbar(jxs,ys,yerr=deviations,color=color,
                          ls='-' if syllables is withSyllables else '--')

    #plot.ylim(bottom=0.)
    plot.xlabel("# training examples")
    plot.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if arguments.sigmoid:
        plot.ylabel(r"$\sigma(\log\frac{\mathrm{P}(\mathrm{consistent}|\mathrm{train})}{\mathrm{P}(\mathrm{inconsistent}|\mathrm{train})})$")
    else:
        plot.ylabel(r"$\log\frac{\mathrm{P}(\mathrm{consistent}|\mathrm{train})}{\mathrm{P}(\mathrm{inconsistent}|\mathrm{train})}$")
    plot.legend([Line2D([0],[0],color=c,lw=2)
                 for c,_ in zip(COLORS,arguments.testCases)] + \
                [Line2D([0],[0],color='k',lw=2),
                 Line2D([0],[0],color='k',lw=2,ls='--')],
                ["train %s, test %s (consistent) / %s (inconsistent)"%(c,c,i)
                 for c,i in arguments.testCases ] + \
                ["w/ syllables", "w/o syllables"],
                ncol=1,
                loc='best')
    plot.show()
