from utilities import *
from rule import *
from features import *


from foma import *

class Solution():
    def __init__(self,rules = [],prefixes = [],suffixes = [],underlyingForms = [],adjustedCost = None):
        assert len(prefixes) == len(suffixes)
        self.rules = rules
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.underlyingForms = underlyingForms
        self.adjustedCost = adjustedCost

    def __str__(self):
        return "\n".join([ "rule: %s"%(str(r)) for r in self.rules ] +
                         [ "%s + stem + %s"%(str(self.prefixes[j]),str(self.suffixes[j]))
                           for j in range(len(self.prefixes)) ] +
                         (["underlying form: %s"%str(u)
                           for u in self.underlyingForms ]))

    def cost(self):
        return sum([ r.cost() for r in self.rules ] +
                   [ len(s) for s in (self.prefixes + self.suffixes + self.underlyingForms) ])

    def modelCost(self):
        return sum([ r.cost() for r in self.rules ] +
                   [ len(s) for s in (self.prefixes + self.suffixes) ])

    def depth(self): return len(self.rules)

    def showMorphologicalAnalysis(self):
        print "Morphological analysis:"
        for i in range(len(self.prefixes)):
            print "Inflection %d:\t"%i,
            print self.prefixes[i],
            print "+ stem +",
            print self.suffixes[i]

    def showRules(self):
        print "Phonological rules:"
        for r in self.rules: print r

    def phonologyTransducer(self,bank):
        return composedTransducer(bank, self.rules)

    def morphologyTransducers(self, bank):
        def makeTransducer(prefix, suffix):
            if len(prefix) > 0:
                t1 = '[..] -> %s || .#. _'%(' '.join([ bank.phoneme2fst(p) for p in prefix.phonemes ]))
                if getVerbosity() >= 5:
                    print "prefix regular expression", t1
                t1 = FST(t1)
            else:
                t1 = getIdentityFST()
            if len(suffix) > 0:
                t2 = '[..] -> %s ||  _ .#.'%(' '.join([ bank.phoneme2fst(p) for p in suffix.phonemes ]))
                if getVerbosity() >= 5:
                    print "suffix regular expression",t2
                t2 = FST(t2)
            else:
                t2 = getIdentityFST()
            return t1.compose(t2)

        return [ makeTransducer(prefix, suffix) for prefix, suffix in zip(self.prefixes, self.suffixes) ]

    def inflectionTransducers(self, bank):
        phonology = self.phonologyTransducer(bank)
        # return [ phonology.compose(m) for m in self.morphologyTransducers(bank) ]
        return [ m.compose(phonology) for m in self.morphologyTransducers(bank) ]

    def transduceUnderlyingForm(self, bank, surfaces):
        transducers = self.inflectionTransducers(bank)

        ur = invertParallelTransducers(transducers,
                                       [ ''.join([ bank.phoneme2fst(p) for p in tokenize(s) ]) for s in surfaces])
        candidates = []
        for u in ur:
            candidates.append(u)
            if len(candidates) > 10: break
            
        if candidates == []: return None
        bestCandidate = min(candidates, key = lambda c: len(c))
        return Morph([ bank.fst2phoneme(p) for p in bestCandidate ])
