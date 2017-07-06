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

    def mutate(self,bank):
        # mutate a phoneme
        if random() < 0.3:
            newPrefixes = list(self.prefixes)
            newSuffixes = list(self.suffixes)
            for _ in range(sampleGeometric(0.7) + 1):
                i = choice(range(len(self.prefixes)))
                if choice([True,False]): # mutate a prefix
                    newPrefixes[i] = newPrefixes[i].mutate(bank)
                else:
                    newSuffixes[i] = newSuffixes[i].mutate(bank)
            return Solution(self.rules,newPrefixes,newSuffixes)
                
        # mutate a rule
        if random() < 0.5:
            r = choice(self.rules)
            newRules = [ (r.mutate(bank) if q == r else q) for q in self.rules ]
            return Solution(newRules,self.prefixes,self.suffixes)
        # reorder the rules
        if len(self.rules) > 1 and random() < 0.3:
            i = choice(range(len(self.rules)))
            j = choice([ k for k in range(len(self.rules)) if k != i ])
            newRules = [ self.rules[i if k == j else (j if k == i else k)]
                         for k in range(len(self.rules)) ]
            return Solution(newRules,self.prefixes,self.suffixes)
        # delete a rule
        if len(self.rules) > 1 and random() < 0.3:
            newRules = randomlyRemoveOne(self.rules)
            return Solution(newRules,self.prefixes,self.suffixes)
        # insert a rule
        newRules = list(self.rules)
        newRules.insert(choice(range(len(self.rules)+1)), EMPTYRULE.mutate(bank).mutate(bank).mutate(bank).mutate(bank))
        return Solution(newRules,self.prefixes,self.suffixes)
        
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
        if not hasattr(self,'savedInflectionTransducers'):
            phonology = self.phonologyTransducer(bank)
            self.savedInflectionTransducers = [ m.compose(phonology) for m in self.morphologyTransducers(bank) ]
        return self.savedInflectionTransducers

    def clearTransducers(self):
        if hasattr(self,'savedInflectionTransducers'): del self.savedInflectionTransducers

    def transduceUnderlyingForm(self, bank, surfaces):
        try:
            transducers = self.inflectionTransducers(bank)
        except InvalidRule: return None

        applicableTransducersAndSurfaces = [(t,s) for (t,s) in zip(transducers, surfaces) if s != None ]
        transducers = [t for t,_ in applicableTransducersAndSurfaces ]
        surfaces = [s for _,s in applicableTransducersAndSurfaces ]

        ur = invertParallelTransducers(transducers,
                                       [ ''.join([ bank.phoneme2fst(p) for p in tokenize(s) ]) for s in surfaces])
        candidates = []
        for u in ur:
            candidates.append(u)
            if len(candidates) > 10: break
            
        if candidates == []: return None
        bestCandidate = min(candidates, key = lambda c: len(c))
        return Morph([ bank.fst2phoneme(p) for p in bestCandidate ])
