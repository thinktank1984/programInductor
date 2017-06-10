

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
