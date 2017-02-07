# -*- coding: utf-8 -*-

from sketchSyntax import define, FunctionCall, Constant
from features import FeatureBank,featureMap
from morph import Morph

import re

class Specification():
    def __init__(self): pass
    
    @staticmethod
    def parse(bank, output, variable):
        try:
            return FeatureMatrix.parse(bank, output, variable)
        except:
            try:
                return EmptySpecification.parse(bank, output, variable)
            except:
                return ConstantPhoneme.parse(bank, output, variable)
        
class ConstantPhoneme(Specification):
    def __init__(self, p): self.p = p
    def __unicode__(self): return self.p
    def __str__(self): return unicode(self).encode('utf-8')
    def doesNothing(self): return False
    def cost(self): return 2
    def skeleton(self): return "K"
    
    @staticmethod
    def parse(bank, output, variable):
        pattern = " %s = new ConstantPhoneme\(phoneme=phoneme_([0-9]+)_" % str(variable)
        m = re.search(pattern, output)
        if not m: raise Exception('Failure parsing ConstantPhoneme %s, pattern = %s'%(variable,pattern))
        return ConstantPhoneme(bank.phonemes[int(m.group(1))])
    def makeConstant(self, bank):
        return "new ConstantPhoneme(phoneme = phoneme_%d)" % bank.phoneme2index[self.p]

    def matches(self, test):
        return featureMap[self.p] == test
    def apply(self, test):
        return featureMap[self.p]

class EmptySpecification():
    def __init__(self): pass
    def __unicode__(self): return u"Ø"
    def __str__(self): return unicode(self).encode('utf-8')

    def doesNothing(self): return False
    def skeleton(self): return "0"
    def cost(self): return 2

    @staticmethod
    def parse(bank, output, variable):
        pattern = " %s = null;" % variable
        m = re.search(pattern, output)
        if not m: raise Exception('Failure parsing empty specification %s'%variable)
        return EmptySpecification()
    def makeConstant(self, bank):
        return "null"

    def matches(self, test):
        raise Exception('Can not try to match with deletion rule')
    def apply(self, test):
        raise Exception('cannot apply deletion rule')
    
class FeatureMatrix():
    def __init__(self, featuresAndPolarities): self.featuresAndPolarities = featuresAndPolarities
    def __str__(self):
        elements = [ ('+' if polarity else '-')+f for polarity,f in self.featuresAndPolarities ]
        return u"[ {} ]".format(u" ".join(elements))

    def doesNothing(self):
        return len(self.featuresAndPolarities) == 0

    def cost(self):
        return 1 + len(self.featuresAndPolarities)

    def skeleton(self):
        if self.featuresAndPolarities == []: return "[ ]"
        else: return "[ +/-features ]"

    @staticmethod
    def parse(bank, output, variable):
        pattern = " %s = new Vector\(mask={([01,]+)}, preference={([01,]+)}"%variable
        m = re.search(pattern, output)
        if not m: raise Exception('Failure parsing Vector %s'%variable)
        preference = [ int(x) for x in m.group(2).split(",") ]
        mask = [ int(x) for x in m.group(1).split(",") ]
        fs = [ (preference[f] == 1, bank.features[f]) for f in range(len(bank.features)) if mask[f] ]
        return FeatureMatrix(fs)
    def makeConstant(self, bank):
        mask = [0]*len(bank.features)
        preference = [0]*len(bank.features)
        for polarity,feature in self.featuresAndPolarities:
            mask[bank.feature2index[feature]] = 1
            preference[bank.feature2index[feature]] = 1 if polarity else 0
        mask = " ,".join(map(str, mask))
        preference = " ,".join(map(str, preference))
        return "new Vector(mask = {%s}, preference = {%s})" % (mask, preference)

    def implicitMatrix(self):
        def exclude(fs, m):
            for j in range(len(fs)):
                if (True,fs[j]) in m:
                    fs += [(False,fs[k]) for k in range(len(fs)) if k != j ]
            return list(set(fs))
        return exclude(['front','central','back'],
                       exclude(['high','middle','low'], self.featuresAndPolarities))

    def matches(self, test):
        for p,f in self.featuresAndPolarities:
            if p:
                if not (f in test): return False
            else:
                if f in test: return False
        return True
    def apply(self, test):
        for p,f in self.implicitMatrix():
            if p:
                test += [f]
            else:
                test = [_f for _f in test if not _f == f ]
        return list(set(test))

class Guard():
    def __init__(self, side, endOfString, starred, specifications):
        self.side = side
        self.endOfString = endOfString
        self.starred = starred
        self.specifications = [ s for s in specifications if s != None ]

    def doesNothing(self):
        return not self.endOfString and len(self.specifications) == 0

    def cost(self):
        return int(self.starred) + int(self.endOfString) + sum([ s.cost() for s in self.specifications ])
    
    def __str__(self): return unicode(self).encode('utf-8')
    def __unicode__(self):
        parts = []
        parts += map(unicode,self.specifications)
        if self.starred: parts[-2] += u'*'
        if self.endOfString: parts += [u'#']
        if self.side == 'L': parts.reverse()
        return u" ".join(parts)
    def skeleton(self):
        parts = []
        parts += map(lambda spec: spec.skeleton(),self.specifications)
        if self.starred: parts[-2] += '*'
        if self.endOfString: parts += ['#']
        if self.side == 'L': parts.reverse()
        return " ".join(parts)

    @staticmethod
    def parse(bank, output, variable, side):
        pattern = " %s = new Guard\(endOfString=([01]), starred=([01]), spec=([a-zA-Z0-9_]+), spec2=([a-zA-Z0-9_]+)"%variable
        m = re.search(pattern, output)
        if not m: raise Exception('Could not parse guard %s using pattern %s'%(variable,pattern))

        endOfString = m.group(1) == '1'
        starred = m.group(2) == '1'
        spec = None if m.group(3) == 'null' else Specification.parse(bank, output, m.group(3))
        spec2 = None if m.group(4) == 'null' else Specification.parse(bank, output, m.group(4))
        return Guard(side, endOfString, starred, [spec,spec2])
    def makeConstant(self, bank):
        if len(self.specifications) == 2:
            [spec1,spec2] = self.specifications
            spec1 = self.specifications[0].makeConstant(bank)
            spec2 = self.specifications[1].makeConstant(bank)
        elif len(self.specifications) == 1:
            spec1 = self.specifications[0].makeConstant(bank)
            spec2 = "null"
        else:
            spec1 = "null"
            spec2 = "null"
        
        return "new Guard(endOfString = %d, starred = %d, spec = %s, spec2 = %s)" % (1 if self.endOfString else 0,
                                                                                     1 if self.starred else 0,
                                                                                     spec1,
                                                                                     spec2)

class Rule():
    def __init__(self, focus, structuralChange, leftTriggers, rightTriggers):
        self.focus = focus
        self.structuralChange = structuralChange
        self.leftTriggers = leftTriggers
        self.rightTriggers = rightTriggers

    def cost(self):
        return self.focus.cost() + self.structuralChange.cost() + self.leftTriggers.cost() + self.rightTriggers.cost()
    def alternationCost(self):
        return self.leftTriggers.cost() + self.rightTriggers.cost()   

    def doesNothing(self):
        '''Does this rule do nothing? Equivalently is it [  ] ---> [  ] /  _ '''
        return self.leftTriggers.doesNothing() and self.rightTriggers.doesNothing() and self.focus.doesNothing() and self.structuralChange.doesNothing()

    def __str__(self): return unicode(self).encode('utf-8')
    def __unicode__(self):
        return u"{} ---> {} / {} _ {}".format(self.focus,
                                              self.structuralChange,
                                              self.leftTriggers,
                                              self.rightTriggers)

    def skeleton(self):
        return "{} ---> {} / {} _ {}".format(self.focus.skeleton(),
                                              self.structuralChange.skeleton(),
                                              self.leftTriggers.skeleton(),
                                              self.rightTriggers.skeleton())

    # Produces sketch object
    def makeConstant(self, bank):
        return Constant("new Rule(focus = %s, structural_change = %s, left_trigger = %s, right_trigger = %s)" % (self.focus.makeConstant(bank),
                                                                                                                 self.structuralChange.makeConstant(bank),
                                                                                                                 self.leftTriggers.makeConstant(bank),
                                                                                                                 self.rightTriggers.makeConstant(bank)))
                                         
    # Produces sketch object
    @staticmethod
    def sample():
        return define("Rule", FunctionCall("unknown_rule",[]))

    # Produces a rule object from a sketch output
    @staticmethod
    def parse(bank, output, variable):
        pattern = 'Rule.*%s.* = new Rule\(focus=([a-zA-Z0-9_]+), structural_change=([a-zA-Z0-9_]+), left_trigger=([a-zA-Z0-9_]+), right_trigger=([a-zA-Z0-9_]+)\)' % str(variable)
        m = re.search(pattern, output)
        if not m:
            raise Exception('Failure parsing rule')
        focus = Specification.parse(bank, output, m.group(1))
        structuralChange = Specification.parse(bank, output, m.group(2))
        leftTrigger = Guard.parse(bank, output, m.group(3), 'L')
        rightTrigger = Guard.parse(bank, output, m.group(4), 'R')
        return Rule(focus, structuralChange, leftTrigger, rightTrigger)

    def apply(self, u):
        # First off we convert u to feature matrices if it is a morph
        if isinstance(u,Morph):
            u = [ featureMap[p] for p in u.phonemes ]
            
        middleOkay = [ self.focus.matches(p) for p in u ]

        leftOkay = []
        accepting = False
        # check to see if the left matches
        for j in range(len(u)):
            okay = True
            if self.leftTriggers.starred:
                if j == 0:
                    okay = False
                    accepting = self.leftTriggers.specifications[1].matches(u[0])
                else:
                    okay = accepting
                    accepting = ((not self.leftTriggers.endOfString) and self.leftTriggers.specifications[1].matches(u[j])) or (accepting and self.leftTriggers.specifications[0].matches(u[j]))
            elif self.leftTriggers.endOfString and len(self.leftTriggers.specifications) == 0: # #_
                okay = j == 0
            elif self.leftTriggers.specifications != []:
                okay = j > 0 and self.leftTriggers.specifications[0].matches(u[j - 1])
                if len(self.leftTriggers.specifications) == 2: # (#?)gg_
                    okay = okay and j > 1 and self.leftTriggers.specifications[1].matches(u[j - 2])

                if self.leftTriggers.endOfString:
                    if len(self.leftTriggers.specifications) == 1:
                        okay = okay and j == 1
                    else:
                        okay = okay and j == 2
            leftOkay.append(okay)

        # do the same thing on the right but walk backwards
        rightOkay = [None]*len(u)
        accepting = False
        for j in range(len(u) - 1, -1, -1):
            okay = True
            if self.rightTriggers.starred: # _g*g(#?)
                if j == len(u) - 1:
                    okay = False
                    accepting = self.rightTriggers.specifications[1].matches(u[len(u) - 1])
                else:
                    okay = accepting
                    accepting = ((not self.rightTriggers.endOfString) and self.rightTriggers.specifications[1].matches(u[j])) or (accepting and self.rightTriggers.specifications[0].matches(u[j]))
            elif self.rightTriggers.endOfString and self.rightTriggers.specifications == []: # _#
                okay = j == len(u) - 1
            elif self.rightTriggers.specifications != []: # _gg?#?
                okay = j < len(u) - 1 and self.rightTriggers.specifications[0].matches(u[j + 1])
                if len(self.rightTriggers.specifications) == 2: # _gg#?
                    okay = okay and j < len(u) - 2 and self.rightTriggers.specifications[1].matches(u[j + 2])

                if self.rightTriggers.endOfString:
                    if len(self.rightTriggers.specifications) == 1: # _g#
                        okay = okay and j == len(u) - 2
                    else: # _gg#
                        okay = okay and j == len(u) - 3
            rightOkay[j] = okay
                
            
        triggered = [ middleOkay[j] and rightOkay[j] and leftOkay[j] for j in range(len(u)) ]

        change = self.structuralChange
        if isinstance(change,EmptySpecification):
            return [ u[j] for j in range(len(u)) if not triggered[j] ]
        else:
            return [ (u[j] if not triggered[j] else change.apply(u[j])) for j in range(len(u)) ]
