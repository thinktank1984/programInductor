# -*- coding: utf-8 -*-

from sketchSyntax import define, FunctionCall, Constant
from features import FeatureBank

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
    
    @staticmethod
    def parse(bank, output, variable):
        pattern = " %s = new ConstantPhoneme\(phoneme=phoneme_([0-9]+)_" % str(variable)
        m = re.search(pattern, output)
        if not m: raise Exception('Failure parsing ConstantPhoneme %s, pattern = %s'%(variable,pattern))
        return ConstantPhoneme(bank.phonemes[int(m.group(1))])
    def makeConstant(self, bank):
        return "new ConstantPhoneme(phoneme = phoneme_%d)" % bank.phoneme2index[self.p]

class EmptySpecification():
    def __init__(self): pass
    def __unicode__(self): return u"Ø"
    def __str__(self): return unicode(self).encode('utf-8')

    @staticmethod
    def parse(bank, output, variable):
        pattern = " %s = null;" % variable
        m = re.search(pattern, output)
        if not m: raise Exception('Failure parsing empty specification %s'%variable)
        return EmptySpecification()
    def makeConstant(self, bank):
        return "null"
    
class FeatureMatrix():
    def __init__(self, featuresAndPolarities): self.featuresAndPolarities = featuresAndPolarities
    def __str__(self):
        elements = [ ('+' if polarity else '-')+f for polarity,f in self.featuresAndPolarities ]
        return u"[ {} ]".format(u" ".join(elements))

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

class Guard():
    def __init__(self, side, endOfString, starred, specifications):
        self.side = side
        self.endOfString = endOfString
        self.starred = starred
        self.specifications = [ s for s in specifications if s != None ]
    def __str__(self): return unicode(self).encode('utf-8')
    def __unicode__(self):
        parts = []
        parts += map(unicode,self.specifications)
        if self.starred: parts[-2] += u'*'
        if self.endOfString: parts += [u'#']
        if self.side == 'L': parts.reverse()
        return u" ".join(parts)

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
    def __init__(self, focus, structuralChange, leftTriggers, rightTriggers, ending, copyOffset):
        self.copyOffset = copyOffset
        self.focus = focus
        self.structuralChange = structuralChange
        self.leftTriggers = leftTriggers
        self.rightTriggers = rightTriggers
        self.ending = ending

    def __str__(self): return unicode(self).encode('utf-8')
    def __unicode__(self):
        return u"{} ---> {} / {} _ {}".format(u'#' if self.ending else self.focus,
                                              self.copyOffset if self.copyOffset else self.structuralChange,
                                              self.leftTriggers,
                                              self.rightTriggers)
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
        pattern = 'Rule.*%s.* = new Rule\(focus=([a-zA-Z0-9_]+), structural_change=([a-zA-Z0-9_]+), left_trigger=([a-zA-Z0-9_]+), right_trigger=([a-zA-Z0-9_]+), ending=([01]), copyOffset=([\-01\(\)]+)\)' % str(variable)
        m = re.search(pattern, output)
        if not m:
            raise Exception('Failure parsing rule')
        focus = Specification.parse(bank, output, m.group(1))
        structuralChange = Specification.parse(bank, output, m.group(2))
        leftTrigger = Guard.parse(bank, output, m.group(3), 'L')
        rightTrigger = Guard.parse(bank, output, m.group(4), 'R')
        ending = m.group(5) == '1'
        offset = eval(m.group(6))
        return Rule(focus,
                    structuralChange,
                    leftTrigger,
                    rightTrigger,
                    ending,
                    offset)
    
