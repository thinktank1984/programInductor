# -*- coding: utf-8 -*-

from sketchSyntax import define, FunctionCall
from features import FeatureBank

import re

class Specification():
    def __init__(self): pass
    
    @staticmethod
    def parse(bank, output, variable):
        try:
            return FeatureMatrix.parse(bank, output, variable)
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

class Guard():
    def __init__(self, side, endOfString, specifications):
        self.side = side
        self.endOfString = endOfString
        self.specifications = [ s for s in specifications if s != None ]
    def __str__(self): return unicode(self).encode('utf-8')
    def __unicode__(self):
        parts = []
        if self.endOfString: parts = [u'#']
        parts += map(unicode,self.specifications)
        if self.side == 'R': parts.reverse()
        return u" ".join(parts)

    @staticmethod
    def parse(bank, output, variable, side):
        pattern = " %s = new Guard\(endOfString=([01]), spec=([a-zA-Z0-9_]+), spec2=([a-zA-Z0-9_]+)"%variable
        m = re.search(pattern, output)
        if not m: raise Exception('Could not parse guard %s using pattern %s'%(variable,pattern))

        endOfString = m.group(1) == '1'
        spec = None if m.group(2) == 'null' else Specification.parse(bank, output, m.group(2))
        spec2 = None if m.group(3) == 'null' else Specification.parse(bank, output, m.group(3))
        return Guard(side, endOfString, [spec,spec2])

class Rule():
    def __init__(self, focus = [], structuralChange = [], leftTriggers = [], rightTriggers = []):
        self.focus = focus
        self.structuralChange = structuralChange
        self.leftTriggers = leftTriggers
        self.rightTriggers = rightTriggers

    def __str__(self): return unicode(self).encode('utf-8')
    def __unicode__(self):
        return u"{} ---> {} / {} _ {}".format(self.focus,
                                              self.structuralChange,
                                              self.leftTriggers,
                                              self.rightTriggers)
                                         
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
