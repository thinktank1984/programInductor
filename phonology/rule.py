from sketchSyntax import define, FunctionCall
from features import FeatureBank


from random import random
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
    def __str__(self): return self.p

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
        return "[ %s ]" % (" ".join(elements))

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
    def __init__(self, side, endOfString, specification):
        self.side = side
        self.endOfString = endOfString
        self.specification = specification
    def __str__(self):
        parts = []
        if self.endOfString: parts = ['#']
        if self.specification: parts.append(str(self.specification))
        if self.side == 'R': parts.reverse()
        return " ".join(parts)

    @staticmethod
    def parse(bank, output, variable, side):
        pattern = " %s = new Guard\(endOfString=([01]), spec=([a-zA-Z0-9_]+)"%variable
        m = re.search(pattern, output)
        if not m: raise Exception('Could not parse guard %s using pattern %s'%(variable,pattern))

        endOfString = m.group(1) == '1'
        spec = None if m.group(2) == 'null' else Specification.parse(bank, output, m.group(2))
        return Guard(side, endOfString, spec)

class Rule():
    def __init__(self, focus = [], structuralChange = [], leftTriggers = [], rightTriggers = []):
        self.focus = focus
        self.structuralChange = structuralChange
        self.leftTriggers = leftTriggers
        self.rightTriggers = rightTriggers
    
    def __str__(self):
        return "%s ---> %s / %s _ %s" % (str(self.focus),
                                         str(self.structuralChange),
                                         str(self.leftTriggers),
                                         str(self.rightTriggers))
                                         
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
