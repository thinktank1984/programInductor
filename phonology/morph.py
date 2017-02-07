# -*- coding: utf-8 -*-

from sketchSyntax import define, FunctionCall
from sketch import makeConstantWord
from features import FeatureBank,featureMap

import re

class Morph():
    def __init__(self, phonemes):
        self.phonemes = phonemes
    def __unicode__(self):
        return u"/ {} /".format(u" ".join(self.phonemes))
    def __str__(self): return unicode(self).encode('utf-8')
    def __len__(self): return len(self.phonemes)
    def __add__(self, other): return Morph(self.phonemes + other.phonemes)
    def __eq__(self, other):
        return str(self) == str(other)

    @staticmethod
    def sample():
        return define("Word", FunctionCall("unknown_word",[]))

    def makeConstant(self, bank):
        return makeConstantWord(bank, "".join(self.phonemes))

    @staticmethod
    def parse(bank, output, variable):
        pattern = 'Word.* %s.* = new Word\(l=([0-9]+)\);'%variable
        m = re.search(pattern, output)
        if not m: raise Exception('Could not find word %s'%variable)

        l = int(m.group(1))
        phones = []
        for p in range(l):
            pattern = '%s.*\.s\[%d\] = phoneme_([0-9]+)_'%(variable,p)
            m = re.search(pattern, output)
            if not m:
                print output
                print pattern
                raise Exception('Could not find %dth phoneme of %s'%(p,variable))
            phones.append(bank.phonemes[int(m.group(1))])
        return Morph(phones)

    @staticmethod
    def fromMatrix(m):
        def process(p):
            for s in featureMap:
                if set(featureMap[s]) == set(p): return s
            raise Exception('could not find a phoneme for the matrix: %s'%str(p))
        return Morph(map(process,m))
