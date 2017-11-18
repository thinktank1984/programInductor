# -*- coding: utf-8 -*-

from sketchSyntax import define, FunctionCall, getGeneratorDefinition, globalConstant
from sketch import makeConstantWord, getGeneratorDefinition
from features import FeatureBank,featureMap,tokenize
from utilities import *

from random import choice
import re

class Morph():
    def __init__(self, phonemes):
        if isinstance(phonemes,unicode): phonemes = tokenize(phonemes)
        self.phonemes = phonemes
    def __unicode__(self):
        return u"/ {} /".format(u" ".join(self.phonemes))
    def __str__(self): return unicode(self).encode('utf-8')
    def __repr__(self): return str(self)
    # this interferes with parallel computation - probably because it messes up serialization
    # def __repr__(self): return unicode(self)
    def __len__(self): return len(self.phonemes)
    def __add__(self, other): return Morph(self.phonemes + other.phonemes)
    def __eq__(self, other):
        return str(self) == str(other)
    def __ne__(self, other):
        return str(self) != str(other)
    def __getitem__(self, sl):
        if isinstance(sl,int): return self.phonemes[sl]
        if isinstance(sl,slice):
            return Morph(self.phonemes[sl.start:sl.stop:sl.step])

    def mutate(self,bank):
        # remove a phoneme
        if len(self) > 0 and choice([True,False]):
            return Morph(randomlyRemoveOne(self.phonemes))
        # insert a phoneme
        p = choice(bank.phonemes)
        newPhonemes = list(self.phonemes)
        newPhonemes.insert(choice(range(len(self) + 1)),p)
        return Morph(newPhonemes)

    @staticmethod
    def sample():
        return define("Word", FunctionCall("unknown_word",[]), globalToHarnesses = True)

    def makeConstant(self, bank):
        return makeConstantWord(bank, "".join(self.phonemes))

    # Returns a variable that refers to a sketch object
    def makeDefinition(self, bank):
        return globalConstant("Word", self.makeConstant(bank))

    def fst(self,bank):
        return ''.join([bank.phoneme2fst(p) for p in self.phonemes ])

    @staticmethod
    def fromFST(bank,transducerOutput):
        return Morph([ bank.fst2phoneme(o) for o in transducerOutput ])

    @staticmethod
    def parse(bank, output, variable):
        if variable.definingFunction != None:
            # Search for the global definition, get the unique variable name it corresponds to, and parse that
            variable, output = getGeneratorDefinition(variable.definingFunction, output)
            return Morph.parse(bank, output, variable)
        
        pattern = 'Word.* %s.* = new Word\(l=([0-9]+)\);'%variable
        m = re.search(pattern, output)
        if not m: raise Exception('Could not find word %s in:\n%s'%(variable,output))

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
