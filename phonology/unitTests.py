# -*- coding: utf-8 -*-

from supervised import SupervisedProblem
from problems import *
from matrix import *
from parseSPE import *

def spread():
    s = UnderlyingProblem(sevenProblems[1].data)
    assert s.applyRuleUsingSketch(parseRule("a > e/#CC*_"), Morph(u"kkaek"),0) == Morph(u"kkeek")
    assert s.applyRuleUsingSketch(parseRule("a > e/CC*_"), Morph(u"kkaek"),0) == Morph(u"kkeek")
    assert s.applyRuleUsingSketch(parseRule("a > e/#CC_"), Morph(u"kkaek"),0) == Morph(u"kkeek")
    assert s.applyRuleUsingSketch(parseRule("a > e/#C_"), Morph(u"kaek"),0) == Morph(u"keek")


def deleteInitial():
    s = UnderlyingProblem([[u"katigtde"]])
    assert  s.applyRuleUsingSketch(parseRule("C > 0/#_"), Morph(u"kat"),0) == Morph(u"at")
    assert  s.applyRuleUsingSketch(parseRule("C > 0/#_"), Morph(u"ekat"),0) == Morph(u"ekat")
    assert  s.applyRuleUsingSketch(parseRule("V > 0/#_"), Morph(u"ekat"),0) == Morph(u"kat")

def supervisedDeleteInitial():
    s = SupervisedProblem([(Morph("kat"),0,Morph("at"))])
    r = s.solve(1)[0]
    assert isinstance(r.structuralChange, EmptySpecification)
    s = SupervisedProblem([(Morph("kat"),0,Morph("at")),
                           (Morph("dat"),0,Morph("at")),
                           (Morph("fat"),0,Morph("at")),
                           (Morph("rat"),0,Morph("at")),
                           (Morph("iat"),0,Morph("iat")),
                           (Morph("oat"),0,Morph("oat"))])
    r = s.solve(1)[0]
    assert isinstance(r.structuralChange, EmptySpecification)
    assert isinstance(r.focus, FeatureMatrix)
    assert str(r.focus) == "[ -vowel ]"

def supervisedDuplicateSyllable():
    s = SupervisedProblem([(Morph("xa"),0,Morph("xaxa"))],
                          syllables = True)
    r = s.solve(1)[0]
    print r
    assert isinstance(r.focus, EmptySpecification)
    assert r.copyOffset == 1 and unicode(r.rightTriggers.specifications[0]) == u'σ'\
        or r.copyOffset == -1 and unicode(r.leftTriggers.specifications[0]) == u'σ'

if __name__ == "__main__":
    import sys
    A = sys.argv
    if len(A) > 1:
        for f in A[1:]: eval('%s()'%f)
    else:
        # import types as t
        # for n,f in globals().iteritems():
        #     if isinstance(f,t.FunctionType):
        #         print "Unit testing",n
        #         f()
        supervisedDuplicateSyllable()
        supervisedDeleteInitial()
        deleteInitial()
        spread()
