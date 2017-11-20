# -*- coding: utf-8 -*-

from utilities import *
from Marcus import *
from supervised import SupervisedProblem
from problems import *
from matrix import *
from parseSPE import *
from incremental import *

TESTS = []
def test(f):
    TESTS.append(f)
    return f

@test
def editSequences():
    assert len(everyEditSequence([0,1],[1,2],allowSubsumption = True)) > \
        len(everyEditSequence([0,1],[1,2],allowSubsumption = False))
    bounded = everyEditSequence([0,1],[1,2],allowSubsumption = False,maximumLength = 2)
    unbounded = everyEditSequence([0,1],[1,2],allowSubsumption = False,maximumLength = None)
    assert len(unbounded) > len(bounded)
    assert len(bounded) == 1
    assert len(everyEditSequence([0,1,2],[1,2],allowSubsumption = False)) == 19
    
@test
def spread():
    s = UnderlyingProblem(sevenProblems[1].data)
    assert s.applyRuleUsingSketch(parseRule("a > e/#CC*_"), Morph(u"kkaek"),0) == Morph(u"kkeek")
    assert s.applyRuleUsingSketch(parseRule("a > e/CC*_"), Morph(u"kkaek"),0) == Morph(u"kkeek")
    assert s.applyRuleUsingSketch(parseRule("a > e/#CC_"), Morph(u"kkaek"),0) == Morph(u"kkeek")
    assert s.applyRuleUsingSketch(parseRule("a > e/#C_"), Morph(u"kaek"),0) == Morph(u"keek")
@test
def deleteInitial():
    s = UnderlyingProblem([[u"katigtde"]])
    assert  s.applyRuleUsingSketch(parseRule("C > 0/#_"), Morph(u"kat"),0) == Morph(u"at")
    assert  s.applyRuleUsingSketch(parseRule("C > 0/#_"), Morph(u"ekat"),0) == Morph(u"ekat")
    assert  s.applyRuleUsingSketch(parseRule("V > 0/#_"), Morph(u"ekat"),0) == Morph(u"kat")
@test
def supervisedDeleteInitial():
    s = SupervisedProblem([(Morph("kat"),0,Morph("at"))])
    r = s.solve(1)[0]
    assert isinstance(r.structuralChange, EmptySpecification)
    s = SupervisedProblem([(Morph("kat"),0,Morph("at")),
                           (Morph("dat"),0,Morph("at")),
                           (Morph("fat"),0,Morph("at")),
                           (Morph("rat"),0,Morph("at")),
                           (Morph("iat"),0,Morph("iat")),
                           (Morph(u"ɩat"),0,Morph(u"ɩat")),
                           (Morph("oat"),0,Morph("oat"))])
    r = s.solve(1)[0]
    assert isinstance(r.structuralChange, EmptySpecification)
    assert isinstance(r.focus, FeatureMatrix)
    assert str(r.focus) == "[ -vowel ]"
@test
def supervisedDuplicateSyllable():
    s = SupervisedProblem([(Morph("xa"),0,Morph("xaxa"))],
                          syllables = True)
    r = s.solve(1)[0]
    assert isinstance(r.focus, EmptySpecification)
    assert r.copyOffset == 1 and unicode(r.rightTriggers.specifications[0]) == u'σ'\
        or r.copyOffset == -1 and unicode(r.leftTriggers.specifications[0]) == u'σ'
@test
def testMarcus():
    s = UnderlyingProblem([ (w,) for w in sampleABB(6) ],
                          useSyllables = True).sketchJointSolution(1,canAddNewRules = True)
    assert len(s.rules) == 1
    assert any([ unicode(spec) == u'σ'
                 for spec in s.rules[0].rightTriggers.specifications + s.rules[0].leftTriggers.specifications ])
    assert all([ len(u) == 4 for u in s.underlyingForms ])
    assert s.rules[0].copyOffset != 0

    
if __name__ == "__main__":
    import sys
    A = sys.argv
    if len(A) > 1:
        for f in A[1:]:
            print " [+] Running test",f
            eval('%s()'%f)
    else:
        for f in TESTS:
            print " [+] Running test",f.__name__
            f()
            print 
