# -*- coding: utf-8 -*-

from utilities import *
from Marcus import *
from supervised import SupervisedProblem
from problems import *
from matrix import *
from parseSPE import *
from incremental import *
from features import *


TESTS = []
def test(f):
    TESTS.append(f)
    return f

@test
def features():
    for p in featureMap:
        for q in featureMap:
            if p == q: continue
            assert set(featureMap[p]) != set(featureMap[q]), "Expected %s and %s to have different features"%(p,q)

@test
def editSequences():
    assert len(everyEditSequence([0,1],[1,2],allowSubsumption = True)) > \
        len(everyEditSequence([0,1],[1,2],allowSubsumption = False))
    bounded = everyEditSequence([0,1],[1,2],allowSubsumption = False,maximumLength = 2)
    unbounded = everyEditSequence([0,1],[1,2],allowSubsumption = False,maximumLength = None)
    assert len(unbounded) > len(bounded)
    assert len(bounded) == 1
    assert len(everyEditSequence([0,1,2],[1,2],allowSubsumption = False)) == 16
    
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
def supervisedOptionalEndOfString():
    # Intended rule: m > n/_{#,t}
    s = SupervisedProblem([(Morph("man"),0,Morph("man")),
                           (Morph("kamt"),0,Morph("kant")),
                           (Morph("kamk"),0,Morph("kamk")),
                           (Morph("kamd"),0,Morph("kamd")),
                           (Morph("om"),0,Morph("on")),
                           (Morph("mta"),0,Morph("nta")),
                           (Morph("iatm"),0,Morph("iatn"))])
    r = s.solve(1)[0]
    assert unicode(r.rightTriggers) == u'{#,t}'
    assert len(r.leftTriggers.specifications) == 0
    assert not r.leftTriggers.endOfString
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
@test
def induceBoundary():
    inventory = FeatureBank([u"utestadz"])
    Model.Global()
    r = Rule.sample()
    condition(FunctionCall("rule_uses_boundary",[r]))
    # stem = ute
    # suffix = st
    prefix = Morph([]).makeConstant(inventory)
    stem = Morph(u"ute").makeConstant(inventory)
    suffix = Morph(u"st").makeConstant(inventory)
    x = concatenate3(prefix, stem, suffix)
    y = Morph(u"utezt").makeConstant(inventory)
    prediction = applyRules([r,r,r], x, wordLength(prefix) + wordLength(stem), 6)
    auxiliaryCondition(wordEqual(prediction, y))

    minimize(ruleCost(r))

    output = solveSketch(inventory)
    g = Rule.parse(inventory,output,r).leftTriggers.specifications
    assert len(g) == 1
    assert isinstance(g[0],BoundarySpecification)
    
@test
def suffixBoundary():
    data = sevenProblems[1].data[:3]
    s = IncrementalSolver(sevenProblems[1].data,2).restrict(data)
    solution = parseSolution(''' + stem + 
 + stem + am
 + stem + ov^yi
 + stem + i
 + stem + ov^yi
C > [+palletized] / _ i ;; i is the only thing in the data which is [+high -back]
o > e / [+palletized] + _ ;; i is the only thing that is [+vowel +high -back]. "vowel fronting"
[ -glide -vowel ] ---> [ -palletized ] /  _ e
''')
    s.fixedMorphology = solution
    new = s.sketchChangeToSolution(solution, [solution.rules[0],None,solution.rules[2]], allTheData = data)
    assert new != None, "Should be able to incrementally change to accommodate an example"
    assert new.cost() <= solution.cost(), "Should have found an optimal solution"
    for d in data:
        assert s.verify(solution, [Morph(x) if x != None else None
                                   for x in d]), "Could not verify ground truth solution"
        assert s.verify(new, [Morph(x) if x != None else None
                              for x in d]), "Could not verify learned solution"

    
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
