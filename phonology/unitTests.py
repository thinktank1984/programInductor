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

if __name__ == "__main__":
    deleteInitial()
    spread()
