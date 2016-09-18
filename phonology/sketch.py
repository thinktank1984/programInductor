import re
from features import featureVectorMap,tokenize,FEATURELIST
from rule import Rule

from random import random
import os

class FunctionCall():
    def __init__(self, f, arguments):
        self.f = f
        self.x = arguments
    def __str__(self):
        return str(self.f) + "(" + ", ".join([str(a) for a in self.x ]) + ")"

class Variable():
    def __init__(self,n): self.n = n
    def __str__(self): return self.n

class Definition():
    def __init__(self, ty, name, value):
        self.ty = ty
        self.name = name
        self.value = value
    def __str__(self):
        return "%s %s = %s;" % (self.ty,self.name,str(self.value))

class Conditional():
    def __init__(self,t,y,n):
        self.t = t
        self.y = y
        self.n = n
    def __str__(self):
        return "((%s) ? %s : %s)" % (self.t,self.y,self.n)

class Assertion():
    def __init__(self,p): self.p = p
    def __str__(self): return "assert %s;" % str(self.p)

flipCounter = 0
def flip(p = 0.5):
    global flipCounter
    flipCounter += 1

    return Variable("__FLIP__%d"%flipCounter)

def ite(condition,yes,no):
    return Conditional(condition,yes,no)

definitionCounter = 0
def define(ty, value):
    global definitionCounter
    name = "__DEFINITION__%d"%definitionCounter
    definitionCounter += 1
    statements.append(Definition(ty, name, value))
    return Variable(name)

statements = []
def condition(predicate):
    statements.append(Assertion(predicate))

def sketchImplementation(name):
    def namedImplementation(f):
        def wrapper(*args, **kwargs):
            return FunctionCall(name, args)
        return wrapper
    return namedImplementation

@sketchImplementation("applyRule")
def applyRule(rule,i):
    pass
@sketchImplementation("make_word")
def makeWord(features): return features
@sketchImplementation("word_equal")
def wordEqual(w1,w2):
    pass
@sketchImplementation("phonological_rule")
def phonologicalRule(i): pass
@sketchImplementation("unknown_rule_improved")
def sampleRule(): pass
@sketchImplementation("apply_rule")
def applyRule(r,w): pass

def makeConstantMatrix(matrix):
    return "{ %s }" % ", ".join([ "{ %s }" % ", ".join(map(str,fs)) for fs in matrix ])

def makeConstantWord(word):
    matrix = [featureVectorMap[t] for t in tokenize(word) ]
    return makeConstantWordOfMatrix(matrix)
def makeConstantWordOfMatrix(matrix):
    return makeWord(makeConstantMatrix(matrix))


def makeSketch():
    global flipCounter
    h = ""
    h += "#define NUMBEROFFEATURES %d\n" % len(FEATURELIST)
    h += "#include \"common.skh\"\n"

    for f in range(flipCounter):
        h += "bit __FLIP__%d = ??;\n" % (f + 1)

    h += "\nharness void main() {\n"
    for a in statements:
        h += "\t" + str(a) + "\n"
    h += "}\n"
    return h

def solveSketch():
    source = makeSketch()
    with open("test.sk","w") as f:
        f.write(source)
    outputFile = "solver_output/%f" % random()
    os.system("sketch --bnd-unroll-amnt 50 test.sk > %s" % outputFile)
    return open(outputFile,'r').read()


def decodeStructure(preference,mask):
    return [ (preference[f] == 1,FEATURELIST[f]) for f in range(len(FEATURELIST)) if mask[f] == 1]

def parseRule(output):
    structures = {}
    for specification in ['focus','structural_change','left_trigger','right_trigger']:
        pattern = 'Rule.*%s=new Specification\(mask={([01,]+)}, preference={([01,]+)}\)' % specification
        m = re.search(pattern, output)
        if not m:
            print "Could not find the following pattern:"
            print pattern
            return None
        s = decodeStructure([int(x) for x in m.group(2).split(",") ],
                            [int(x) for x in m.group(1).split(",") ])
        structures[specification] = s
    return Rule(structures['focus'],
                structures['structural_change'],
                [structures['left_trigger']],
                [structures['right_trigger']])
