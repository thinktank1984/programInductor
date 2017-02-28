# DSL independent model generation code
# Focuses on generating and parsing the syntax of sketch/webppl programs

import re

class Expression:
    def __add__(self,o):
        if not isinstance(o,Expression): o = Constant(o)
        return Addition(self,o)
    def __sub__(self,o):
        if not isinstance(o,Expression): o = Constant(o)
        return Subtraction(self,o)
    def __eq__(self,o):
        if not isinstance(o,Expression):
            o = Constant(o)
        return Equals(self,o)
    def __gt__(self,o):
        if not isinstance(o,Expression): o = Constant(o)
        return LessThan(o,self)
    def __lt__(self,o):
        if not isinstance(o,Expression): o = Constant(o)
        return LessThan(self,o)
    def __str__(self): return self.web()
    def __radd__(self,o):
        if not isinstance(o,Expression): o = Constant(o)
        return Addition(self,o)

class FunctionCall(Expression):
    def __init__(self, f, arguments):
        self.f = f
        self.x = arguments
    def sketch(self):
        return str(self.f) + "(" + ", ".join([a.sketch() for a in self.x ]) + ")"
    def web(self):
        return str(self.f) + "(" + ", ".join([a.web() for a in self.x ]) + ")"

class Variable(Expression):
    def __init__(self,n): self.n = n
    def sketch(self): return self.n
    def web(self): return self.n

class Constant(Expression):
    def __init__(self,k): self.k = str(k)
    def sketch(self): return self.k
    def web(self): return self.k

class LessThan(Expression):
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def sketch(self): return "((%s) < (%s))" % (self.a.sketch(), self.b.sketch())
    def web(self): return "((%s) < (%s))" % (self.a.web(), self.b.web())

class Equals(Expression):
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def sketch(self): return "((%s) == (%s))" % (self.a.sketch(), self.b.sketch())
    def web(self): return "((%s) == (%s))" % (self.a.web(), self.b.web())

class Array(Expression):
    def __init__(self,elements): self.elements = elements
    def sketch(self):
        return "{%s}" % ", ".join([ e.sketch() for e in self.elements ])
    def web(self):
        return "[%s]" % ", ".join([ e.web() for e in self.elements ])

class Minimize():
    def __init__(self,n): self.n = n
    def sketch(self): return "minimize(%s);" % self.n.sketch()
    def web(self): return "factor( - (%s))" % self.n.web()
class Maximize():
    def __init__(self,n): self.n = n
    def sketch(self): return "minimize(10 - (%s));" % self.n.sketch()
    def web(self): return "factor(%s)" % self.n.web()

class Definition():
    def __init__(self, ty, name, value):
        self.ty = ty
        self.name = name
        self.value = value
    def sketch(self):
        return "%s %s = %s;" % (self.ty,self.name,self.value.sketch())
    def web(self):
        return "var %s = %s" % (self.name,self.value.web())

class Conditional(Expression):
    def __init__(self,t,y,n):
        self.t = t
        self.y = y
        self.n = n
    def sketch(self):
        return "((%s) ? %s : %s)" % (self.t.sketch(),self.y.sketch(),self.n.sketch())
    def web(self):
        return "((%s) ? %s : %s)" % (self.t.web(),self.y.web(),self.n.web())

class And(Expression):
    def __init__(self,clauses):
        self.clauses = clauses
    def sketch(self):
        return "(%s)" % (" && ".join([c.sketch() for c in self.clauses ]))
    def web(self):
        return "(%s)" % (" && ".join([c.web() for c in self.clauses ]))

class Or(Expression):
    def __init__(self,clauses):
        self.clauses = clauses
    def sketch(self):
        return "(%s)" % (" || ".join([c.sketch() for c in self.clauses ]))
    def web(self):
        return "(%s)" % (" || ".join([c.web() for c in self.clauses ]))

class Assertion():
    def __init__(self,p): self.p = p
    def sketch(self): return "assert %s;" % self.p.sketch()
    def web(self):
        return "factor((%s) ? 0 : -Infinity)" % (self.p.web())

class QuantifiedAssertion():
    def __init__(self,p,i):
        self.p = p
        self.i = i
    def sketch(self): return "if (__ASSERTIONCOUNT__ == %d) assert %s;" % (self.i, str(self.p))
    def web(self):
        return "factor((%s) ? 0 : -Infinity)" % (self.p.web())

    
class Addition(Expression):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def sketch(self): return "((%s) + (%s))" % (self.x.sketch(),self.y.sketch())
    def web(self): return "((%s) + (%s))" % (self.x.web(),self.y.web())
class Subtraction(Expression):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def sketch(self): return "((%s) - (%s))" % (self.x.sketch(),self.y.sketch())
    def web(self): return "((%s) - (%s))" % (self.x.web(),self.y.web())

class Model():
    def __init__(self):
        self.flipCounter = 0
        self.integerCounter = 0
        self.definitionCounter = 0
        self.statements = []
        self.quantifiedConditions = 0
    def flip(self, p = 0.5):
        self.flipCounter += 1
        return Variable("__FLIP__%d"%self.flipCounter)
    def unknownInteger(self):
        self.integerCounter += 1
        return Variable("__INTEGER__%d"%self.integerCounter)
    def define(self, ty, value):
        name = "__DEFINITION__%d"%self.definitionCounter
        self.definitionCounter += 1
        self.statements.append(Definition(ty, name, value))
        return Variable(name)
    def condition(self, predicate):
        self.statements.append(Assertion(predicate))
    def quantifiedCondition(self, predicate):
        self.quantifiedConditions += 1
        self.statements.append(QuantifiedAssertion(predicate,self.quantifiedConditions))
    def minimize(self, expression):
        self.statements.append(Minimize(expression))
    def maximize(self, expression):
        self.statements.append(Maximize(expression))
    def removeSoftConstraints(self):
        '''removes all minimize and maximize statements that have been previously added.'''
        self.statements = [ s for s in self.statements if not (s is Maximize or s is Minimize) ]

    def sketch(self):
        h = ""
            
        for f in range(self.flipCounter):
            h += "bit __FLIP__%d = ??;\n" % (f + 1)
        for f in range(self.integerCounter):
            h += "int __INTEGER__%d = ??;\n" % (f + 1)

        h += "\nharness void main(int __ASSERTIONCOUNT__) {\n"
        for a in self.statements:
            h += "\t" + a.sketch() + "\n"
        h += "}\n"
        return h
    def web(self):
        h = "var posterior = function() {\n"
            
        for f in range(self.flipCounter):
            h += "\tvar __FLIP__%d = flip()\n" % (f + 1)

        for a in self.statements:
            h += "\t" + a.web() + "\n"

        h += "\treturn ["
        h += ", ".join([ "__FLIP__%d" % (f + 1) for f in range(self.flipCounter) ])
        h += "]\n}"
        h += "\nInfer({method: 'enumerate'},posterior)"
        return h
    @staticmethod
    def Global():
        global currentModel
        currentModel = Model()
        return currentModel
    
currentModel = None


def flip(p = 0.5):
    global currentModel
    return currentModel.flip(p)
def unknownInteger():
    global currentModel
    return currentModel.unknownInteger()
def ite(condition,yes,no):
    return Conditional(condition,yes,no)

def define(ty, value):
    return currentModel.define(ty, value)

def condition(predicate):
    currentModel.condition(predicate)

def quantifiedCondition(predicate):
    currentModel.quantifiedCondition(predicate)

def minimize(expression):
    currentModel.minimize(expression)

def maximize(expression):
    currentModel.maximize(expression)

def removeSoftConstraints():
    currentModel.removeSoftConstraints()

def sketchImplementation(name):
    def namedImplementation(f):
        def wrapper(*args, **kwargs):
            return FunctionCall(name, args)
        return wrapper
    return namedImplementation

def makeSketchSkeleton():
    return currentModel.sketch()
def makeWebSkeleton():
    return currentModel.web()

def parseFlip(output, variable):
    pattern = 'void glblInit_%s__ANONYMOUS_'%str(variable)
    ls = output.splitlines()
    for l in range(len(ls)):
        if pattern in ls[l]:
            return " = 1;" in ls[l + 2]
    print "Could not find",variable
    print pattern
    print output
    return None
def parseInteger(output, variable):
    pattern = 'void glblInit_%s__ANONYMOUS_'%str(variable)
    ls = output.splitlines()
    for l in range(len(ls)):
        if pattern in ls[l]:
            m = re.search(" = ([0-9]+);", ls[l + 2])
            if not m:
                raise Exception('error parsing integer')
            return int(m.group(1))
    print "Could not find",variable
    print pattern
    print output
    return None
