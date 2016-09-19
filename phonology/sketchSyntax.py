# DSL independent sketch generation code
# Focuses on generating and parsing the syntax of sketch programs

class FunctionCall():
    def __init__(self, f, arguments):
        self.f = f
        self.x = arguments
    def __str__(self):
        return str(self.f) + "(" + ", ".join([str(a) for a in self.x ]) + ")"

class Variable():
    def __init__(self,n): self.n = n
    def __str__(self): return self.n

class Minimize():
    def __init__(self,n): self.n = n
    def __str__(self): return "minimize(%s);" % self.n

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

def minimize(expression):
    statements.append(Minimize(expression))

def sketchImplementation(name):
    def namedImplementation(f):
        def wrapper(*args, **kwargs):
            return FunctionCall(name, args)
        return wrapper
    return namedImplementation

def makeSketchSkeleton():
    global flipCounter
    h = ""
    
    for f in range(flipCounter):
        h += "bit __FLIP__%d = ??;\n" % (f + 1)

    h += "\nharness void main() {\n"
    for a in statements:
        h += "\t" + str(a) + "\n"
    h += "}\n"
    return h
