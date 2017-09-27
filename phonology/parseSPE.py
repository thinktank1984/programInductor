# -*- coding: utf-8 -*-

from solution import Solution
from morph import Morph

import re
from features import *
from rule import *

'''
Why is there no good parser combination library for Python...
A parser is a function from strings to a stream of tuples of (unconsumed suffix, value)
'''

def eatWhiteSpace(s):
    j = 0
    while j < len(s) and s[j].isspace(): j += 1
    yield (s[j:],None)
def constantParser(k,v = None):
    def p(s):
        if s.startswith(k):
            yield (s[len(k):],v)
    return p
def whitespaceDelimited(p):
    return concatenate(eatWhiteSpace,
                       concatenate(p,eatWhiteSpace,lambda v1,v2: v1),
                       lambda v1,v2: v2)

noneCombiner = lambda v1,v2:  None
def concatenate(p,q,combiner = noneCombiner):
    def newParser(s):
        for suffix, first in p(s):
            for newSuffix, second in q(suffix):
                yield (newSuffix, combiner(first, second))
    return newParser

def alternation(*alternatives):
    def newParser(s):
        for a in alternatives:
            for x in a(s): yield x
    return newParser

def repeat(p):
    def newParser(s):
        generator = p(s)
        haveYieldedSomething = False

        while True:
            try:
                (suffix, value) = generator.next()
                for finalSuffix,values in newParser(suffix):
                    yield (finalSuffix,[value] + values)                
                    haveYieldedSomething = True
            except StopIteration:
                if not haveYieldedSomething: yield (s,[])
                break
            
    return newParser

def mapParserOutput(p,f):
    def newParser(s):
        for suffix, value in p(s): yield (suffix,f(value))
    return newParser

def optional(p):
    return alternation(constantParser(""), p)


def whitespaceDelimitedSequence(*things):
    if len(things) < 1:
        return constantParser('',[])
    return concatenate(whitespaceDelimited(things[0]), whitespaceDelimitedSequence(*things[1:]),
                       lambda v1,v2: [v1] + v2)

def runParser(p,s):
    for suffix, result in p(s):
        if len(suffix) == 0: return result
    assert False

featureParser = alternation(*[ constantParser(p + f, (p == '+',f)) 
                               for f in set([ f for fs in featureMap.values() for f in fs ])
                               for p in ['-','+'] ])
whitespaceFeatureParser = whitespaceDelimited(featureParser)
featuresParser = repeat(whitespaceFeatureParser)
matrixParser = mapParserOutput(concatenate(concatenate(constantParser('['), featuresParser, lambda v1,v2: v2),
                                           constantParser(']'),
                                           lambda v1,v2: v1),
                               lambda fp: FeatureMatrix(fp))
phonemeParser = alternation(*[ constantParser(k,ConstantPhoneme(k)) for k in featureMap ])
specificationParser = alternation(matrixParser,phonemeParser)
guardSpecificationParser = concatenate(specificationParser,
                                       optional(whitespaceDelimited(constantParser('*','*'))),
                                       lambda v1,v2: (v1,v2))
nullParser = alternation(constantParser("0",EmptySpecification()),
                         constantParser(u"Ø",EmptySpecification()))

focusChangeParser = alternation(*([ constantParser(str(n),n) for n in [-2,-1,1,2] ] + [specificationParser,nullParser]))

rightGuardParser = whitespaceDelimitedSequence(repeat(whitespaceDelimited(guardSpecificationParser)),
                                              optional(constantParser('#','#')))
leftGuardParser = whitespaceDelimitedSequence(optional(constantParser('#','#')),
                                              repeat(whitespaceDelimited(guardSpecificationParser)))

arrowParser = alternation(concatenate(repeat(constantParser('-')),constantParser('>')),
                          constantParser(u'⟶'))
ruleParser = whitespaceDelimitedSequence(focusChangeParser,
                                         arrowParser,
                                         focusChangeParser,
                                         constantParser('/'),
                                         leftGuardParser,
                                         constantParser('_'),
                                         rightGuardParser)

def parseRule(s):
    p = runParser(ruleParser,s)
    if s == None: return None
    [focus,_,change,_,l,_,r] = p

    l = Guard(endOfString = '#' == l[0],
              specifications = reversed([ s for s,_ in l[1] ]),
              starred = any([ s == '*' for _,s in l[1] ]),
              side = 'L')
    r = Guard(endOfString = '#' == r[1],
              specifications = [s for s,_ in r[0] ],
              starred = any([s == '*' for _,s in r[0] ]),
              side = 'R')

    copyOffset = 0
    if focus in [-2,-1,1,2]:
        copyOffset = focus
        focus = FeatureMatrix([])
    if change in [-2,-1,1,2]:
        copyOffset = change
        change = FeatureMatrix([])

    return Rule(focus, change, l,r,copyOffset)

def parseSolution(s):
    lines = [ x.strip() for x in s.split('\n') ]
    prefixes = []
    suffixes = []
    rules = []
    for l in lines:
        if 'stem' in l:
            [prefix,_,suffix] = l.split('+')
            prefixes.append(Morph(tokenize(prefix)))
            suffixes.append(Morph(tokenize(suffix)))
        else:
            rules.append(parseRule(l))
    return Solution(rules, prefixes, suffixes)

if __name__ == '__main__':
    print parseRule('0 > -2 / #[-vowel][]* _ e #').pretty()
    print parseSolution(u''' + stem + 
 + stem + ə
    [-sonorant] > [-voice] / _ #
    [+stop +voice] > [+fricative] / [+sonorant -nasal] _ [+sonorant]''')
