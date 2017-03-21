# -*- coding: utf-8 -*-

from rule import *
from time import time
from math import log
from utilities import *
import cProfile

class MatchFailure(Exception):
    pass

class Fragment():
    def __init__(self, name, children, matcher = None, logPrior = 0.0):
        self.name = name
        self.children = children
        self.matcher = matcher
        self.logPrior = logPrior
    def __unicode__(self): return self.name
    def __str__(self): return unicode(self).encode('utf-8')
    def __eq__(self,other): return self.name == other.name
    def __hash__(self):
        return hash(str(self))

    def match(self,program): return self.matcher(program)


def programSubexpressions(program):
    '''Yields the sequence of tuples of (ty,expression)'''
    if isinstance(program, Rule):
        yield ('RULE',program)
        for x in programSubexpressions(program.focus): yield x
        for x in programSubexpressions(program.structuralChange): yield x
        for x in programSubexpressions(program.leftTriggers): yield x
        for x in programSubexpressions(program.rightTriggers): yield x
    elif isinstance(program, Guard):
        yield ('GUARD', program)
        for x in programSubexpressions(program.specifications): yield x
    elif isinstance(program, FeatureMatrix):
        yield ('SPECIFICATION', program)
       
    
logVariablePrior = -1.6

def abstractRuleFragments(p,q):
    if p.copyOffset != 0 or q.copyOffset != 0:
        raise Exception('abstractRuleFragments: copy offsets not yet supported')

    fragments = []
    if unicode(p) == unicode(q):
        fragments.append(Fragment(unicode(p),[],makeConstantMatcher(unicode(p)), -p.cost()))

    def makeMatcher(focus, change, l, r):
        return lambda test: focus.match(test.focus) + change.match(test.structuralChange) + l.match(test.leftTriggers) + r.match(test.rightTriggers)

    fragments += [
        Fragment(unicode(Rule(focus.name,change.name,l.name,r.name,0)),
                 focus.children + change.children + l.children + r.children,
                 makeMatcher(focus, change, l, r),
                 focus.logPrior + change.logPrior + l.logPrior + r.logPrior)
        for focus in abstractFcFragments(p.focus,q.focus)
        for change in abstractFcFragments(p.structuralChange,q.structuralChange)
        for l in abstractGuardFragments(p.leftTriggers, q.leftTriggers)
        for r in abstractGuardFragments(p.rightTriggers,q.rightTriggers)
    ]
    return fragments

def makeConstantMatcher(const):
    def m(test):
        if unicode(test) == const:
            return []
        else:
            raise MatchFailure()
    return m
variableMatcher = lambda test: [test]
def typeMatcher(ty):
    def m(test):
        if isinstance(test,ty):
            return [test]
        else:
            raise MatchFailure()
    return m

def abstractFcFragments(p,q): # focus/change, which are special specifications because they can be empty
    fragments = [Fragment(u"FC",["FC"], variableMatcher, logVariablePrior)]
    return fragments + abstractSpecificationFragments(p,q)

def abstractSpecificationFragments(p,q):
    fragments = []
    if unicode(p) == unicode(q):
        fragments.append(Fragment(unicode(p),[],makeConstantMatcher(unicode(p)),-p.cost()))
    else:
        fragments.append(Fragment(u"SPECIFICATION",["SPECIFICATION"], variableMatcher, logVariablePrior))

    if isinstance(p,ConstantPhoneme) and isinstance(q,ConstantPhoneme):
        fragments.append(Fragment(u"CONSTANT",["CONSTANT"], typeMatcher(ConstantPhoneme), logVariablePrior))
    elif isinstance(p,FeatureMatrix) and isinstance(q,FeatureMatrix):
        fragments.append(Fragment(u"MATRIX",["MATRIX"], typeMatcher(FeatureMatrix), logVariablePrior))

    return fragments

def abstractGuardFragments(p,q):    
    # end of string: do we have it?
    ending = u"(#?)"
    endingChildren = ['ENDING']
    endingMatcher = lambda test: [test.endOfString]
    if p.endOfString == q.endOfString:
        ending = p.endOfString
        endingChildren = []
        def qm(test):
            if test.endOfString == p.endOfString: return []
            raise MatchFailure()
        endingMatcher = qm

    fragments = [Fragment(u"GUARD",["GUARD"], variableMatcher, logVariablePrior)]

    def makeString(specifications, starred):
        parts = list(specifications)
        if starred: parts[-2] += u'*'
        if ending == True: parts += [u'#']
        if isinstance(ending, unicode): parts += [ending]
        return u" ".join(parts)

    # they have to look sufficiently similar in order to unify
    if p.starred != q.starred or len(p.specifications) != len(q.specifications):
        return fragments

    if len(p.specifications) == 0:
        def matcher0(test):
            if len(test.specifications) != 0: raise MatchFailure()
            return endingMatcher(test)
        fragments += [ Fragment(makeString([], p.starred),
                                endingChildren,
                                matcher0,
                                0.0) ]
    elif len(p.specifications) == 1:
        def matcher1(s):
            def inner1(test):
                if len(test.specifications) != 1: raise MatchFailure()
                return endingMatcher(test) + s.match(test.specifications[0])
            return inner1
        fragments += [
            Fragment(makeString([specification.name], p.starred),
                     endingChildren + specification.children,
                     matcher1(specification),
                     specification.logPrior)
            for specification in abstractSpecificationFragments(p.specifications[0], q.specifications[0]) ]
    elif len(p.specifications) == 2:
        def matcher2(s1,s2):
            def inner2(test):
                if len(test.specifications) != 2: raise MatchFailure()
                return endingMatcher(test) + s1.match(test.specifications[0]) + s2.match(test.specifications[1])
            return inner2
        fragments += [
            Fragment(makeString([s1.name,s2.name], p.starred),
                     endingChildren + s1.children + s2.children,
                     matcher2(s1,s2),
                     -s1.logPrior - s2.logPrior)
            for s1 in abstractSpecificationFragments(p.specifications[0], q.specifications[0])
            for s2 in abstractSpecificationFragments(p.specifications[1], q.specifications[1]) ]
    else:
        raise Exception('abstractGuardFragments: too many specifications')

    return fragments


def proposeFragments(problems, verbose = False):
    ruleSets = []
    for problem in problems:
        # problem should be a list of solutions
        # each solution should be a list of rules
        ruleSets.append(set([ r for s in problem for r in s ]))

    abstractFragments = {
        'RULE': abstractRuleFragments,
        'GUARD': abstractGuardFragments,
        'SPECIFICATION': abstractSpecificationFragments
    }

    startTime = time()
    fragments = {} # map from type to a set of fragments
    for i in range(len(ruleSets) - 1):
        for j in range(i+1, len(ruleSets)):
            for p in ruleSets[i]:
                for q in ruleSets[j]:
                    for pt,pf in programSubexpressions(p):
                        fragments[pt] = fragments.get(pt,set([]))
                        for qt,qf in programSubexpressions(q):
                            if pt != qt: continue
                            # the extra condition here is to avoid fragments like "GUARD -> GUARD"
                            fragments[pt] = fragments[pt] | set([ f for f in abstractFragments[pt](pf,qf) if str(f) != pt ])

    totalNumberOfFragments = sum([len(v) for v in fragments.values() ])
    print "Discovered %d unique fragments in %f seconds"%(totalNumberOfFragments,time() - startTime)
    if verbose:
        for ty in fragments:
            print "Fragments of type",ty
            for f in fragments[ty]:
                print f
            print ""

    return [ (t, f) for t in fragments for f in fragments[t]  ]

def induceGrammar(problems, maximumGrammarSize = 2):
    fragments = proposeFragments(problems)

    def problemLikelihood(problem, grammar):
        return max([ sum([ grammar.ruleLogLikelihood(r) for r in s ]) for s in problem ])
    def grammarLikelihood(grammar):
        return sum([ problemLikelihood(p, grammar) for p in problems ])

    print "Empty grammar likelihood:",grammarLikelihood(FragmentGrammar())

    chosenFragments = []
    
    while len(chosenFragments) < maximumGrammarSize:
        bestFragment = None
        bestPosterior = float('-inf')
        for f in [ x for x in fragments if not x in chosenFragments]:
            l = grammarLikelihood(FragmentGrammar(chosenFragments + [f]))
            posterior = l+f[1].logPrior
            print "Considering %s %s\n\t%f + %f = %f"%(f[0],f[1],l,f[1].logPrior,posterior)
            if posterior > bestPosterior:
                bestPosterior = posterior
                bestFragment = f
        print "Best fragment:\n",bestFragment[0],bestFragment[1]
        chosenFragments.append(bestFragment)
    print "Final grammar:"
    for t,f in chosenFragments: print t,f

    return FragmentGrammar(chosenFragments)
            

class FragmentGrammar():
    def __init__(self, fragments = []):
        self.featureLogLikelihoods = {}

        self.likelihoodCalculator = {}
        self.likelihoodCalculator['RULE'] = lambda r: self.ruleLogLikelihood(r)
        self.likelihoodCalculator['SPECIFICATION'] = lambda s: self.specificationLogLikelihood(s)
        self.likelihoodCalculator['GUARD'] = lambda g: self.guardLogLikelihood(g)
        self.likelihoodCalculator['CONSTANT'] = lambda k: self.constantLogLikelihood(k)
        self.likelihoodCalculator['MATRIX'] = lambda m: self.matrixLogLikelihood(m)
        self.likelihoodCalculator['FC'] = lambda fc:  self.fCLogLikelihood(fc)
        self.likelihoodCalculator['ENDING'] = lambda e: self.endingLogLikelihood(e)
        
        # different types of fragments
        # fragments of type rule, etc
        self.ruleFragments = [ f for t,f in fragments if t == 'RULE' ]
        self.guardFragments = [ f for t,f in fragments if t == 'GUARD' ]
        self.specificationFragments = [ f for t,f in fragments if t == 'SPECIFICATION' ]

        self.numberOfPhonemes = 40 # should this be the number of phonemes? or number of phonemes in a data set?
        self.numberOfFeatures = 40 # same thing


    def fragmentLikelihood(self, program, fragments):
        ll = float('-inf')
        for fragment in fragments:
            try:
                m = fragment.match(program)
            except MatchFailure:
                continue
            
            fragmentLikelihood = 0.0
            for child,childType in zip(m,fragment.children):
                fragmentLikelihood += self.likelihoodCalculator[childType](child)
            ll = lse(ll, fragmentLikelihood + log(0.5) - log(len(fragments)))
        return ll
        
    def ruleLogLikelihood(self, r):
        ll = self.fragmentLikelihood(r, self.ruleFragments)
        ll = lse(ll,
                 self.fCLogLikelihood(r.focus) +
                 self.fCLogLikelihood(r.structuralChange) +
                 self.guardLogLikelihood(r.leftTriggers) +
                 self.guardLogLikelihood(r.rightTriggers) +
                 log(0.5))
        return ll

    def fCLogLikelihood(self,s):
        ll = log(0.33)
        if isinstance(s, EmptySpecification):
            pass
        elif isinstance(s, FeatureMatrix):
            ll += self.matrixLogLikelihood(s)
        elif isinstance(s, ConstantPhoneme):
            ll += self.constantLogLikelihood(s)
        else:
            raise Exception('fCLogLikelihood: got a bad specification!')

        return lse(ll, self.fragmentLikelihood(s, self.specificationFragments))
    
    def specificationLogLikelihood(self, s):
        ll = log(0.5)
        if isinstance(s, FeatureMatrix):
            ll += self.matrixLogLikelihood(s)
        elif isinstance(s, ConstantPhoneme):
            ll += self.constantLogLikelihood(s)
        else:
            #raise Exception('specificationLogLikelihood: got a bad specification!')
            return float('-inf')

        return lse(ll, self.fragmentLikelihood(s, self.specificationFragments))

    def constantLogLikelihood(self, k):
        return -log(float(self.numberOfPhonemes))

    def matrixLogLikelihood(self, m):
        # todo: come up with a better model of the number of features in the matrix
        return len(m.featuresAndPolarities)*(-log(float(self.numberOfFeatures))) - log(4.0)

    def endingLogLikelihood(self,e):
        # slight prior preference to not attending to the ending
        if e: return log(0.33)
        else: return log(0.66)

    def guardLogLikelihood(self, m):
        # non- fragment case
        ll = sum([ self.specificationLogLikelihood(s) for s in m.specifications ])
        ll += log(0.33) if m.endOfString else log(0.66)
        if len(m.specifications) == 2:
            ll += log(0.33) if m.starred else log(0.66)
        ll -= log(3.0) # todo: improved model of number of matrixes

        ll = lse(ll, self.fragmentLikelihood(m, self.guardFragments))
        return ll
