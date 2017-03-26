# -*- coding: utf-8 -*-


from rule import *
from time import time
from math import log
from utilities import *

from os import system
import cProfile
from multiprocessing import Pool

class MatchFailure(Exception):
    pass

class Fragment():
    def __str__(self): return unicode(self).encode('utf-8')
    def __eq__(self,other): return unicode(self) == unicode(other)
    def __hash__(self):
        return hash(str(self))
    def match(self,program): raise Exception('Match not implemented for fragment: %s'%str(self))

class VariableFragment(Fragment):
    def __init__(self, ty, k = []):
        self.ty = ty
        self.logPrior = -1.6
        self.k = k
    def __unicode__(self): return unicode(self.ty)
    def match(self, program):
        if self.k == [] or any([ isinstance(program,_k) for _k in self.k ]):
            return [(self.ty,program)]
        raise MatchFailure()

class RuleFragment(Fragment):
    def __init__(self, focus, change, left, right):
        self.focus, self.change, self.left, self.right = focus, change, left, right
        self.logPrior = focus.logPrior + change.logPrior + left.logPrior + right.logPrior
    def match(self,program):
        return self.focus.match(program.focus) + self.change.match(program.structuralChange) + self.left.match(program.leftTriggers) + self.right.match(program.rightTriggers)
    def __unicode__(self):
        return u"{} ---> {} / {} _ {}".format(self.focus,
                                              self.change,
                                              self.left,
                                              self.right)
    @staticmethod
    def abstract(p,q):
        if p.copyOffset != 0 or q.copyOffset != 0:
            raise Exception('abstractRuleFragments: copy offsets not yet supported')

        return [
            RuleFragment(focus,change,l,r)
            for focus in FCFragment.abstract(p.focus,q.focus)
            for change in FCFragment.abstract(p.structuralChange,q.structuralChange)
            for l in GuardFragment.abstract(p.leftTriggers, q.leftTriggers)
            for r in GuardFragment.abstract(p.rightTriggers,q.rightTriggers)
        ]


class FCFragment(Fragment):
    def __init__(self, child, logPrior = None):
        self.child = child
        self.logPrior = child.logPrior if logPrior == None else logPrior

    def __unicode__(self): return unicode(self.child)

    def match(self, program):
        if isinstance(program, EmptySpecification):
            if isinstance(self.child, EmptySpecification):
                return []
            else:
                raise MatchFailure()
        else:
            if isinstance(self.child, EmptySpecification):
                raise MatchFailure()
            else:
                return self.child.match(program)

    @staticmethod
    def abstract(p,q):
        fragments = []
        if unicode(p) != unicode(q):
            fragments += [VariableFragment('FC')]
        if isinstance(p,EmptySpecification) and isinstance(q,EmptySpecification):
            fragments += [FCFragment(EmptySpecification(), -1)]
        if (not isinstance(p,EmptySpecification)) and (not isinstance(q,EmptySpecification)):
            fragments += SpecificationFragment.abstract(p,q)
        return fragments

class SpecificationFragment(Fragment):
    def __init__(self, child, logPrior = None):
        self.child = child
        self.logPrior = child.logPrior if logPrior == None else logPrior

    def __unicode__(self): return unicode(self.child)

    def match(self, program): raise Exception('not implemented')

    @staticmethod
    def abstract(p,q):
        fragments = []
        if unicode(p) != unicode(q):
            fragments += [VariableFragment('MATRIX',[FeatureMatrix])]
        if isinstance(p,FeatureMatrix) and isinstance(q,FeatureMatrix):
            fragments += MatrixFragment.abstract(p,q)
        if isinstance(p,ConstantPhoneme) and isinstance(q,ConstantPhoneme):
            fragments += ConstantFragment.abstract(p,q)
        return fragments

class MatrixFragment(Fragment):
    def __init__(self, child, logPrior = None):
        self.child = child
        self.childUnicode = unicode(child)
        self.logPrior = child.logPrior if logPrior == None else logPrior

    def __unicode__(self): return self.childUnicode

    def match(self, program):
        if unicode(program) == self.childUnicode: return []
        raise MatchFailure()
        
    @staticmethod
    def abstract(p,q):
        if unicode(p) == unicode(q):
            if len(p.featuresAndPolarities) < 3:
                return [MatrixFragment(p, -p.cost())]
            else:
                return [] # prefer matrix fragments that are short
        else:
            return [VariableFragment('MATRIX',[FeatureMatrix])]

class ConstantFragment(Fragment):
    def __init__(self): raise Exception('should never make a constant fragment')
    def __unicode__(self): return u"CONSTANT"
    @staticmethod
    def abstract(p,q):
        return [VariableFragment('CONSTANT',[ConstantPhoneme])]

class GuardFragment(Fragment):
    def __init__(self, specifications, endOfString, starred):
        self.logPrior = sum([s.logPrior for s in specifications ])
        if starred: self.logPrior -= 1.0
        if endOfString: self.logPrior -= 1.0

        self.specifications = specifications
        self.starred = starred
        self.endOfString = endOfString

    def __unicode__(self):
        parts = map(unicode, self.specifications)
        if self.starred: parts[-2] += u'*'
        if self.endOfString: parts += [u'#']
        return u" ".join(parts)

    def match(self, program):
        if self.endOfString != program.endOfString or self.starred != program.starred or len(self.specifications) != len(program.specifications):
            raise MatchFailure()

        return [ binding for f,p in zip(self.specifications,program.specifications)
                 for binding in f.match(p) ]

    @staticmethod
    def abstract(p,q):
        if p.endOfString != q.endOfString or p.starred != q.starred or len(p.specifications) != len(q.specifications):
            return [VariableFragment('GUARD')]
        if len(p.specifications) == 0:
            return [GuardFragment([],p.endOfString,False)]
        if len(p.specifications) == 1:
            return [GuardFragment([s1],p.endOfString,p.starred)
                    for s1 in SpecificationFragment.abstract(p.specifications[0],q.specifications[0]) ]
        if len(p.specifications) == 2:
            return [GuardFragment([s1,s2],p.endOfString,p.starred)
                    for s1 in SpecificationFragment.abstract(p.specifications[0],q.specifications[0])
                    for s2 in SpecificationFragment.abstract(p.specifications[1],q.specifications[1])]
        raise Exception('GuardFragment.abstract: should never reach this point')
    

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
       
    
def proposeFragments(problems, verbose = False):
    ruleSets = []
    for problem in problems:
        # problem should be a list of solutions
        # each solution should be a list of rules
        ruleSets.append(set([ r for s in problem for r in s ]))

    abstractFragments = {
        'RULE': RuleFragment.abstract,
        'GUARD': GuardFragment.abstract,
        'SPECIFICATION': SpecificationFragment.abstract
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
                            newFragments = [ f for f in abstractFragments[pt](pf,qf) if str(f) != pt ]
                            # if [ f for f in newFragments if "instance at" in str(f) ]:
                            #     print pt,pf
                            #     print qt,qf
                            #     print [ str(f) for f in newFragments if "instance at" in str(f) ]
                            #     assert False
                            fragments[pt] = fragments[pt] | set(newFragments)

    
    totalNumberOfFragments = sum([len(v) for v in fragments.values() ])
    print "Discovered %d unique fragments in %f seconds"%(totalNumberOfFragments,time() - startTime)
    if verbose:
        for ty in fragments:
            print "Fragments of type %s (%d):"%(ty,len(fragments[ty]))
            for f in fragments[ty]:
                print f
            print ""

    return [ (t, f) for t in fragments for f in fragments[t]  ]

def posteriorWithFragmentAdded(parameters):
    (problems, fragments, newFragment) = parameters
    newGrammar = FragmentGrammar(fragments + [newFragment])
    l = sum([ max([ sum([ newGrammar.ruleLogLikelihood(r) for r in s ]) for s in problem ])
              for problem in problems ])
    posterior = l + newFragment[1].logPrior
    f = newFragment
    print "Considering %s %s\n\t%f + %f = %f"%(f[0],f[1],l,f[1].logPrior,posterior)
    return posterior
    
def pickFragments(problems, fragments, maximumGrammarSize):
    chosenFragments = []

    expressionTable = {}
    problems = [ [ [ r.share(expressionTable) for r in s ] for s in p ]
                 for p in problems ]

    startTime = time()
    while len(chosenFragments) < maximumGrammarSize:
        bestFragment = None
        bestPosterior = float('-inf')
        parameters = [ (problems, chosenFragments, x) for x in fragments if not x in chosenFragments]
        posteriors = map(posteriorWithFragmentAdded, parameters) #Pool(2).c
        (bestPosterior, (_p,_c,bestFragment)) = max(zip(posteriors, parameters))
        print "Best fragment:\n",bestFragment[0],bestFragment[1]
        print "Best posterior:",bestPosterior
        chosenFragments.append(bestFragment)
    print "Final grammar:"
    for t,f in chosenFragments: print t,f
    print "Search time:",(time() - startTime),"seconds"
    return chosenFragments

def induceGrammar(problems, maximumGrammarSize = 2):
    fragments = proposeFragments(problems, verbose = True)
    p = problems
    picker = pickFragments
    chosenFragments = picker(p, fragments, maximumGrammarSize)

    return FragmentGrammar(chosenFragments)
            

class FragmentGrammar():
    def __init__(self, fragments = []):
        self.featureLogLikelihoods = {}

        # memorization table for likelihood calculations
        self.ruleTable = {}
        self.guardTable = {}
        self.matrixTable = {}
        self.specificationTable = {}
        self.fCTable = {}
        
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
            for childType,child in m:
                fragmentLikelihood += self.likelihoodCalculator[childType](child)
            ll = lse(ll, fragmentLikelihood + log(0.5) - log(len(fragments)))
        return ll
        
    def ruleLogLikelihood(self, r):
        key = unicode(r)
        if key in self.ruleTable:
            return self.ruleTable[key]
        
        penalty = 0.0 if self.ruleFragments == [] else log(0.5)
        
        ll = self.fragmentLikelihood(r, self.ruleFragments)
        ll = lse(ll,
                 self.fCLogLikelihood(r.focus) +
                 self.fCLogLikelihood(r.structuralChange) +
                 self.guardLogLikelihood(r.leftTriggers) +
                 self.guardLogLikelihood(r.rightTriggers) +
                 penalty)
        self.ruleTable[key] = ll
        return ll

    def fCLogLikelihood(self,s):
        key = unicode(s)
        if key in self.fCTable: return self.fCTable[key]
        
        penalty = 0.0 if self.specificationFragments == [] else log(0.5)
        ll = log(0.33) + penalty
        if isinstance(s, EmptySpecification):
            pass
        elif isinstance(s, FeatureMatrix):
            ll += self.matrixLogLikelihood(s)
        elif isinstance(s, ConstantPhoneme):
            ll += self.constantLogLikelihood(s)
        else:
            raise Exception('fCLogLikelihood: got a bad specification!')

        ll = lse(ll, self.fragmentLikelihood(s, self.specificationFragments))
        self.fCTable[key] = ll
        return ll
    
    def specificationLogLikelihood(self, s):
        key = unicode(s)
        if key in self.specificationTable: return self.specificationTable[key]
        
        penalty = 0.0 if self.specificationFragments == [] else log(0.5)
        ll = log(0.5) + penalty
        if isinstance(s, FeatureMatrix):
            ll += self.matrixLogLikelihood(s)
        elif isinstance(s, ConstantPhoneme):
            ll += self.constantLogLikelihood(s)
        else:
            raise Exception('specificationLogLikelihood: got a bad specification!')

        ll = lse(ll, self.fragmentLikelihood(s, self.specificationFragments))
        self.specificationTable[key] = ll
        return ll

    def constantLogLikelihood(self, k):
        if isinstance(k,ConstantPhoneme):
            return -log(float(self.numberOfPhonemes))
        else:
            raise Exception('constantLogLikelihood: did not get a constant')

    def matrixLogLikelihood(self, m):
        if isinstance(m,FeatureMatrix):
            # todo: come up with a better model of the number of features in the matrix
            return len(m.featuresAndPolarities)*(-log(float(self.numberOfFeatures))) - log(4.0)
        else:
            raise Exception('matrixLogLikelihood')

    def endingLogLikelihood(self,e):
        # slight prior preference to not attending to the ending
        if e: return log(0.33)
        else: return log(0.66)

    def guardLogLikelihood(self, m):
        key = unicode(m)
        if key in self.guardTable: return self.guardTable[key]
        penalty = 0.0 if self.guardFragments == [] else log(0.5)
        # non- fragment case
        ll = sum([ self.specificationLogLikelihood(s) for s in m.specifications ]) + penalty
        ll += log(0.33) if m.endOfString else log(0.66)
        if len(m.specifications) == 2:
            ll += log(0.33) if m.starred else log(0.66)
        ll -= log(3.0) # todo: improved model of number of matrixes

        ll = lse(ll, self.fragmentLikelihood(m, self.guardFragments))
        self.guardTable[key] = ll
        return ll

