# -*- coding: utf-8 -*-

from rule import *
from time import time
from math import log
from utilities import *

from os import system
import cProfile
from multiprocessing import Pool

universalEmpty = False
def enableUniversalEmpty():
    global universalEmpty
    universalEmpty = True

class MatchFailure(Exception):
    pass

class Fragment():
    def leftUnicode(self): return unicode(self)
    def __str__(self): return unicode(self).encode('utf-8')
    def __repr__(self): return str(self)
    def __eq__(self,other): return unicode(self) == unicode(other)
    def __hash__(self):
        return hash(str(self))
    def match(self,program): raise Exception('Match not implemented for fragment: %s'%str(self))

class VariableFragment(Fragment):
    def __init__(self, ty):
        # ty should be a Python class within the rule structure
        # for example it could be Rule, FeatureMatrix, ...
        self.ty = ty
        self.logPrior = -1.6
    def __unicode__(self): return unicode(self.ty.__name__)
    def latex(self,_=None): return {"Guard":"Trigger",
                                    "PlaceSpecification":"$\\alpha\\text{place}$",
                                    "BoundarySpecification":"+"}.get(self.ty.__name__,self.ty.__name__)
    def match(self, program):
        if isinstance(program,self.ty):
            return [(self.ty,program)]
        raise MatchFailure()
    def sketchCost(self,v,b):
        calculator = {}
        calculator[FC] = 'specification_cost'
        calculator[Guard] = 'guard_cost'
        calculator[FeatureMatrix] = 'specification_cost'
        calculator[ConstantPhoneme] = 'specification_cost'
        calculator[PlaceSpecification] = 'specification_cost'
        return ([], ['%s(%s)'%(calculator[self.ty],v)])
    def numberOfVariables(self): return 1
    def hasConstants(self): return False
    def isDegenerate(self): return False
    def violatesGeometry(self): return False

class RuleFragment(Fragment):
    CONSTRUCTOR = Rule
    def __init__(self, focus, change, left, right):
        self.focus, self.change, self.left, self.right = focus, change, left, right
        self.logPrior = focus.logPrior + change.logPrior + left.logPrior + right.logPrior
        assert isNumber(self.logPrior)
    def violatesGeometry(self):
        return self.focus.violatesGeometry() or self.left.violatesGeometry() or self.right.violatesGeometry()
    def match(self,program):
        return self.focus.match(program.focus) + self.change.match(program.structuralChange) + self.left.match(program.leftTriggers) + self.right.match(program.rightTriggers)
    def __unicode__(self):
        return u"{} ---> {} / {} _ {}".format(self.focus,
                                              self.change,
                                              self.left.leftUnicode(),
                                              self.right)
    def isDegenerate(self):
        return any( f.isDegenerate() for f in [self.focus, # self.change, 
                                               self.left, self.right] )

    def latex(self):
        haveLeft = len(unicode(self.left)) > 0
        haveRight = len(unicode(self.right)) > 0
        if haveLeft and haveRight:
            return "\\phonb{%s}{%s}{%s}{%s}"%(self.focus.latex(),
                                              self.change.latex(),
                                              self.left.latex(True),
                                              self.right.latex())
        if haveLeft and not haveRight:
            return "\\phonl{%s}{%s}{%s}"%(self.focus.latex(),
                                          self.change.latex(),
                                          self.left.latex(True))
        if not haveLeft and haveRight:
            return "\\phonr{%s}{%s}{%s}"%(self.focus.latex(),
                                          self.change.latex(),
                                          self.right.latex())
        if not haveLeft and not haveRight:
            return "\\phonr{%s}{%s}"%(self.focus.latex(),
                                      self.change.latex())
        assert False
            
    @staticmethod
    def abstract(p,q):
        return [
            RuleFragment(focus,change,l,r)
            for focus in FCFragment.abstract(p.focus,q.focus)
            for change in FCFragment.abstract(p.structuralChange,q.structuralChange)
            for l in GuardFragment.abstract(p.leftTriggers, q.leftTriggers)
            for r in GuardFragment.abstract(p.rightTriggers,q.rightTriggers)
        ]

    def numberOfVariables(self): return self.focus.numberOfVariables() + self.change.numberOfVariables() + self.left.numberOfVariables() + self.right.numberOfVariables()
    def hasConstants(self): return self.focus.hasConstants() or self.change.hasConstants() or self.left.hasConstants() or self.right.hasConstants()

    def sketchCost(self,v,b):
        '''v: string representation of a sketch variable. v should have type Rule in the actual sketch.
        b: a feature bank
        returns: (listOfChecksThatHavetoBeTrueToMatch, listOfExtraExpenses)'''
        (fc,fe) = self.focus.sketchCost('%s.focus'%v,b)
        (sc,se) = self.change.sketchCost('%s.structural_change'%v,b)
        (lc,le) = self.left.sketchCost('%s.left_trigger'%v,b)
        (rc,re) = self.right.sketchCost('%s.right_trigger'%v,b)
        return (fc + sc + rc + lc,
                fe + se + le + re)

RuleFragment.BASEPRODUCTIONS = [RuleFragment(VariableFragment(FC),VariableFragment(FC),
                                             VariableFragment(Guard),VariableFragment(Guard))]


class FCFragment(Fragment):
    CONSTRUCTOR = FC
    def __init__(self, child, logPrior = None):
        self.child = child
        self.logPrior = child.logPrior if logPrior == None else logPrior
        assert isNumber(self.logPrior)

    def __unicode__(self): return unicode(self.child)

    def latex(self): return self.child.latex()

    def isDegenerate(self):
        return self.child.isDegenerate()

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
    # The only FC ever created should be in the base grammar
    def numberOfVariables(self): raise Exception('FCFragment: numberOfVariables should never be called')
    def hasConstants(self): raise Exception('FCFragment: hasConstants should never be called')
    
    def sketchCost(self,v,b):
        assert isinstance(self.child,EmptySpecification)
        return (['(%s == null)'%v],[])
    def violatesGeometry(self): return self.child.violatesGeometry()

    @staticmethod
    def abstract(p,q):
        fragments = []
        if unicode(p) != unicode(q):
            fragments += [VariableFragment(FC)]
        if isinstance(p,OffsetSpecification) and isinstance(q,OffsetSpecification):
            fragments += [VariableFragment(OffsetSpecification)]
        if isinstance(p,PlaceSpecification) and isinstance(q,PlaceSpecification):
            fragments += [VariableFragment(PlaceSpecification)]
        if isinstance(p,Specification) and isinstance(q,Specification):
            fragments += SpecificationFragment.abstract(p,q)
        return fragments

FCFragment.BASEPRODUCTIONS = [FCFragment(EmptySpecification(),-1),
                              FCFragment(VariableFragment(OffsetSpecification)),
                              FCFragment(VariableFragment(PlaceSpecification)),
                              FCFragment(VariableFragment(Specification))]

class SpecificationFragment(Fragment):
    CONSTRUCTOR = Specification
    def __init__(self, child, logPrior = None):
        self.child = child
        self.logPrior = child.logPrior if logPrior == None else logPrior
        assert isNumber(self.logPrior)

    def __unicode__(self): return unicode(self.child)

    def latex(self):
        return self.child.latex()

    def match(self, program):
        return self.child.match(program)

    def violatesGeometry(self): return self.child.violatesGeometry()

    def numberOfVariables(self): return self.child.numberOfVariables()
    def hasConstants(self): return self.child.hasConstants()

    def isDegenerate(self): return self.child.isDegenerate()

    @staticmethod
    def abstract(p,q):
        fragments = []
        if unicode(p) != unicode(q):
            fragments += [VariableFragment(FeatureMatrix)]
        if isinstance(p,FeatureMatrix) and isinstance(q,FeatureMatrix):
            fragments += MatrixFragment.abstract(p,q)
        if isinstance(p,ConstantPhoneme) and isinstance(q,ConstantPhoneme):
            fragments += ConstantFragment.abstract(p,q)
        return fragments

    def sketchCost(self,v,b):
        assert False
SpecificationFragment.BASEPRODUCTIONS = [SpecificationFragment(VariableFragment(FeatureMatrix)),
                                         SpecificationFragment(VariableFragment(BoundarySpecification)),
                                         SpecificationFragment(VariableFragment(ConstantPhoneme))]

class MatrixFragment(Fragment):
    CONSTRUCTOR = FeatureMatrix
    def __init__(self, child, logPrior = None):
        self.child = child
        self.childUnicode = unicode(child)
        self.logPrior = child.logPrior if logPrior == None else logPrior
        assert isNumber(self.logPrior)
        if self.violatesGeometry():
            self.logPrior -= 10.0


    def __unicode__(self): return self.childUnicode

    def match(self, program):
        if unicode(program) == self.childUnicode: return []
        raise MatchFailure()

    def numberOfVariables(self): return 0
    def hasConstants(self): return True
    def isDegenerate(self): return self.child.isDegenerate()
    def violatesGeometry(self): return self.child.violatesGeometry()

    @staticmethod
    def fromFeatureMatrix(m):
        return MatrixFragment(m, EMPTYFRAGMENTGRAMMAR.matrixLogLikelihood(m)[0])
        
    @staticmethod
    def abstract(p,q):
        if unicode(p) == unicode(q) and (universalEmpty or len(p.featuresAndPolarities) > 0):
            if len(p.featuresAndPolarities) < 3:
                return [MatrixFragment.fromFeatureMatrix(p)]
            else:
                return [] # prefer matrix fragments that are short
        else:
            return [VariableFragment(FeatureMatrix)]

    def sketchCost(self,v,b):
        assert isinstance(self.child,FeatureMatrix)
        return ([self.child.sketchEquals(v,b)],[])

    def latex(self):
        return self.child.latex()

MatrixFragment.BASEPRODUCTIONS = [] #VariableFragment(FeatureMatrix)]

class ConstantFragment(Fragment):
    CONSTRUCTOR = ConstantPhoneme
    def __init__(self): raise Exception('should never make a constant fragment')
    def __unicode__(self): raise Exception('should never try to print the constant fragment')
    def violatesGeometry(self): return False
    @staticmethod
    def abstract(p,q):
        return [VariableFragment(ConstantPhoneme)]
ConstantFragment.BASEPRODUCTIONS = [] #VariableFragment(ConstantPhoneme)]

class GuardFragment(Fragment):
    CONSTRUCTOR = Guard
    def __init__(self, specifications, endOfString, starred, optionalEnding):
        self.logPrior = sum([s.logPrior for s in specifications ])
        if starred: self.logPrior -= 1.0
        if endOfString: self.logPrior -= 1.0
        if optionalEnding: self.logPrior -= 2.0
        assert not (optionalEnding and endOfString)
        # if optionalEndOfString: self.logPrior -= 1.0
        # assert not (endOfString and optionalEndOfString
        assert isNumber(self.logPrior)

        self.specifications = specifications
        self.starred = starred
        self.endOfString = endOfString
        self.optionalEnding = optionalEnding

    def isDegenerate(self):
        return any( s.isDegenerate() for s in self.specifications )

    def latex(self, l=False):
        pieces = [s.latex() for s in self.specifications]
        if self.starred: pieces[0] = pieces[0] + "*"
        if self.endOfString: pieces.append("\\\\#")
        if self.optionalEnding: pieces[-1] = "{#,%s}"%(pieces[-1])
        if l: pieces.reverse()
        return " ".join(pieces)        

    def violatesGeometry(self):
        return any( s.violatesGeometry() for s in self.specifications )

    def __unicode__(self):
        parts = map(unicode, self.specifications)
        if self.starred: parts[-2] += u'*'
        if self.endOfString: parts += [u'#']
        if self.optionalEnding: parts[-1] = u"{#,%s}"%(parts[-1])
        return u" ".join(parts)
    def leftUnicode(self):
        parts = map(unicode, self.specifications)
        if self.starred: parts[-2] += u'*'
        if self.optionalEnding: parts[-1] = u"{#,%s}"%(parts[-1])
        if self.endOfString: parts += [u'#']
        return u" ".join(reversed(parts))

    def match(self, program):
        if self.endOfString != program.endOfString or self.starred != program.starred or len(self.specifications) != len(program.specifications) or self.optionalEnding != program.optionalEnding:
            raise MatchFailure()

        # This is subtle:
        # We do not support optional end of string within the fragment grammar.
        # But some of the rules have optional end of string.
        # They need to match with something.
        # They should never match with a learned fragment;
        # but we can try and match them with the closest non-learned template
        # if program.optionalEnding and not any( self is b for b in GuardFragment.BASEPRODUCTIONS ):
        #     print "look at me", self, program
        #     raise MatchFailure()
            
        return [ binding for f,p in zip(self.specifications,program.specifications)
                 for binding in f.match(p) ]

    def numberOfVariables(self): return sum( s.numberOfVariables() for s in self.specifications)
    def hasConstants(self): return any(s.hasConstants() for s in self.specifications)

    @staticmethod
    def abstract(p,q):
        if p.endOfString != q.endOfString or p.starred != q.starred or len(p.specifications) != len(q.specifications) or p.optionalEnding != q.optionalEnding:
            return [VariableFragment(Guard)]
        if len(p.specifications) == 0:
            return [GuardFragment([],p.endOfString,False,p.optionalEnding)]
        if len(p.specifications) == 1:
            return [GuardFragment([s1],p.endOfString,p.starred,p.optionalEnding)
                    for s1 in SpecificationFragment.abstract(p.specifications[0],q.specifications[0]) ]
        if len(p.specifications) == 2:
            return [GuardFragment([s1,s2],p.endOfString,p.starred,p.optionalEnding)
                    for s1 in SpecificationFragment.abstract(p.specifications[0],q.specifications[0])
                    for s2 in SpecificationFragment.abstract(p.specifications[1],q.specifications[1])]
        raise Exception('GuardFragment.abstract: should never reach this point. p=%s ; q=%s ; %s ; %s'%(p,q,
                                                                                              len(p.specifications),
                                                                                              p.specifications))

    def sketchCost(self,v,b):
        checks = ['(%s.endOfString == %d)'%(v,int(self.endOfString)),
                  '(%s.starred == %d)'%(v,int(self.starred)),
                  '(%s.optionalEndOfString == %d)'%(v,int(self.optionalEnding))]
        expenses = []
        for component, suffix in zip(self.specifications,['spec','spec2']):
            k,e = component.sketchCost('%s.%s'%(v,suffix),b)
            checks += k
            expenses += e
        #expenses += ['(%s.optionalEndOfString ? 2 : 0)'%v]
        if len(self.specifications) < 1: checks += ['(%s.spec == null)'%v]
        if len(self.specifications) < 2: checks += ['(%s.spec2 == null)'%v]
        return (checks, expenses)
GuardFragment.BASEPRODUCTIONS = [GuardFragment([VariableFragment(Specification)]*s,e,starred,o)
                                 for e in [True,False]
                                 for o in [True,False]
                                 if not (e and o)
                                 for s in range(3)
                                 if (not o or s > 0)
                                 for starred in ([True,False] if s > 1 else [False]) ]    

def programSubexpressions(program):
    '''Yields the sequence of tuples of (ty,expression)'''
    if isinstance(program, Rule):
        yield (Rule,program)
        for x in programSubexpressions(program.focus): yield x
        for x in programSubexpressions(program.structuralChange): yield x
        for x in programSubexpressions(program.leftTriggers): yield x
        for x in programSubexpressions(program.rightTriggers): yield x
    elif isinstance(program, Guard):
        yield (Guard, program)
        for x in programSubexpressions(program.specifications): yield x
    elif isinstance(program, FeatureMatrix):
        yield (Specification, program)
       
    
def proposeFragments(ruleSets, verbose = False):
    abstractFragments = {
        Rule: RuleFragment.abstract,
        Guard: GuardFragment.abstract,
        Specification: SpecificationFragment.abstract
    }

    # Don't allow fragments that are already in the grammar, for example SPECIFICATION -> MATRIX
    badFragments = {
        Guard: [VariableFragment(Guard)],
        Specification: [MatrixFragment(FeatureMatrix([]),0)],
        Rule: []
        }
    badFragments[Guard] += [ f for _,_,f in EMPTYFRAGMENTGRAMMAR.guardFragments ]
    badFragments[Rule] += [ f for _,_,f in EMPTYFRAGMENTGRAMMAR.ruleFragments ]
    badFragments[Specification] += [ f for _,_,f in EMPTYFRAGMENTGRAMMAR.specificationFragments ]

    numberOfPairs = len(ruleSets)*(len(ruleSets) + 1)/2
    startTime = time()
    updateFrequency = 60 # update every this many seconds
    nextUpdate = startTime + updateFrequency
    completedPairs = 0
    
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
                            newFragments = [ f for f in abstractFragments[pt](pf,qf)
                                             if (f not in badFragments[pt]) and (not f.isDegenerate())
                            ]

                            # newFragments = [f for f in newFragments
                                            # if not f.violatesGeometry()]
                            
                            # if [ f for f in newFragments if "instance at" in str(f) ]:
                            #     print pt,pf
                            #     print qt,qf
                            #     print [ str(f) for f in newFragments if "instance at" in str(f) ]
                            #     assert False
                            fragments[pt] = fragments[pt] | set(newFragments)

            completedPairs += 1
            if time() > nextUpdate:
                print "Completed %d/%d (%d%%) antiunifications in %f seconds. ETA %f seconds"%(
                    completedPairs, numberOfPairs, int(float(completedPairs)/numberOfPairs*100),
                    time() - startTime,
                    (numberOfPairs - completedPairs)/(completedPairs/(time() - startTime)))
    for fs in fragments.values():
        for f in fs:
            if f.violatesGeometry():
                print "VIOLATION",f
    fragments = {tp: {f for f in fs if not f.violatesGeometry() }
                 for tp,fs in fragments.iteritems()} 


    totalNumberOfFragments = sum([len(v) for v in fragments.values() ])
    print "Discovered %d unique fragments in %f seconds"%(totalNumberOfFragments,time() - startTime)
    if verbose:
        for ty in fragments:
            print "Fragments of type %s (%d):"%(ty,len(fragments[ty]))
            for f in fragments[ty]:
                print f
            print ""

    return [ (t, f) for t in fragments for f in fragments[t] ] # if t != Rule ] #and t != 'GUARD' ]

GLOBALCANDIDATEDATA = None
def scoreCandidate((currentFragments,t,f)):
    global GLOBALCANDIDATEDATA
    smoothing = GLOBALCANDIDATEDATA["smoothing"]
    ruleEquivalenceClasses = GLOBALCANDIDATEDATA["ruleEquivalenceClasses"]
    newGrammar = FragmentGrammar(currentFragments + [(t,0,f)]).\
                 estimateParameters(ruleEquivalenceClasses,smoothing = smoothing)
    newScore = newGrammar.AIC(ruleEquivalenceClasses)
    bestUses = newGrammar.MAP_uses(ruleEquivalenceClasses,f)
    newGrammar.clearCaches()
    if bestUses <= 1.1: newScore = float('inf')
    return newScore, newGrammar.fragments    
    
def induceFragmentGrammar(ruleEquivalenceClasses, maximumGrammarSize = 40, smoothing = 1.0,
                          CPUs = 1):
    # Fork workers for evaluating candidates
    # They will need some global data
    global GLOBALCANDIDATEDATA
    GLOBALCANDIDATEDATA = GLOBALCANDIDATEDATA or {}
    GLOBALCANDIDATEDATA["ruleEquivalenceClasses"] = ruleEquivalenceClasses
    GLOBALCANDIDATEDATA["smoothing"] = smoothing
    workers = Pool(CPUs) # create this before we propose of fragments to save memory
    
    startTime = time()
    
    fragments = proposeFragments(ruleEquivalenceClasses, verbose = False)

    currentGrammar = EMPTYFRAGMENTGRAMMAR
    previousDescriptionLength = float('inf')

    typeOrdering = [Specification,Guard,Rule]

    # def scoreCandidate((t,f)):
    #     newGrammar = FragmentGrammar(currentGrammar.fragments + [(t,0,f)]).\
    #                  estimateParameters(ruleEquivalenceClasses,smoothing = smoothing)
    #     newScore = newGrammar.AIC(ruleEquivalenceClasses)
    #     bestUses = newGrammar.MAP_uses(ruleEquivalenceClasses,f)
    #     newGrammar.clearCaches()
    #     if bestUses <= 1.1: newScore = float('inf')
    #     return newScore, newGrammar
    
    while len(currentGrammar.fragments) - len(EMPTYFRAGMENTGRAMMAR.fragments) < maximumGrammarSize:
        possibleNewFragments = [ (currentGrammar.fragments,t,f)
                                 for (t,f) in fragments
                                 if t == typeOrdering[0] and not currentGrammar.hasFragment(f) ]
        candidates = workers.map(scoreCandidate, possibleNewFragments)

        if candidates == []:
            bestScore = float('inf')
            print("No candidates.")
        else:
            (bestScore,bestGrammar) = min(candidates)
            bestGrammar = FragmentGrammar(bestGrammar)
            # for os,og in candidates:
            #     if os == bestScore:
            #         print "This is as good as the best:"
            #         print og
                    
        if candidates != [] and bestScore <= previousDescriptionLength:
            previousDescriptionLength = bestScore
            currentGrammar = bestGrammar
            print "Updated grammar to (overall score = %.2f):"%(bestScore)
            print bestGrammar
        else:
            print "No improvement possible using fragments of type",typeOrdering[0]
            typeOrdering = typeOrdering[1:]
            if typeOrdering == []:
                print "Done inducing fragment grammar"            
                break
            else:
                print "Moving on to fragments of type",typeOrdering[0]

    print "Total grammar induction time:", time() - startTime
    return currentGrammar
            

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
        self.likelihoodCalculator[Rule] = lambda r: self.ruleLogLikelihood(r)
        self.likelihoodCalculator[Specification] = lambda s: self.specificationLogLikelihood(s)
        self.likelihoodCalculator[Guard] = lambda g: self.guardLogLikelihood(g)
        self.likelihoodCalculator[ConstantPhoneme] = lambda k: self.constantLogLikelihood(k)
        self.likelihoodCalculator[FeatureMatrix] = lambda m: self.matrixLogLikelihood(m)
        self.likelihoodCalculator[FC] = lambda fc:  self.fCLogLikelihood(fc)
        self.likelihoodCalculator[BoundarySpecification] = lambda o: self.boundaryLogLikelihood(o)
        self.likelihoodCalculator[OffsetSpecification] = lambda o: self.offsetLogLikelihood(o)
        self.likelihoodCalculator[PlaceSpecification] = lambda o: self.placeLogLikelihood(o)
        
        # different types of fragments
        # fragments of type rule, etc
        self.ruleFragments = normalizeLogDistribution([ f for f in fragments if f[0] == Rule ],
                                                      index = 1)
        self.guardFragments = normalizeLogDistribution([ f for f in fragments if f[0] == Guard ],
                                                       index = 1)
        self.specificationFragments = normalizeLogDistribution([ f for f in fragments if f[0] == Specification ],
                                                               index = 1)
        self.focusChangeFragments = normalizeLogDistribution([ f for f in fragments if f[0] == FC ],
                                                             index = 1)
        assert len(fragments) == len(self.ruleFragments) + len(self.guardFragments) + len(self.specificationFragments) + len(self.focusChangeFragments)
        self.fragments = self.ruleFragments + self.guardFragments + self.specificationFragments + self.focusChangeFragments
        
        self.numberOfPhonemes = 40 # should this be the number of phonemes? or number of phonemes in a data set?
        self.numberOfFeatures = 40 # same thing

    def clearCaches(self):
        self.ruleTable = {}
        self.guardTable = {}
        self.matrixTable = {}
        self.specificationTable = {}
        self.fCTable = {}


    def __str__(self):
        def makingNamingIntuitive(n):
            correspondence = [('Guard', 'Trigger'),
                              ('BoundarySpecification', '+'),
                              ('OffsetSpecification', 'ℤ'),
                              ('PlaceSpecification', 'αplace'),
                              ('Specification', 'PhonemeSet'),
                              ('ConstantPhoneme', 'Phoneme')]
            for k,v in correspondence:
                n = n.replace(k,v)
            return n
        return formatTable([ map(makingNamingIntuitive, ["%f"%l,"%s"%t.__name__ + " ::= ", str(f) ])
                             for t,l,f in self.fragments])

    def hasFragment(self,f):
        return any(_f == f for _,_,_f in self.fragments)

    def fragmentLikelihood(self, program, fragments):
        '''returns (likelihood, {fragment: expected uses})'''
        z = float('-inf')
        uses = []
        for concreteClass,lf,fragment in fragments:
            assert isinstance(program, concreteClass)
            try:
                m = fragment.match(program)
            except MatchFailure:
                continue
            
            fragmentLikelihood = 0.0
            theseUses = {fragment:1}
            for childType,child in m:
                (childLikelihood, childUses) = self.likelihoodCalculator[childType](child)
                theseUses = mergeCounts(childUses, theseUses)
                fragmentLikelihood += childLikelihood
            uses.append((fragmentLikelihood + lf,theseUses))
            z = lse(z, fragmentLikelihood + lf)
        expectedUses = {}
        for ll,u in uses:
            probabilityOfTheseUses = math.exp(ll - z)
            expectedUses = mergeCounts(expectedUses, scaleDictionary(probabilityOfTheseUses, u))
        return z,expectedUses

    def ruleLogLikelihood(self, r):
        key = unicode(r)
        if key in self.ruleTable:
            return self.ruleTable[key]
        
        ll,u = self.fragmentLikelihood(r, self.ruleFragments)
        self.ruleTable[key] = (ll,u)
        return ll,u

    def fCLogLikelihood(self,s):
        key = unicode(s)
        if key in self.fCTable: return self.fCTable[key]
        ll,u = self.fragmentLikelihood(s, self.focusChangeFragments)
        self.fCTable[key] = ll,u
        return ll,u
    
    def specificationLogLikelihood(self, s):
        key = unicode(s)
        if key in self.specificationTable: return self.specificationTable[key]
        ll,u = self.fragmentLikelihood(s, self.specificationFragments)
        self.specificationTable[key] = (ll,u)
        return ll,u

    def constantLogLikelihood(self, k):
        if isinstance(k,ConstantPhoneme):
            return -log(float(self.numberOfPhonemes)), {}
        else:
            raise Exception('constantLogLikelihood: did not get a constant')

    def offsetLogLikelihood(self, o):
        if isinstance(o,OffsetSpecification):
            return -log(1.), {}
        else:
            raise Exception('offsetLogLikelihood: did not get offset')

    def placeLogLikelihood(self, o):
        if isinstance(o,PlaceSpecification):
            return -log(2.0), {}
        else:
            raise Exception('placeLogLikelihood: did not get offset')

    def boundaryLogLikelihood(self, o):
        if isinstance(o,BoundarySpecification):
            return 0.0, {}
        else:
            raise Exception('boundaryLogLikelihood: did not get boundary')

    def matrixSizeLogLikelihood(self,l):
        return log(0.25) # uniform prior
        if l == 0: return log(0.3)
        if l == 1: return log(0.6)
        return log(0.05)

    def matrixLogLikelihood(self, m):
        if isinstance(m,FeatureMatrix):
            # empirical probabilities on matrix problems: [(0, 0.3225806451612903), (1, 0.6129032258064516), (2, 0.03225806451612903), (3, 0.03225806451612903)]
            return len(m.featuresAndPolarities)*(-log(float(self.numberOfFeatures))) + self.matrixSizeLogLikelihood(len(m.featuresAndPolarities)),{}
        else:
            raise Exception('matrixLogLikelihood')

    def guardLengthLogLikelihood(self,l):
        if l == 0: return log(0.5)
        if l == 1: return log(0.33)
        if l == 2: return log(0.17)
        raise Exception('unhandled guardflength')

    def guardLogLikelihood(self, m):
        key = unicode(m)
        if key in self.guardTable: return self.guardTable[key]
        ll,u = self.fragmentLikelihood(m, self.guardFragments)
        if m.optionalEnding:
            ll -= 1.
        self.guardTable[key] = (ll,u)
        return ll,u

    def sketchUniversalGrammar(self,bank):
        from sketchSyntax import definePreprocessor
        definitions = {}
        for dictionaryKey, fragments, v in [('UNIVERSALRULEGRAMMAR',self.ruleFragments,'r'),
                                            ('UNIVERSALSPECIFICATIONGRAMMAR',self.specificationFragments,'s'),
                                            ('UNIVERSALGUARDGRAMMAR',self.guardFragments, 'g')]:
            # Sort them so that the ones with fewer variables come first
            fragments = sorted(fragments, key = lambda z: z[2].numberOfVariables())
            for baseType,_,f in fragments:
                # Make sure it is a new fragment and not already in the base grammar
                if f in baseType2fragmentType[baseType].BASEPRODUCTIONS: continue
                if not f.hasConstants(): continue

                checks, expenses = f.sketchCost(v,bank)

                # print "Sketching fragment:",f
                # print checks
                # print expenses
                
                check = "&&".join(['1'] + checks)
                cost = " + ".join(['1'] + expenses)
                definitions[dictionaryKey] = definitions.get(dictionaryKey,'')
                definitions[dictionaryKey] += " if (%s) return %s; "%(check, cost)
        for k,v in definitions.iteritems():
            definePreprocessor(k,v)

    def insideOutside(self, frontiers, smoothing = 0):
        '''frontiers: list of list of rules.
        returns a new fragment grammar with the same structure but different probabilities'''
        uses = {}
        for frontier in frontiers:
            weightedUses = [ self.ruleLogLikelihood(r) for r in frontier ]
            uses = reduce(mergeCounts, [uses] + [ scaleDictionary(math.exp(w), u)
                                                  for w,u in normalizeLogDistribution(weightedUses) ])
        newFragments = [ (k, safeLog(uses.get(f,0.0) + smoothing), f) for k,_,f in self.fragments ]

        return FragmentGrammar(newFragments)

    def estimateParameters(self, frontiers, smoothing = 0):
        '''frontiers: list of list of rules.
        returns a new fragment grammar with the same structure but different probabilities'''
        flatFragments = [(t,0.0,f) for t,_,f in self.fragments ]
        return FragmentGrammar(flatFragments).insideOutside(frontiers, smoothing = smoothing)

    def frontierLikelihood(self, frontier):
        '''frontier: list of rules.
           returns log sum P[r|G]'''
        #return lseList([ self.ruleLogLikelihood(r)[0] for r in frontier ])
        return max( self.ruleLogLikelihood(r)[0] for r in frontier )
    def frontiersLikelihood(self,frontiers):
        return sum([self.frontierLikelihood(f) for f in frontiers ])
    def logPrior(self):
        return sum([ f.logPrior for _,_,f in self.fragments ])
    def frontiersLogJoint(self,frontiers, priorWeight = 0.05):
        return self.frontiersLikelihood(frontiers) + priorWeight*self.logPrior()
    def AIC(self, frontiers, alpha = 0.1):
        return alpha*len(self.fragments) - self.frontiersLogJoint(frontiers)
    def MAP_uses(self, frontiers, fragment):
        """returns the number of times that this fragment is used in the map rules"""
        uses = 0
        for frontier in frontiers:
            bestUses = max([self.ruleLogLikelihood(r) for r in frontier], key=lambda lu: lu[0])[1]
            uses += bestUses.get(fragment,0.)
        return uses

    def export(self,f):
        dumpPickle(self.fragments, f)
    @staticmethod
    def load(f):
        return FragmentGrammar(loadPickle(f))
        


BASEPRODUCTIONS = [(k.CONSTRUCTOR,
                    # Penalize deletion/insertion/copying
                    -math.log(50)*int(unicode(f) in [u"Ø",u"OffsetSpecification",u"BoundarySpecification",
                                                     u"PlaceSpecification"]),
                    f)
                   for k in [RuleFragment,FCFragment,SpecificationFragment,MatrixFragment,ConstantFragment,GuardFragment]
                   for f in k.BASEPRODUCTIONS]
EMPTYFRAGMENTGRAMMAR = FragmentGrammar(BASEPRODUCTIONS)
def getEmptyFragmentGrammar():
    global EMPTYFRAGMENTGRAMMAR
    return EMPTYFRAGMENTGRAMMAR
baseType2fragmentType = dict((k.CONSTRUCTOR,k)\
                             for k in [RuleFragment,FCFragment,SpecificationFragment,MatrixFragment,ConstantFragment,GuardFragment])

if __name__ == '__main__':
    from parseSPE import parseRule
    r = parseRule('[ -tense ] ---> 0 /  _ {#,[ -sonorant ]}')
    print EMPTYFRAGMENTGRAMMAR.ruleLogLikelihood(r)[1]
    print r.rightTriggers
    assert False
    ruleSets = [[parseRule('e > a / # _ [ -voice ]* h #')],
                [parseRule('e > 0 / # _ [ -voice ]* [ +vowel ]#')]]
    proposeFragments(ruleSets,verbose = True)
    
    print EMPTYFRAGMENTGRAMMAR.ruleLogLikelihood(parseRule('0 > a / _#'))[0]
    print EMPTYFRAGMENTGRAMMAR.ruleLogLikelihood(parseRule('e > -1 / _#'))[0]
    assert False
    
    print str(EMPTYFRAGMENTGRAMMAR)
    r = parseRule('0 > 1 / # _ [ -voice ]* h #')
    print r
    print EMPTYFRAGMENTGRAMMAR.ruleLogLikelihood(r)
    print EMPTYFRAGMENTGRAMMAR.insideOutside([[r]],smoothing = 0)
    print EMPTYFRAGMENTGRAMMAR.insideOutside([[r]],smoothing = 0).ruleLogLikelihood(r)[0]

