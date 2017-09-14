from pynini import *

def transducerOfRule(mapping, leftContext, rightContext, alphabet):
    valid = union(*alphabet).closure()
    language = union(*(['.'] + alphabet)).closure()

    return cdrewrite(string_map(mapping),
                     leftContext,
                     rightContext,
                     language,
                     direction = "sim")*valid

def unionTransducer(listOfStuff): return union(*listOfStuff)
        

def runForward(t,x,k = 1):
    try:
        return shortestpath(compose(x,t),nshortest = k).stringify()
    except:
        return None
def inputAcceptor(t):
    return t.project(False)

def makeConstantTransducer(k):
    return transducer(language,k)

def parallelInversion(transducersAndOutputs, alphabet = None):
    try:
        a = [ compose(y,invert(t)).project(True) for y,t in transducersAndOutputs ]
        a = reduce(intersect,a)
        if alphabet != None:
            lm = union(*alphabet).closure()
            a = a*lm
        return shortestpath(a).stringify()
    except:
        # print "Got an exception in parallel inversion..."
        # for y,t in transducersAndOutputs:
        #     print "inverting:"
        #     t = invert(t)
        #     print t
        #     print "composing:"
        #     t = compose(y,t)
        #     print t
        #     print "projecting:"
        #     t = project(True)
        #     print t
        return None

if __name__ == '__main__':
    alphabet = ['a','b','c','z']
    if False:
        language = union(*(['.'] + alphabet)).closure()

        deletionRule = cdrewrite(string_map({'a': ''}),'b','',language,direction = "sim")
        print runForward(deletionRule, 'bacbaa')
        print shortestpath(compose('bbc',deletionRule),nshortest = 2).stringify()

    

    r1 = transducerOfRule({'': 'z'}, '[BOS]', '', alphabet)*\
         transducerOfRule({'a': ''},
                          union('c','b'),
                          "",
                          alphabet)
    r2 = transducerOfRule({'': 'z'}, '', '[EOS]', alphabet)*\
         transducerOfRule({'a': ''},
                          union("c","b"),
                          "",
                          alphabet)
    # x*y is saying run x and then feed its output into y
    if False:
        m1 = transducerOfRule({'': 'a'},'','[EOS]',alphabet)
        m2 = transducerOfRule({'': 'z'},'','[EOS]',alphabet)
        print runForward(m1*m2,'ccc')
    
    stem = 'bcab'
    y1 = runForward(r1,stem)
    y2 = runForward(r2,stem)

    print y1,"~",y2
    print runForward(r1,'bcb'),'~',runForward(r2,'bcb')

    print parallelInversion([(y1,r1),
                             (y2,r2)])

    
