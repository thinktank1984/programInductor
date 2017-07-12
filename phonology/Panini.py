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

def parallelInversion(transducersAndOutputs):
    try:
        a = [ compose(y,invert(t)).project(True) for y,t in transducersAndOutputs ]
        return shortestpath(reduce(intersect,a)).stringify()
    except:
        return None

if __name__ == '__main__':
    alphabet = ['a','b','c','z']

    r1 = transducerOfRule({'': 'aaa'}, '[BOS]', '', alphabet)*\
         transducerOfRule({'a': 'a'},
                          "c",
                          "",
                          alphabet)
    r2 = transducerOfRule({'': 'zzz'}, '', '[EOS]', alphabet)*\
         transducerOfRule({'a': ''},
                          union("c","b"),
                          "",
                          alphabet)
    # x*y is saying run x and then feed its output into y
    m1 = transducerOfRule({'': 'a'},'','[EOS]',alphabet)
    m2 = transducerOfRule({'': 'z'},'','[EOS]',alphabet)
    print runForward(m1*m2,'ccc')
    
    stem = 'bcaab'
    y1 = runForward(r1,stem)
    y2 = runForward(r2,stem)

    print y1,"~",y2

    print parallelInversion([(y1,r1),
                             (y2,r2)])

    
