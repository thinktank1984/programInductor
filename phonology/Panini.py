from pynini import *

def transducerOfRule(mapping, leftContext, rightContext, alphabet):
    valid = union(*alphabet).closure()
    language = union(*(['.'] + alphabet)).closure()
    leftContext = leftContext.replace('#','[BOS]')
    rightContext = rightContext.replace('#','[EOS]')

    return cdrewrite(string_map(mapping),
                     leftContext,
                     rightContext,
                     language,
                     direction = "sim")*valid

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

    r1 = transducerOfRule({'': 'aaa'}, '#', '', alphabet)*\
         transducerOfRule({'a': '.'},
                          "c",
                          "",
                          alphabet)
    r2 = transducerOfRule({'': 'zzz'}, '', '#', alphabet)*\
         transducerOfRule({'a': ''},
                          "c",
                          "",
                          alphabet)
    stem = 'bcb'
    y1 = runForward(r1,stem)
    y2 = runForward(r2,stem)

    print y1,"~",y2

    print parallelInversion([(y1,r1),
                             (y2,r2)])

    
