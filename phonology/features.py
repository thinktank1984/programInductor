# -*- coding: utf-8 -*-

palatal = "palatal"
palletized = "palletized"
sibilant = "sibilant"
sonorant = "sonorant"
coronal = "coronal"
# approximate = "approximate"
# because stress and high tones are written the same we model stress as a high tone
# stressed = "stressed"
retroflex = "retroflex"
creaky = "creaky"
risingTone = "risingTone"
highTone = "highTone"
lowTone = "lowTone"
middleTone = "middleTone"
longVowel = "long"
vowel = "vowel"
tense = "tense"
lax = "lax"
high = "high"
middle = "middle"
low = "low"
#front = "front"
#central = "central"
back = "back"
rounded = "rounded"
#unrounded = "unrounded"
#bilabial = "bilabial"
#stop = "stop"
voice = "voice"
#fricative = "fricative"
#labiodental = "labiodental"
#dental = "dental"
#alveolar = "alveolar"
#labiovelar = "labiovelar"
#velar = "velar"
nasal = "nasal"
#uvular = "uvular"
glide = "glide"
liquid = "liquid"
lateral = "lateral"
trill = "trill"
flap = "flap"
#affricate = "affricate"
#alveopalatal = "alveopalatal"
anterior = "anterior"
aspirated = "aspirated"
unreleased = "unreleased"
#laryngeal = "laryngeal"
#pharyngeal = "pharyngeal"
syllableBoundary = "syllableBoundary"
wordBoundary = "wordBoundary"
continuant = "continuant"
syllabic = "syllabic"
delayedRelease = "delayedRelease"

featureMap = {
    # unrounded vowels
    u"i": [voice,tense,high],
    u"ɨ": [voice,tense,high,back],
    u"ɩ": [voice,high],
    u"e": [voice,tense],
    u"ə": [voice,back],
    u"ɛ": [voice],
    u"æ": [voice,low,tense],
    u"a": [voice,low,tense,back],
    u"ʌ": [voice,back,tense],
    # rounded vowels
    u"u": [voice,tense,high,back,rounded],
    u"ü": [voice,tense,high,rounded],
    u"ʊ": [voice,high, back, rounded],
    u"o": [voice,tense,back,rounded],
    u"ö": [voice,tense,rounded],
    u"ɔ": [voice,back,rounded],
    #possibly missing are umlauts

    # consonance
    u"p": [anterior,],
    u"p|": [anterior,unreleased],
    u"p^h": [anterior,aspirated],
    u"b": [anterior,voice],
    u"f": [anterior,continuant],
    u"v": [anterior,continuant,voice],
    u"β": [anterior,continuant,voice],
    u"m": [anterior,nasal,voice,sonorant],#continuant],
    u"m̥": [anterior,nasal,sonorant],#,continuant],
    u"θ": [anterior,continuant,coronal],
    u"d": [anterior,voice,coronal],
    #u"d̪": [voice,coronal],
    u"d^z": [anterior,coronal,voice,delayedRelease],
    u"t": [anterior,coronal],
    #u"t̪": [coronal],
    u"t|": [anterior,coronal,unreleased],
    u"t^s": [anterior,coronal,delayedRelease],
    u"t^h": [anterior,aspirated,coronal],
    u"ṭ": [anterior,retroflex,coronal],
    u"ḍ": [anterior,retroflex,coronal,voice],
    u"ṛ": [anterior,retroflex,coronal,voice,continuant],
    u"ð": [anterior,continuant,voice,coronal],
    u"z": [anterior,continuant,voice,coronal, sibilant],
    u"ǰ": [voice,coronal,sibilant],#alveopalatal,
    u"ž": [continuant,voice,coronal, sibilant],#alveopalatal,
    u"s": [anterior,continuant,coronal, sibilant],
    u"n": [anterior,nasal,voice,coronal,sonorant],#continuant],
    u"ṇ": [anterior,retroflex,nasal,voice,sonorant],#continuant],
    u"n̥": [anterior,nasal,coronal,sonorant],#continuant],
    u"ñ": [nasal,voice,coronal,sonorant],#continuant],alveopalatal,
    u"š": [continuant,coronal, sibilant],#alveopalatal,
    u"c": [palatal,coronal], # NOT the same thing as palletized
    u"č": [coronal,sibilant],#alveopalatal,
    u"č^h": [coronal,sibilant,aspirated],#alveopalatal,
    u"k": [back,high],
    u"k|": [back,high,unreleased],
    u"k^h": [back,high,aspirated],
    u"k^y": [back,high,palletized],
    u"x": [back,high,continuant],
    u"X": [back,continuant], # χ
    u"x^y": [back,high,continuant,palletized],
    u"g": [back,high,voice],
    u"g^y": [back,high,voice,palletized],
    u"ɣ": [back,high,continuant,voice],
    u"ŋ": [back,high,nasal,voice,sonorant],#continuant],
    u"q": [back],
    u"N": [back,nasal,voice],#continuant],
    u"G": [back,voice],
    u"ʔ": [sonorant,low],#laryngeal,
    u"h": [continuant,sonorant,low],#laryngeal,
    u"ħ": [back, low,continuant,sonorant],

    # glides
    u"w": [glide,voice,sonorant,continuant],
    u"y": [glide,palletized,voice,sonorant,continuant],
    u"j": [glide,palletized,voice,sonorant,continuant],

    # liquids
    u"r": [liquid,voice,coronal,sonorant,continuant],
    u"r̃": [liquid,trill,voice,coronal,sonorant,continuant],
    u"r̥̃": [liquid,trill,coronal,sonorant,continuant],
    u"ř": [liquid,flap,voice,coronal,sonorant,continuant],
    u"l": [liquid,lateral,voice,coronal,sonorant,continuant],
#    u"̌l": [liquid,lateral,voice,coronal,sonorant],

    # I'm not sure what this is
    # I think it is a mistranscription, as it is in IPA but not APA
    # u"ɲ": []

    u"ʕ": [back, low, voice,continuant],
    u"-": [syllableBoundary],
    u"##": [wordBoundary],
}


# Automatically annotate vowels
vs = [u"i",u"ɨ",u"ɩ",u"e",u"ə",u"ɛ",u"æ",u"a",u"ʌ",u"u",u"ü",u"ʊ",u"o",u"ö",u"ɔ"]
for k in featureMap:
    features = featureMap[k]
    if k in vs:
        features.append(vowel)

# Include vowel/consonants diacritics
vs = [ k for k in featureMap if vowel in featureMap[k] ]
cs = [ k for k in featureMap if not (vowel in featureMap[k]) ]
for v in vs:
    featureMap[v] += [sonorant]
    featureMap[v] += [continuant]
    featureMap[v + u"́"] = featureMap[v] + [highTone]
    featureMap[v + u"`"] = featureMap[v] + [lowTone]
    featureMap[v + u"¯"] = featureMap[v] + [middleTone]
    featureMap[v + u":"] = featureMap[v] + [longVowel]
    featureMap[v + u"̌"] =  featureMap[v] + [risingTone]
    featureMap[v + u"̃"] = featureMap[v] + [nasal]

# palletization
for p in [u'v',u'b',u't',u'z',u'š',u'l',u'd',u'm',u's',u't^s',u'n',u'r']:
    featureMap[p + u'^y'] = featureMap[p] + [palletized]


def tokenize(word):
    # š can be realized in two different ways
    if u"š" in word:
        print u"ERROR: š should have been purged."
        print "word =",word
        assert False
    # ɲ is valid IPA but is invalid APA
    word = word.replace(u"ɲ", u"ñ")
    # remove all the spaces
    word = word.replace(u" ",u"")
    # not sure what this is but let's remove it
    word = word.replace(u"’",u"")
    originalWord = word
    tokens = []
    while len(word) > 0:
        # Find the largest prefix which can be looked up in the feature dictionary
        for suffixLength in range(len(word)):
            prefixLength = len(word) - suffixLength
            prefix = word[:prefixLength]
            if prefix in featureMap:
                tokens.append(prefix)
                word = word[prefixLength:]
                break
            elif suffixLength == len(word) - 1:
                print word
                print originalWord
                raise Exception(u"No valid prefix: " + word + u" when processing " + originalWord)
    return tokens

class FeatureBank():
   
    """Builds a bank of features and sounds that are specialized to a particular data set.
    The idea is that we don't want to spend time reasoning about features/phonemes that are not attested"""
    mutuallyExclusiveClasses = []#["stop","fricative","affricate"]]
    
    def __init__(self, words):
        self.phonemes = list(set([ p for w in words for p in (tokenize(w) if isinstance(w,unicode) else w.phonemes) ]))
        self.features = list(set([ f for p in self.phonemes for f in featureMap[p] ]))
        self.featureMap = dict([
            (p, list(set(featureMap[p]) & set(self.features)))
            for p in self.phonemes ])
        self.featureVectorMap = dict([
            (p, [ (f in self.featureMap[p]) for f in self.features ])
            for p in self.phonemes ])
        self.phoneme2index = dict([ (self.phonemes[j],j) for j in range(len(self.phonemes)) ])
        self.feature2index = dict([ (self.features[j],j) for j in range(len(self.features)) ])
        self.matrix2phoneme = dict([ (frozenset(featureMap[p]),p) for p in self.phonemes ])

        self.hasSyllables = syllableBoundary in self.features

    @staticmethod
    def fromData(d):
        return FeatureBank([ w for i in d for w in i if w != None ])
        
    def wordToMatrix(self, w):
        return [ self.featureVectorMap[p] for p in tokenize(w) ]
    
    def variablesOfWord(self, w):
        tokens = tokenize(w)
        p2v = dict([ (self.phonemes[j],j) for j in range(len(self.phonemes)) ])
        return [ "phoneme_%d" % p2v[t] for t in tokens ]

    def defineFeaturesToSound(self):
        d = "Sound features2sound(bit[NUMBEROFFEATURES] f){\n"
        for j,p in enumerate(self.phonemes):
            d += "if (f == {%s})" % (",".join(map(str,self.featureVectorMap[p])))
            d += " return phoneme_%d;\n"%j
        d += "assert 0;}\n"
        return d

    def defineSound(self):
        h = "\nstruct Sound{//@Immutable(\"\")\n"
        for f in self.features:
            h += "  bit %s;\n"%(f)
        h += "}\n"
        return h
    def defineVector(self):
        h = "\n#define DEFINEVECTOR struct Vector extends Specification{@Immutable(\"\")\\\n"
        for f in self.features:
            h += "  bit %s_specified; bit %s;\\\n"%(f,f)
        h += "}\n\n"
        h += "\n#define VECTOREQUAL(p,q) (%s)\n"%(" && ".join([ "p.%s_specified == q.%s_specified"%(f,f)
                                                                for f in self.features] + \
                                                              ["(p.%s_specified && p.%s) == (q.%s_specified && q.%s)"%(f,f,f,f)
                                                                for f in self.features]))
        h += "\n#define EMPTYVECTOR(v) (%s)\n"%(" && ".join("v.%s_specified == 0"%f
                                                            for f in self.features))
        h += "\n#define VECTORCOST(v) "
        c = "0"
        for f in self.features:
            c = "validateCost(v.%s_specified + %s)"%(f,c)
        h += c + "\n"

        h += "\n#define VECTORMATCHESSOUND(vector, sound) (%s)\n"%(" && ".join("(!vector.%s_specified || vector.%s == sound.%s)"%(f,f,f)
                                                                               for f in self.features))
        h += "\n#define PROJECTVECTOR(vector, sound)\\\n"
        for f in self.features:
            h += "  bit %s = (!vector.%s_specified && sound.%s) || (vector.%s_specified && vector.%s);\\\n"%(f,f,f,f,f)
        for p in self.phonemes:
            condition = " && ".join("%s%s"%("" if f in featureMap[p] else "!", f)
                                    for f in self.features)
            h += "  if (%s) return phoneme_%d;\\\n"%(condition, self.phoneme2index[p])
        h += "assert 0;\\\n\n"

        h += "\n#define UNKNOWNVECTOR %s\n"%(", ".join( "%s = ??, %s_specified = ??"%(f,f)
                                                        for f in self.features))        
        return h

    def defineZeroFeatures(self):
        z = "#define ZEROFEATURES(m) ({"
        m = "#define MUTUALLYEXCLUDE(s) "
        for f in self.features:
            excluded = False
            for k in FeatureBank.mutuallyExclusiveClasses:
                if f in k:
                    assert not excluded
                    # only retain other members of the class which are actually used in the data
                    kp = set(k) & set(self.features)

                    # mutual exclusion logic
                    m += "if (s.mask[%d]) assert s.preference[%d] "%(self.feature2index[f], self.feature2index[f])
                    m += " && ".join([''] + [ "!s.mask[%d]"%(self.feature2index[e]) for e in kp if e != f ])
                    m += '; '
                    
                    if len(kp) > 1:
                        z += "||".join([ "m[%d]"%(self.feature2index[e]) for e in kp if e != f ])
                        excluded = True
            if not excluded: z += "0"
            z += ", "

        # replace the final, with a }
        return z[:-2] + '})\n' + m

    def __unicode__(self):
        return u'FeatureBank({' + u','.join(self.phonemes) + u'})'
    def __str__(self): return unicode(self).encode('utf-8')

    def sketch(self):
        """Sketches definitions of the phonemes in the bank"""
        for p in self.featureVectorMap:
            for q in self.featureVectorMap:
                if p == q: continue
                if self.featureVectorMap[p] == self.featureVectorMap[q]:
                    print "WARNING: these have the same feature vectors in the bank:",p,q
                    assert False
        
        h = ""
        h += self.defineSound()
        h += self.defineVector()
        if self.hasSyllables:
            h += "#define SYLLABLEBOUNDARYPHONEME phoneme_%d\n"%(self.phoneme2index[u"-"])
            h += "#define SYLLABLEBOUNDARYFEATURE %d\n"%(self.feature2index[syllableBoundary])

        for j in range(len(self.phonemes)):
            features = ",".join("%s = %d"%(f, int(f in featureMap[self.phonemes[j]]))
                for f in self.features)
            h += "Sound phoneme_%d = new Sound(%s);\n" % (j,features)
        h += "#define UNKNOWNSOUND {| %s |}" % (" | ".join(["phoneme_%d"%j for j in range(len(self.phonemes))
                                                            if self.phonemes[j] != u'-' ]))
        h += "\n#define UNKNOWNCONSTANTSPECIFICATION {| %s |}\n" % (" | ".join(["phoneme_%d"%j for j in range(len(self.phonemes)) ]))
        
        #h += self.defineZeroFeatures()
        h += "\n"
        # h += self.defineFeaturesToSound()
        return h

FeatureBank.GLOBAL = FeatureBank(featureMap.keys())
