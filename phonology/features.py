# -*- coding: utf-8 -*-


palletized = "palletized"
sibilant = "sibilant"
sonorant = "sonorant"
coronal = "coronal"
approximate = "approximate"
# because stress and high tones are written the same we model stress as a high tone
# stressed = "stressed"
retroflex = "retroflex"
creaky = "creaky"
risingTone = "risingTone"
highTone = "highTone"
lowTone = "lowTone"
middleTone = "middleTone"
longVowel = "longVowel"
vowel = "vowel"
tense = "tense"
lax = "lax"
high = "high"
middle = "middle"
low = "low"
front = "front"
central = "central"
back = "back"
rounded = "rounded"
#unrounded = "unrounded"
bilabial = "bilabial"
stop = "stop"
voice = "voice"
fricative = "fricative"
labiodental = "labiodental"
dental = "dental"
alveolar = "alveolar"
palatal = "palatal"
labiovelar = "labiovelar"
velar = "velar"
nasal = "nasal"
uvular = "uvular"
laryngeal = "laryngeal"
glide = "glide"
palatal = "palatal"
liquid = "liquid"
lateral = "lateral"
trill = "trill"
flap = "flap"
affricate = "affricate"
alveopalatal = "alveopalatal"
aspirated = "aspirated"
unreleased = "unreleased"
laryngeal = "laryngeal"
pharyngeal = "pharyngeal"
syllableBoundary = "syllableBoundary"

featureMap = {
    # unrounded vowels
    u"i": [voice,tense,high,front],
    u"ɨ": [voice,tense,high,central],
    u"ɯ": [voice,tense,high,back],
    u"ɩ": [voice,lax,high,front],
    u"e": [voice,tense,middle,front],
    u"ə": [voice,tense,middle,central],
    u"ɛ": [voice,lax,middle,front],
    u"æ": [voice,low,front],
    u"a": [voice,low,central],
    u"ʌ": [voice,central,middle,lax],
    # rounded vowels
    u"u": [voice,tense,high,back,rounded],
    u"ü": [voice,tense,high,front,rounded],
    u"ʊ": [voice,lax,high,back,rounded],
    u"o": [voice,tense,middle,back,rounded],
    u"ö": [voice,tense,middle,front,rounded],
    u"ɔ": [voice,lax,middle,back,rounded],
    #possibly missing are umlauts

    # consonance
    u"p": [bilabial,stop],
    u"p|": [bilabial,stop,unreleased],
    u"p^h": [bilabial,stop,aspirated],
    u"b": [bilabial,stop,voice],
    u"f": [labiodental,fricative],
    u"v": [labiodental,fricative,voice],
    u"β": [bilabial,fricative,voice],
    u"m": [bilabial,nasal,voice,sonorant],
    u"m̥": [bilabial,nasal,sonorant],
    u"θ": [dental,fricative,coronal],
    u"d": [alveolar,stop,voice,coronal],
    u"d̪": [dental,stop,voice,coronal],
    u"d^z": [alveolar,affricate,coronal,voice],
    u"t": [alveolar,stop,coronal],
    u"t̪": [dental,stop,coronal],
    u"t|": [alveolar,stop,coronal,unreleased],
    u"t^s": [alveolar,affricate,coronal],
    u"t^h": [alveolar,stop,aspirated,coronal],
    u"ṭ": [retroflex,stop,coronal],
    u"ḍ": [retroflex,stop,coronal,voice],
    u"ð": [dental,fricative,voice,coronal],
    u"z": [alveolar,fricative,voice,coronal, sibilant],
    u"ǰ": [alveopalatal,affricate,voice,coronal],
    u"ǰ|": [alveopalatal,stop,voice,coronal],
    u"ž": [alveopalatal,fricative,voice,coronal, sibilant],
    u"s": [alveolar,fricative,coronal, sibilant],
    u"n": [alveolar,nasal,voice,coronal,sonorant],
    u"ṇ": [retroflex,nasal,voice,sonorant],
    u"n̥": [alveolar,nasal,coronal,sonorant],
    u"ñ": [alveopalatal,nasal,voice,coronal,sonorant],
    u"š": [alveopalatal,fricative,coronal, sibilant],
    u"c": [palatal,stop,coronal],
    u"č": [alveopalatal,affricate,coronal],
    u"č|": [alveopalatal,stop,coronal],
    u"č^h": [alveopalatal,affricate,coronal,aspirated],
    u"k": [velar,stop],
    u"k|": [velar,stop,unreleased],
    u"k^h": [velar,stop,aspirated],
    u"k^y": [velar,stop,palatal],
    u"x": [velar,fricative],
    u"X": [uvular,fricative], # χ
    u"x^y": [velar,fricative,palatal],
    u"g": [velar,stop,voice],
    u"g^y": [velar,stop,voice,palatal],
    u"ɣ": [velar,fricative,voice],
    u"ŋ": [velar,nasal,voice,sonorant],
    u"q": [uvular,stop],
    u"N": [uvular,nasal,voice],
    u"G": [uvular,stop,voice],
    u"ʔ": [laryngeal,stop,sonorant],
    u"h": [laryngeal,fricative,sonorant],
    u"ħ": [pharyngeal,fricative,sonorant],

    # glides
    u"w": [glide,voice,bilabial,sonorant],
    u"y": [glide,palatal,voice,sonorant],

    # liquids
    u"r": [liquid,voice,approximate,alveolar,coronal,sonorant],
    u"ṛ": [retroflex,coronal,voice,sonorant], # I don't really know what features are for this
    u"r̃": [liquid,trill,voice,coronal,sonorant],
    u"r̥̃": [liquid,trill,coronal,sonorant],
    u"ř": [liquid,flap,voice,coronal,sonorant],
    u"l": [liquid,lateral,voice,alveolar,coronal,sonorant],
#    u"̌l": [liquid,lateral,voice,alveolar,coronal,sonorant],

    # I'm not sure what this is
    # I think it is a mistranscription, as it is in IPA but not APA
    # u"ɲ": []

    u"ʕ": [pharyngeal,voice,fricative],
    u"-": [syllableBoundary],
}

# Automatically annotate vowels
for k in featureMap:
    features = featureMap[k]
    # These indicate that it is a vowel
    if front in features or central in features or back in features:
        features.append(vowel)

# Include vowel/consonants diacritics
vs = [ k for k in featureMap if vowel in featureMap[k] ]
cs = [ k for k in featureMap if not (vowel in featureMap[k]) ]
for v in vs:
    featureMap[v] += [sonorant]
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
    mutuallyExclusiveClasses = [["high","middle","low"],
                                ["front","central","back"],
                                ["stop","fricative"]]
    
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

    FSTSYMBOLS = [chr(ord('a') + j) for j in range(26) ] + [chr(ord('A') + j) for j in range(26) ] + [str(j) for j in range(1,10) ]
    def phoneme2fst(self,p):
        return FeatureBank.FSTSYMBOLS[self.phoneme2index[p]]
    def surface2fst(self,s):
        return ''.join([ self.phoneme2fst(p) for p in tokenize(s) ])
    def fst2phoneme(self,s):
        return self.phonemes[FeatureBank.FSTSYMBOLS.index(s)]
    def transducerAlphabet(self):
        return FeatureBank.FSTSYMBOLS[:len(self.phonemes)]

    def sketch(self):
        """Sketches definitions of the phonemes in the bank"""
        for p in self.featureVectorMap:
            for q in self.featureVectorMap:
                if p == q: continue
                if self.featureVectorMap[p] == self.featureVectorMap[q]:
                    print "WARNING: these have the same feature vectors in the bank:",p,q
                    assert False
        
        h = ""
        if self.hasSyllables:
            h += "#define SYLLABLEBOUNDARYPHONEME phoneme_%d\n"%(self.phoneme2index[u"-"])
            h += "#define SYLLABLEBOUNDARYFEATURE %d\n"%(self.feature2index[syllableBoundary])

        for j in range(len(self.phonemes)):
            features = ",".join(map(str,self.featureVectorMap[self.phonemes[j]]))
            h += "Sound phoneme_%d = new Sound(f = {%s});\n" % (j,features)
        h += "#define UNKNOWNSOUND {| %s |}" % (" | ".join(["phoneme_%d"%j for j in range(len(self.phonemes)) ]))
        # This is more for debugging than anything else - common shouldn't use it
        for featureName in self.features:
            h += "\n#define %sFEATURE %d\n" % (featureName.upper(), self.feature2index[featureName])
        h += self.defineZeroFeatures()
        h += "\n"
        h += self.defineFeaturesToSound()
        return h

if __name__ == 'main':
    for p in featureMap:
        for q in featureMap:
            if p == q: continue
            if set(featureMap[p]) == set(featureMap[q]):
                print p,q
