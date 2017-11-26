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
bilabial = "bilabial"
#stop = "stop"
voice = "voice"
#fricative = "fricative"
labiodental = "labiodental"
dental = "dental"
#alveolar = "alveolar"
labiovelar = "labiovelar"
velar = "velar"
nasal = "nasal"
uvular = "uvular"
laryngeal = "laryngeal"
glide = "glide"
liquid = "liquid"
lateral = "lateral"
trill = "trill"
flap = "flap"
#affricate = "affricate"
alveopalatal = "alveopalatal"
aspirated = "aspirated"
unreleased = "unreleased"
laryngeal = "laryngeal"
pharyngeal = "pharyngeal"
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
    u"p": [bilabial],
    u"p|": [bilabial,unreleased],
    u"p^h": [bilabial,aspirated],
    u"b": [bilabial,voice],
    u"f": [labiodental,continuant],
    u"v": [labiodental,continuant,voice],
    u"β": [bilabial,continuant,voice],
    u"m": [bilabial,nasal,voice,sonorant],#continuant],
    u"m̥": [bilabial,nasal,sonorant],#,continuant],
    u"θ": [dental,continuant,coronal],
    u"d": [voice,coronal],
    u"d̪": [dental,voice,coronal],
    u"d^z": [coronal,voice,delayedRelease],
    u"t": [coronal],
    u"t̪": [dental,coronal],
    u"t|": [coronal,unreleased],
    u"t^s": [coronal,delayedRelease],
    u"t^h": [aspirated,coronal],
    u"ṭ": [retroflex,coronal],
    u"ḍ": [retroflex,coronal,voice],
    u"ṛ": [retroflex,coronal,voice,continuant],
    u"ð": [dental,continuant,voice,coronal],
    u"z": [continuant,voice,coronal, sibilant],
    u"ǰ": [alveopalatal,voice,coronal,sibilant],
    u"ǰ|": [alveopalatal,voice,coronal],
    u"ž": [alveopalatal,continuant,voice,coronal, sibilant],
    u"s": [continuant,coronal, sibilant],
    u"n": [nasal,voice,coronal,sonorant],#continuant],
    u"ṇ": [retroflex,nasal,voice,sonorant],#continuant],
    u"n̥": [nasal,coronal,sonorant],#continuant],
    u"ñ": [alveopalatal,nasal,voice,coronal,sonorant],#continuant],
    u"š": [alveopalatal,continuant,coronal, sibilant],
    u"c": [palatal,coronal], # NOT the same thing as palletized
    u"č": [alveopalatal,coronal,sibilant],
#    u"č|": [alveopalatal,coronal],
    u"č^h": [alveopalatal,coronal,aspirated],
    u"k": [velar],
    u"k|": [velar,unreleased],
    u"k^h": [velar,aspirated],
    u"k^y": [velar,palletized],
    u"x": [velar,continuant],
    u"X": [uvular,continuant], # χ
    u"x^y": [velar,continuant,palletized],
    u"g": [velar,voice],
    u"g^y": [velar,voice,palletized],
    u"ɣ": [velar,continuant,voice],
    u"ŋ": [velar,nasal,voice,sonorant],#continuant],
    u"q": [uvular],
    u"N": [uvular,nasal,voice],#continuant],
    u"G": [uvular,voice],
    u"ʔ": [laryngeal,sonorant,low],
    u"h": [laryngeal,continuant,sonorant,low],
    u"ħ": [pharyngeal,continuant,sonorant],

    # glides
    u"w": [glide,voice,bilabial,sonorant,continuant],
    u"y": [glide,palletized,voice,sonorant,continuant],

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

    u"ʕ": [pharyngeal,voice,continuant],
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
        h += "#define UNKNOWNSOUND {| %s |}" % (" | ".join(["phoneme_%d"%j for j in range(len(self.phonemes))
                                                            if self.phonemes[j] != u'-' ]))
        h += "\n#define UNKNOWNCONSTANTSPECIFICATION {| %s |}" % (" | ".join(["phoneme_%d"%j for j in range(len(self.phonemes)) ]))
        
        # This is more for debugging than anything else - common shouldn't use it
        for featureName in self.features:
            h += "\n#define %sFEATURE %d\n" % (featureName.upper(), self.feature2index[featureName])
        h += self.defineZeroFeatures()
        h += "\n"
        h += self.defineFeaturesToSound()
        return h


