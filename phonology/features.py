# -*- coding: utf-8 -*-

sonorant = "sonorant"
coronal = "coronal"
approximate = "approximate"
# because stress and high tones are written the same we model stress as a high tone
# stressed = "stressed"
retroflex = "retroflex"
creaky = "creaky"
risingTone = "risingTone"
highTone = "highTone"
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

featureMap = {
    # unrounded vowels
    u"i": [voice,tense,high,front],
    u"ɨ": [voice,tense,high,central],
    u"ɯ": [voice,tense,high,back],
    u"ɩ": [voice,lax,high,front],
    u"e": [voice,tense,middle,front],
    u"ə": [voice,tense,middle,central],
    #    u"gamma vowel": [tense,middle,back],
    u"ɛ": [voice,lax,middle,front],
    u"æ": [voice,low,front],
    u"a": [voice,low,central],
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
    u"p^h": [bilabial,stop,aspirated],
    u"b": [bilabial,stop,voice],
    u"f": [labiodental,fricative],
    u"v": [labiodental,fricative,voice],
    u"β": [bilabial,fricative,voice],
    u"m": [bilabial,nasal,voice,sonorant],
#    u"́m": [bilabial,nasal,voice,stressed,sonorant],
    u"m̥": [bilabial,nasal,sonorant],
    u"θ": [dental,fricative,coronal],
    u"d": [alveolar,stop,voice,coronal],
    u"d^z": [alveolar,affricate,coronal,voice],
    u"t": [alveolar,stop,coronal],
    u"t^s": [alveolar,affricate,coronal],
    u"t^h": [alveolar,stop,aspirated,coronal],
    u"ṭ": [retroflex,stop,coronal],
    u"ḍ": [retroflex,stop,coronal,voice],
    u"ð": [dental,fricative,voice,coronal],
    u"z": [alveolar,fricative,voice,coronal],
    u"ǰ": [alveolar,affricate,voice,coronal],
    u"ž": [alveopalatal,fricative,voice,coronal],
    u"s": [alveolar,fricative,coronal],
    u"n": [alveolar,nasal,voice,coronal,sonorant],
    u"n̥": [alveolar,nasal,coronal,sonorant],
    u"ñ": [alveopalatal,nasal,voice,coronal,sonorant],
    u"š": [alveopalatal,fricative,coronal],
    u"c": [palatal,stop,coronal],
    u"č": [alveopalatal,affricate,coronal],
    u"č^h": [alveopalatal,affricate,coronal,aspirated],
    u"k": [velar,stop],
    u"k^h": [velar,stop,aspirated],
    u"k^y": [velar,stop,palatal],
    u"x": [velar,fricative],
    u"χ": [velar,fricative],
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

    # glides
    u"w": [glide,voice,bilabial,sonorant],
    u"y": [glide,palatal,voice,sonorant],

    # liquids
    u"r": [liquid,voice,approximate,alveolar,coronal,sonorant],
    u"r̃": [liquid,trill,voice,coronal,sonorant],
    u"r̥̃": [liquid,trill,coronal,sonorant],
    u"ř": [liquid,flap,voice,coronal,sonorant],
    u"l": [liquid,lateral,voice,alveolar,coronal,sonorant],
    u"̌l": [liquid,lateral,voice,alveolar,coronal,sonorant],

    # I'm not sure what this is
    # I think it is a mistranscription, as it is in IPA but not APA
    # u"ɲ": []
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
    featureMap[v + u":"] = featureMap[v] + [longVowel]
    featureMap[v + u"̌"] =  featureMap[v] + [risingTone]
    featureMap[v + u"̃"] = featureMap[v] + [nasal]
    # Not sure about these ones
    featureMap[u"̌" + v] =  featureMap[v] + [risingTone]

def tokenize(word):
    # š can be realized in two different ways
    word = word.replace(u"š",u"š")
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
    SPECIALFEATURES = ["vowel","high","middle","low","front","central","back"]
    
    """Builds a bank of features and sounds that are specialized to a particular data set.
    The idea is that we don't want to spend time reasoning about features/phonemes that are not attested"""
    def __init__(self, words):
        self.phonemes = list(set([ p for w in words for p in tokenize(w) ]))
        self.features = list(set([ f for p in self.phonemes for f in featureMap[p] ] + FeatureBank.SPECIALFEATURES))
        self.featureMap = dict([
            (p, list(set(featureMap[p]) & set(self.features)))
            for p in self.phonemes ])
        self.featureVectorMap = dict([
            (p, [ (f in self.featureMap[p]) for f in self.features ])
            for p in self.phonemes ])
        self.phoneme2index = dict([ (self.phonemes[j],j) for j in range(len(self.phonemes)) ])
        self.feature2index = dict([ (self.features[j],j) for j in range(len(self.features)) ])
    def wordToMatrix(self, w):
        return [ self.featureVectorMap[p] for p in tokenize(w) ]
    
    def variablesOfWord(self, w):
        tokens = tokenize(w)
        p2v = dict([ (self.phonemes[j],j) for j in range(len(self.phonemes)) ])
        return [ "phoneme_%d" % p2v[t] for t in tokens ]

    def sketch(self):
        """Sketches definitions of the phonemes in the bank"""
        h = ""
        for j in range(len(self.phonemes)):
            features = ",".join(map(str,self.featureVectorMap[self.phonemes[j]]))
            h += "Sound phoneme_%d = new Sound(f = {%s});\n" % (j,features)
        h += "#define UNKNOWNSOUND {| %s |}" % (" | ".join(["phoneme_%d"%j for j in range(len(self.phonemes)) ]))
        for featureName in FeatureBank.SPECIALFEATURES:
            h += "\n#define %sFEATURE %d\n" % (featureName.upper(), self.feature2index[featureName])
        h += "\n"
        return h
