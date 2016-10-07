# -*- coding: utf-8 -*-

coronal = "coronal"
approximate = "approximate"
stressed = "stressed"
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
unrounded = "unrounded"
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
    u"i": [voice,tense,high,front,unrounded],
    u"ɨ": [voice,tense,high,central,unrounded],
    u"ɯ": [voice,tense,high,back,unrounded],
    u"ɩ": [voice,lax,high,front,unrounded],
    u"e": [voice,tense,middle,front,unrounded],
    u"ə": [voice,tense,middle,central,unrounded],
    #    u"gamma vowel": [tense,middle,back,unrounded],
    u"ɛ": [voice,lax,middle,front,unrounded],
    u"æ": [voice,low,front,unrounded],
    u"a": [voice,low,central,unrounded],
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
    u"ph": [bilabial,stop,aspirated],
    u"b": [bilabial,stop,voice],
    u"f": [labiodental,fricative],
    u"v": [labiodental,fricative,voice],
    u"β": [bilabial,fricative,voice],
    u"m": [bilabial,nasal,voice],
    u"́m": [bilabial,nasal,voice,stressed],
    u"m̥": [bilabial,nasal],
    u"θ": [dental,fricative,coronal],
    u"d": [alveolar,stop,voice,coronal],
    u"t": [alveolar,stop,coronal],
    u"th": [alveolar,stop,aspirated,coronal],
    u"ṭ": [retroflex,stop,coronal],
    u"ð": [dental,fricative,voice,coronal],
    u"z": [alveolar,fricative,voice,coronal],
    u"ǰ": [alveolar,affricate,voice,coronal],
    u"ž": [alveopalatal,fricative,voice,coronal],
    u"s": [alveolar,fricative,coronal],
    u"n": [alveolar,nasal,voice,coronal],
    u"n̥": [alveolar,nasal,coronal],
    u"ñ": [alveopalatal,nasal,voice,coronal],
    u"š": [alveopalatal,fricative,coronal],
    u"c": [palatal,stop,coronal],
    u"č": [alveopalatal,affricate,coronal],
    u"čh": [alveopalatal,affricate,coronal,aspirated],
    u"k": [velar,stop],
    u"kh": [velar,stop,aspirated],
    u"ky": [velar,stop,palatal],
    u"x": [velar,fricative],
    u"χ": [velar,fricative],
    u"xy": [velar,fricative,palatal],
    u"g": [velar,stop,voice],
    u"ɣ": [velar,fricative,voice],
    u"ŋ": [velar,nasal,voice],
    u"q": [uvular,stop],
    u"N": [uvular,nasal,voice],
    u"G": [uvular,stop,voice],
    u"ʔ": [laryngeal,stop],
    u"h": [laryngeal,fricative],

    # glides
    u"w": [glide,voice,bilabial],
    u"y": [glide,palatal,voice],

    # liquids
    u"r": [liquid,voice,approximate,alveolar,coronal],
    u"r̃": [liquid,trill,voice,coronal],
    u"r̥̃": [liquid,trill,coronal],
    u"ř": [liquid,flap,voice,coronal],
    u"l": [liquid,lateral,voice,alveolar,coronal],
    u"̌l": [liquid,lateral,voice,alveolar,coronal],

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
    """Builds a bank of features and sounds that are specialized to a particular data set.
    The idea is that we don't want to spend time reasoning about features/phonemes that are not attested"""
    def __init__(self, words):
        self.phonemes = list(set([ p for w in words for p in tokenize(w) ]))
        self.features = list(set([ f for p in self.phonemes for f in featureMap[p] ]))
        self.featureMap = dict([
            (p, list(set(featureMap[p]) & set(self.features)))
            for p in self.phonemes ])
        self.featureVectorMap = dict([
            (p, [ (f in self.featureMap[p]) for f in self.features ])
            for p in self.phonemes ])
    def wordToMatrix(self, w):
        return [ self.featureVectorMap[p] for p in tokenize(w) ]

    def sketch(self):
        """Sketches definitions of the phonemes in the bank"""
        h = "bit [NUMBEROFFEATURES][%d] phonemes = {\n" % len(self.phonemes)
        h += ",\n".join([ "\t// %s\n\t{%s}" % (featureMap[p],
                                               ",".join(map(str,self.featureVectorMap[p])))
                          for p in self.phonemes ])
        h += "};\n"
        return h


