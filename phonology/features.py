# -*- coding: utf-8 -*-

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
    u"i": [tense,high,front,unrounded],
    u"ɨ": [tense,high,central,unrounded],
    u"ɯ": [tense,high,back,unrounded],
    u"ɩ": [lax,high,front,unrounded],
    u"e": [tense,middle,front,unrounded],
    u"ə": [tense,middle,central,unrounded],
    #    u"gamma vowel": [tense,middle,back,unrounded],
    u"ɛ": [lax,middle,front,unrounded],
    u"æ": [low,front,unrounded],
    u"a": [low,central,unrounded],
    # rounded vowels
    u"u": [tense,high,back,rounded],
    u"ü": [tense,high,front,rounded],
    u"ʊ": [lax,high,back,rounded],
    u"o": [tense,middle,back,rounded],
    u"ö": [tense,middle,front,rounded],
    u"ɔ": [lax,middle,back,rounded],
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
    u"θ": [dental,fricative],
    u"d": [alveolar,stop,voice],
    u"t": [alveolar,stop],
    u"th": [alveolar,stop,aspirated],
    u"ṭ": [retroflex,stop],
    u"ð": [dental,fricative,voice],
    u"z": [alveolar,fricative,voice],
    u"ǰ": [alveolar,affricate,voice],
    u"ž": [alveopalatal,fricative,voice],
    u"s": [alveolar,fricative],
    u"n": [alveolar,nasal,voice],
    u"n̥": [alveolar,nasal],
    u"ñ": [alveopalatal,nasal,voice],
    u"š": [alveopalatal,fricative],
    u"c": [palatal,stop],
    u"č": [alveopalatal,affricate],
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
    u"w": [glide,labiovelar,voice],
    u"y": [glide,palatal,voice],

    # liquids
    u"r": [liquid,voice,approximate],
    u"r̃": [liquid,trill,voice],
    u"r̥̃": [liquid,trill],
    u"ř": [liquid,flap,voice],
    u"l": [liquid,lateral,voice],
    u"̌l": [liquid,lateral,voice],

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

FEATURESET = set([ f for p in featureMap for f in featureMap[p] ])
FEATURELIST = list(FEATURESET)

# Map from phoneme to a list of Booleans, one for each element of FEATURELIST
featureVectorMap = {}
for p in featureMap:
    features = featureMap[p]
    featureVectorMap[p] = [ (f in features) for f in FEATURELIST ]
