# -*- coding: utf-8 -*-
from features import *

def transposeInflections(inflections):
    return [ tuple([ inflections[x][y] for x in range(len(inflections)) ]) for y in range(len(inflections[0])) ]

class Problem():
    def __init__(self,description,data,parameters = None):
        self.parameters = parameters
        self.description = description
        self.data = data

        # As a sanity check we try to tokenize all of the data
        # This is to make sure that as we define each problem we check to see if it only uses things for which we know the features
        
        for d in self.data:
            if isinstance(d,basestring):
                tokenize(d)
            else:
                for x in d:
                    if x != "~":
                        tokenize(x)
        # If this is an alternation problem
        if parameters and "alternations" in parameters and False:
            ps = set([ p for w in data for p in tokenize(w) ])
            print "Number of distinct phonemes: %d" % len(ps)
            fs = set([ f  for p in ps for f in featureMap[p] ])
            print "Number of distinct features: %d" % len(fs)
            print " ==  ==  == "

# Learning tone patterns
toneProblems = []
toneProblems.append(Problem(u'''
Explain HLM tone pattern.
''',
                            [u"áu`i¯",
                             u"íe`i¯",
                             u"óa`u¯",
                             u"íu`e¯",
                             u"úi`a¯"],
                            {"type": "alternation",
                             "alternations": [dict([ (toned, toned[:-1])
                                                     for toned in featureMap.keys()
                                                     if highTone in featureMap[toned] or
                                                     lowTone in featureMap[toned] or
                                                     middleTone in featureMap[toned] ])]}))

# Chapter 3: alternation problems
alternationProblems = []

alternationProblems.append(Problem(
    u'''
Kikurai
Provide rules to explain the distribution of the consonants [β,r,ɣ] and [b,d,g] in the following data. Accents mark tone: acute is H tone and ‘hacek’ [   ̌] is rising tone.
    [+voice, +fricative] > [+stop, -fricative] / [+nasal] _
    
    [ -laryngeal -glide -nasal -vowel -alveopalatal ] ---> [ -fricative -approximate +stop -sonorant -central -liquid ] / [ +nasal ] _ 
    ''',
    [u"aβaánto",#"people"),
     u"aβamúra",#"young men"),
     u"amahííndi",#"corn cobs"),
     u"amakɛ́ɛ́ndɔ",#"date fruits"),
     u"eβǎ",#"forget"),
     u"eeŋgwé",#"leopard"),
     u"eɣǎ",#"learn"),
     u"ekeβwɛ́",#"fox"),
     u"hoorá",#"thresh"),
     u"iβiɣúrúβe",#"small pigs"),
     u"iβirúúŋgúuri",#"soft porridges"),
     u"uɣusíri",#"huge rope"),
     u"βáinu",#"you (pl)"),
     u"βoryó",#"on the right"),
     u"ičiiŋgɛ́na",#"grinding stones"),
     u"ičiiŋgúrúβe",#"pig"),
     u"ɣaβǎ",#"share!"),
     u"ičiiŋgúta",#"walls"),
     u"βɛrɛká",#"carry a child!"),
     u"iɣitúúmbe",#"stool"),
     u"ɣúúká",#"ancestor"),
     u"remǎ",#"weed!"),
     u"rɛɛntá",#"bring!"),
     u"oβoɣááká",#"male adulthood"),
     u"oβotééndééru",#"smoothness"),
     u"okoɣéémbá",#"to cause rain"),
     u"okoómbára",#"to count me"),
     u"okoβára",#"to count"),
     u"okoóndɔ́ɣa",#"to bewitch me"),
     u"okorɔ́ɣa",#"to bewitch"),
     u"romǎ",#"bite!"),
     u"teɣetá",#"be late!"),
     u"ukuúmbuuryá",#"to ask me"),
     u"uruɣúta"], #"wall"
    {"type": "alternation",
     "alternations": [{u"b": u"β",
                       u"d": u"r",
                       u"g": u"ɣ"}]}))



alternationProblems.append(Problem(
u'''
2: Modern Greek
Determine whether the two segments [k] and [k^y] are contrastive or are governed by rule; similarly, determine whether the difference between [x] and [x^y] is contrastive or predictable. If the distribution is rule-governed, what is the rule and what do you assume to be the underlying consonants in these cases?
Solution:
{x^y,k^y} occur only when there is a front vowel to the right
[ -liquid +velar ] ---> [ +palatal -alveolar -nasal -liquid -voice ] /  _ [ +front ]
''',
    [u"kano",#"do"),
     u"kori",#"daughter"),
     u"xano",		#"lose"),
     u"xori",		#"dances"),
     u"x^yino",		#"pour"),
     u"k^yino",		#"move"),
     u"krima",		#"shame"),
     u"xrima",		#"money"),
     u"xufta",		#"handful"),
     u"kufeta",		#"bonbons"),
     u"kali",		#"charms"),
     u"xali",		#"plight"),
     u"x^yeli",		#"eel"),
     u"k^yeri",		#"candle"),
     u"x^yeri",		#"hand"),
     u"ox^yi"],
    {"type": "alternation",
     "alternations": [{u"k^y": u"k",
                       u"x^y": u"x"}]}))		#"no")

alternationProblems.append(Problem(
u'''
3: Farsi
Describe the distribution of the trill [r̃] and the flap [ř].
Solution found by system:
trill > flap / [ +vowel ] _ [ -alveolar ]
 / [ +unrounded ] _ [ +vowel ]
[ +liquid +voice ] ---> [ -trill -low +flap ] / [ +sonorant ] _ [ -alveolar ]
''',
    [
	u"ær̃teš",#		"army"),
        u"far̃si",#		"Persian"),
	u"qædr̃i",#		"a little bit"),
        u"r̃ah",#		"road"),
	u"r̃ast",#		"right"),
        u"r̃iš",#		"beard"),
	u"ahar̥̃",#		"starch"),
        u"axær̥̃",#		"last"),
	u"hær̃towr̥̃",#	"however"),
        u"šir̥̃",#		"lion"),
	u"ahaři",#		"starched"),
        u"bæřadær̥̃",#	"brother"),
	u"čeřa",#		"why?"),
        u"dařid",#		"you have"),
	u"biřæng",#		"pale"),
        u"šiřini"],#		"pastry")
    {"type": "alternation",
     "alternations": [{u"ř": u"r̃"}]}))

alternationProblems.append(Problem(
    u'''
4: Osage
What rule governs the distribution of [d] versus [ð] in the following data?
    d occurs when there is [a,á] to the right
    ð occurs when there is [i,e] to the right
    ð > d / _ [+low] (my solution)
    d > ð / _ [ +central ] (systems solution)
    [ -alveolar +coronal -tense -front -alveopalatal +voice ] ---> [ -fricative +alveolar +stop -dental ] /  _ [ +low ]
''',
    [u"dábrĩ",#		"three"),
     u"áðik^hã žã",#	"he lay down"),
     u"dačpé",#		"to eat"),
     u"čʔéðe",#		"he killed it"),
     u"dakʔé",#		"to dig"),
     u"ðéze",#		"tongue"),
     u"dálĩ",#		"good"),
     u"ðíe",#		"you"),
     u"daštú",#		"to bite"),
     u"ðíški"],#		"to wash")])
    {"alternations": [{u"ð": u"d"}]}))

alternationProblems.append(Problem(
    u'''
5: Amharic
Is there a phonemic contrast between the vowels [ə] and [ɛ] in Amharic? If not, say what rule governs the distribution of these vowels, and what the underlying value of the vowel is.
"ə" occurs in the contexts:
    {f,r,t,n,g,z,m,d,k,l,b} _ {r,s,n,b,w,d,m,t,g,b,k,č,#}
ɛ occurs in the contexts    :
    {y,š,ž,č,ñ} _ {l,t,g,m,#}
System discovers:
[ -coronal +lax ] ---> [ +tense -front +central -alveopalatal -lax ] / [ -glide -alveopalatal ] _ 
    ''',
    [
	u"fərəs",#		"horse"),
        u"tənəsa",#		"stand up!"),
#	u"yɛlɨš̌lɨš̌",#		"grandchild"),
        u"yɛlɨšlɨš",#		"grandchild"),
        u"mayɛt",#		"see"),
	u"gənzəb",#		"money"),
        u"šɛgna",#		"brave"),
	u"nəñ",#		"I am"),
        u"məwdəd",#	"to like"),
	u"mənnəsat",#	"get up"),
        u"məmkər",#	"advise"),
	u"žɛle",#		"unarmed"),
        u"yɛlləm",#		"no"),
	u"məč",#		"when"),
        u"məst’ət",#		"give"),
	u"fəlləgə",#		"he wanted"),
        u"agəññɛ",#		"he found"),
	u"təməččɛ",#	"it got comfortable"),
        u"mokkərə",#	"he tried"),
	u"k’ažžɛ",#		"he talked in his sleep"),
        u"žɛmmərə",#	"he started"),
	u"lačč’ɛ",#		"he shaved"),
        u"aššɛ",#		"he rubbed"),
	u"bəkk’ələ",#	"it germinated"),
        u"šɛməggələ"],#	"he became old")])
    {"alternations": [{u"ɛ": u"ə"}]}))


alternationProblems.append(Problem(
    u'''
6: Gen
Determine the rule which accounts for the distribution of [r] and [l] in the following data.
    System learns:
    l > r / [ +coronal ] _ [  ]
    My analysis:
l occurs in the context:
    {b,g,ɔ,p,u,a,v,x,h,ŋ,k,#,m,e,w} _
r occurs in the context:
    {s,t,d,č,ñ,z,s,ǰ} _
These are all coronal
[ -middle -nasal +sonorant +coronal ] ---> [ -lateral +approximate ] / [ +coronal ] _     
''',
    [u"agble",#"farm"),
     u"agoŋglo",#"lizard"),
     u"aŋɔli",#"ghost"),
     u"akplɔ",#"spear"),
     u"sabulɛ",#"onion"),
     u"sra",#"strain"),
     u"alɔ",#"hand"),
     u"atitrwɛ",#"red billed wood dove"),
     u"avlɔ",#"bait"),
     u"blafogbe",#"pineapple"),
     u"drɛ",#"stretch arms"),
     u"edrɔ",# "dream"),
     u"exlɔ",#"friend"),
     u"exle",#"flea"),
     u"hlɛ",#"read"),
     u"ŋlɔ",#"write"),
     u"črɔ̃",#"exterminate"),
     u"ñrã",#"be ugly"),
     u"klɔ",#"wash"),
     u"tre",#"glue"),
     u"vlu",#"stretch a rope"),
     u"lɔ",#"like"),
     u"mla",#"pound a drum"),
     u"pleplelu",#"laughing dove"),
     u"wla",#"hide"),
     u"zro",#"fly"),
     u"esrɔ",#"spouse"),
     u"etro",#"scale"),
     u"eñrɔ̃",#"spitting cobra"),
     u"ǰro"],#,   "hint")])
    {"alternations": [{u"l": u"r"}]}))

alternationProblems.append(Problem(
u'''
7: Kishambaa
Describe the distribution of voiced versus voiceless nasals (voiceless nasals are written with a circle under the letter, as in m̥), and voiceless aspirated, voiceless unaspirated and voiced stops in Kishambaa.
Solution found by system:
Nasals become voiced when followed by a voiced phoneme

My analysis:
 ==  ==  == 

Voiced stops occur in the contexts:
{a,m,#,o,n,ŋ} _ {i,u,e,o}
Voiceless stops occur in the contexts:
{#,i,a,o,n̥,m̥} _ {a,o,u,i,e}
These pretty much looks the same so I don't think there is a alternation between voice/voiceless stop

Aspirated stops occur in the contexts:
{n̥,m̥,o} _ {u,e}
Unaspirated stops occur in similar right contexts but don't occur next to voiceless nasals. So I think that they exist underlyingly, and that what we're seeing is that voiceless nasals don't exist underlying.

[ -laryngeal +sonorant -high -low -velar ] ---> [ +voice ] /  _ [ -aspirated ]
''',
    [
	u"tagi",# "egg"),
	u"kitabu",# "book"),
	u"paalika",#"fly!"),
	u"ni",# "it is"),
	u"ŋombe",# "cow"),
	u"matagi",#"eggs"),
	u"dodoa",# "pick up"),
	u"goša",# "sleep!"),
	u"babu",#"skin"),
	u"ndimi",# "tongues"),
	u"ŋgoto",# "heart"),
	u"mbeu",#"seed"),
	u"n̥t^humbii",# "monkey"),
	u"ŋok^huŋguni",# "bedbug"),
	u"m̥p^heho"],#"wind")
    {"alternations": [{u"n̥": u"n",
                      u"m̥": u"m"}]}))


# Problem 8 has Unicode issues
alternationProblems.append(Problem(
    u'''
8: Thai
The obstruents of Thai are illustrated below. Determine what the obstruent phonemes of Thai are ([p, t and k] are unreleased stops). Are [p, t, k] distinct phonemes, or can they be treated as positional variants of some other phoneme? If so, which ones, and what evidence supports your decision? Note that no words begin with [g].
    Solution: the Unicode isn't formatted correctly here, and were not actually seen the problem.
    [ptk] occur only word finally and might be underlying aspirated or not aspirated
    ''',
    [u"bil",#   "Bill"),
     u"müü",#   "hand"),
     u"rak|",#   "love"),
     u"baa",#   "crazy"),
     u"loŋ",#   "go down"),
     u"brüü",#   "extremely fast"),
     u"haa",#   "five"),
     u"plaa",#   "fish"),
     u"dii",#   "good"),
     u"čaan",#   "dish"),
     u"t^hee",#   "pour"),
     u"t^hruumɛɛn",#   "Truman"),
     u"k^hɛŋ",#   "hard"),
     u"panyaa",#   "brains"),
     u"ləəy",#   "pass"),
     u"p^hyaa",#    "[title]"),
     u"lüak|",#   "choose"),
     u"klaaŋ",#   "middle"),
     u"č^hat|",#   "clear"),
     u"traa",#   "stamp"),
     u"riip|",#   "hurry"),
     u"ɔɔk|",#   "exit"),
     u"p^hrɛɛ",#   "silk cloth"),
     u"kiə",#   "wooden shoes"),
     u"k^hwaa",#   "right side"),
     u"kɛɛ",#   "old"),
     u"dray",#   "drive (golf)"),
     u"düŋ",#   "pull"),
     u"kan",#   "ward off"),
     u"čuək|",#   "pure white"),
     u"p^hleeŋ",#   "song"),
     u"č^han",#   "me"),
     u"staaŋ",#   "money"),
     u"rap|",#   "take"),
     u"yiisip|",#   "twenty"),
     u"p^haa",#   "cloth"),
     u"k^haa",#   "kill"),
     u"dam",#   "black"),
     u"raay",#   "case"),
     u"tit|",#   "get stuck"),
     u"sip|",#   "ten"),
     u"pen"],#,   "alive")])
    {"alternations": [{u"p|": u"p",
                       u"t|": u"t",
                       u"k|": u"k"}]}))

alternationProblems.append(Problem(
u'''
9: Palauan
Analyse the distribution of ð, θ and d in the following data. Examples of the type ‘X ~ Y’ mean that the word can be pronounced either as X or as Y, in free variation.
{ð,d} > θ / _#
ð > d / #_, optionally
Systems finds symmetric solution of:
[ +dental -unrounded -liquid -tense -voice ] ---> [ -fricative +alveolar +stop -dental +voice ] / # _ [  ]
todo: incorporate optional rules
''',
    [u"kəðə",	#"we (inclusive)"
     u"bəðuk",	#"my stone"
#     ("~", u"ðiak", u"diak"),	#"negative verb"
     u"ðiak", u"diak",
     u"maθ"	#"eye"
     u"tŋoθ",	#"tattoo needle"
#     ("~", u"ðe:l", u"de:l"),	#"nail"
     u"ðe:l", u"de:l",
#     ("~", u"ðiosəʔ", u"diosəʔ"),#	"place to bathe"),
     u"ðiosəʔ", u"diosəʔ",
#     ("~", u"ðik", u"dik"),#	"wedge"),
     u"ðik", u"dik",
     u"kuθ",	#"louse"
     u"ʔoðiŋəl",	#"visit"
     u"koaθ",	#"visit"
     u"eaŋəθ",	#"sky"
     u"ŋərarəðə",	#"a village"
     u"baθ",	#"stone"
     u"ieðl"	#"mango"
     u"ʔəðip",	#"ant"
     u"kəðeb",	#"short"
     u"məðəŋei",	#"knew"
     u"uðouθ",	#"money"
     u"olðak"],	#"put together"
    {"alternations": [{u"θ": u"d"}]}))

alternationProblems.append(Problem(
    u'''
'10: Quechua (Cuzco dialect)
	Describe the distribution of the following four sets of segments: k, x, q, χ; ŋ, N; i, e; u, o. Some pairs of these segments are allophones (positional variants) of a single segment. You should state which contrasts are phonemic (unpredictable) and which could be predicted by a rule. For segments which you think are positional variants of a single phoneme, state which phoneme you think is the underlying variant, and explain why you think so; provide a rule which accounts for all occurrences of the predictable variant. (Reminder: N is a uvular nasal).
    [ +sonorant +velar ] ---> [ +uvular -low -sonorant -front -liquid -velar ] /  _ [ +uvular ]
    [ -middle +front -liquid +voice ] ---> [ -high +middle ] / [ -palatal -aspirated -nasal ] _ [ -fricative -bilabial ]* [ -glide -nasal -coronal ]
''',
    [
	u"qori",	#"gold"
        u"čoXlu",	#"corn on the cob"
	u"q’omir",	#"green"
        u"niŋri",	#"ear"
	u"moqo",	#"runt"
        u"hoq’ara",	#"deaf"
	u"p^hulyu",	#"blanket"
        u"yuyaŋ",	#"he recalls"
	u"tulyu",	#"bone"
        u"api",	#"take"
	u"suti",	#"name"
        u"oNqoy",	#"be sick!"
	u"čilwi",	#"baby chick"
        u"č^hičiŋ",	#"be whispers"
	u"č^haNqay",	 #"granulate"
        u"aNqosay", 	#"toast"
	u"qečuŋ",	#"he disputes"
        u"p’isqo",	#"bird"
	u"musoX",	#"new"
        u"čuŋka",	#"ten"
	u"yaNqaŋ", 	#"for free"
        u"čulyu",	#"ice"
	u"qhelya",	#"lazy"
        u"q’eNqo",	#"zigzagged"
	u"čeqaŋ",	#"straight"
        u"qaŋ",	#"you"
	u"noqa",	#"I"
        u"čaxra",	#"field"
	u"čeXniŋ",	#"he hates"
        u"soXta",	#"six"
	u"aXna",	#"thus"
        u"lyixlya",	#"small shawl"
	u"qosa",	#"husband"
        u"qara",	#"skin"
	u"alqo",	#"dog"
        u"seNqa",	#"nose"
	u"karu",	#"far"
        u"atoX",	#"fox"
	u"qaŋkuna",	#"you pl."
        u"pusaX",	#"eight"
	u"t’eXway",	#"pluck"
        u"č’aki",	#"dry"
	u"wateX",	#"again"
        u"aŋka",	#"eagle"
	u"waXtay",	#"hit!"
        u"haku",	#"let’s go"
	u"waqay",	#"tears"
        u"kaŋka",	#"roasted"
	u"waxča",	#"poor"
        u"waleX",	#"poor"
	u"t^hakay",	#"drop"
        u"reXsisqa"],#	"known"
    {"alternations": [{u"ŋ": u"N"},
                      {u"o": u"u"},
                      {u"i": u"e"},
                      ]}))


alternationProblems.append(Problem(
    u'''
11: Lhasa Tibetan
	There is no underlying contrast in this language between velars and uvulars, or between voiced or voiceless stops or fricatives (except /s/, which exists underlyingly). State what the underlying segments are, and give rules which account for the surface distribution of these consonant types. [Notational reminder: [G] represents a voiced uvular stop]
    Cornel treated specially!
    uvula context:
    _ {a,o,ɔ,ã,G}
    velar context:
    _ {u,ṭ,ɨ,i,ɩ,ɛ,e,g,b}
    
    {a,o,ɔ} follows uvular [-hi,-front]
    
    voice(less) stop:
    voice context:
    {ŋ,N,m}_{u,a,o}
    voiceless context:
    {ŋ,u,a}

[ -retroflex +velar ] ---> [ +uvular -sonorant -central -velar ] /  _ g* [ -high -stop -front ]    
    ''',
    [
	u"aŋgu",	#"pigeon"
	u"aŋṭãã",	#"a number"
	u"aŋba",	#"duck"
	u"apsoo",	#"shaggy dog"
	u"amčɔɔ",	#"ear"
	u"tukṭüü",	#"poison snake"
	u"amto",	#"a province"
	u"ɨɣu",	#"uncle"
	u"ɨmči",	#"doctor"
	u"uṭɨ",	#"hair"
	u"uβɩɩ",	#"forehead"
	u"ea",	#"bells"
	u"embo",	#"deserted"
	u"ʊʊtsi",	#"oh-oh"
	u"qa",	#"saddle"
	u"qaa",	#"alphabet"
	u"qaŋba",	#"foot"
	u"qamba",	#"pliers"
	u"qam",	#"to dry"
	u"qamtoo",	#"overland"
	u"qaaβo",	#"white"
	u"kɨkṭi",	#"belch"
	u"kɨβu",	#"crawl"
	u"kɨɨŋguu",	#"trip"
	u"kik",	#"rubber"
	u"kiṭuu",	#"student"
	u"kɩɩcuu",	#"translator"
	u"kɩɩrii",	#"roll over"
	u"kiiɣuu",	#"window"
	u"ku",	#"nine"
	u"kupcɨ",	#"900"
	u"kupcaa",	#"chair"
	u"kɛnca",	#"contract"
	u"kɛmbo",	#"headman"
	u"keɣöö",	#"head monk"
	u"kerβa",	#"aristrocrat"
	u"qo",	#"head"
	u"qomba",	#"monastery"
	u"qɔr",	#"coat"
	u"qɔɔɔɔ",	#"round"
	u"č^hea",	#"half"
	u"č^huɣum",	#"cheese"
	u"topcaa",	#"stairs"
	u"t^hoõõ",	#"tonight"
	u"ṭaaãã",	#"post office"
	u"ṭuɣɨ",	#"harbor"
	u"ṭuNGo",	#"China"
	u"nɛNGaa",	#"important"
	u"paNGɔɔ",	#"chest"
	u"pɛɛβãã",	#"frog"
	u"simGãã", #"build a house"
        ],
    {"alternations": [{u"q": u"k",
                       u"G": u"g",
                       u"N": u"ŋ"},
                      {u"b": u"p",
                       u"g": u"k",
                       u"d": u"t",
                       u"ḍ": u"ṭ"}]}))


# Chapter 4
underlyingProblems = []
underlyingProblems.append(Problem(
    '''"
1. Axininca Campa
	Provide underlying representations and a phonological rule which will account for the following alternations.
    Output of system:
Phonological rules:
p ---> w / [  ] _ 
k ---> y / [  ] _ 
    ''',
    [(u"toniro",	u"notoniroti"),
     (u"yaarato",	u"noyaaratoti"),
     (u"kanari",	u"noyanariti"),
     (u"kosiri",	u"noyosiriti"),
     (u"pisiro",	u"nowisiroti"),
     (u"porita",	u"noworitati")
    ]))

underlyingProblems.append(Problem(
    '''
2. Kikuyu
	What is the underlying form of the infinitive prefix in Kikuyu? Give a rule that explains the non-underlying pronunciation of the prefix.
    ko in context:
    {r,o,m,h,ɣ,i}
    ɣo in context:
    {t,k,č}

    System discovers: (todo: bug? see the other solution below)
    ɣ > k / # _ [ -affricate -stop ] o
Final solution:
Morphological analysis:
Inflection 0:	/ ɣ o / + stem + / a /
Phonological rules:
[  ] ---> k / # _ [  ] [ -affricate -stop ]
    ''',
    [(u"ɣotɛŋɛra",),
     (u"ɣokuua",),
     (u"ɣokoora",),
     (u"koruɣa",),
     (u"kooria",),
     (u"komɛɲa",),
     (u"kohɔta",),
     (u"ɣočina",),
     (u"koɣeera",),
     (u"koina",),
     (u"ɣočuuka",),
     (u"ɣokaya",),
     (u"koɣaya",)]))

underlyingProblems.append(Problem(
    '''
3: Korean
	Give the underlying representations of each of the verb stems found below; state what phonological rule applies to these data. [Note: there is a vowel harmony rule which explains the variation between final a and ə in the imperative, which you do not need to be concerned with]
    My analysis:
    Aspirated/un- aspirated distinction exists underlying
    However aspirated consonants become unaspirated followed by k - probably the rule is something like:
           [+aspirated] > [-aspirated] / _ {[-vowel],[-stop],[-son],k}
    There is also a harmony rule something like:
    ə > a / a[-v]*_#
    but you don't need kleene star in this data
    
Final solution:
Morphological analysis:
Inflection 0:	/  / + stem + / ə /
Inflection 1:	/  / + stem + / k o /
Phonological rules:
[  ] ---> [ -aspirated ] /  _ [ -central ]
[  ] ---> a / [ +low ] [  ] _ #
    ''',
    [(u"ipə",		u"ipko"),
     (u"kupə",		u"kupko"),
     (u"kap^ha",		u"kapko"),
     (u"cip^hə",		u"cipko"),
     (u"tata",		u"tatko"),
     (u"put^hə",		u"putko"),
     (u"məkə",		u"məkko"),
     (u"čukə",		u"čukko"),
     (u"ikə",		u"ikko"),
     (u"taka",		u"takko"),
     (u"kaka", u"kakko"),
     (u"səkə",		u"səkko")]))

underlyingProblems.append(Problem(
    '''
4: Hungarian
	Explain what phonological process affects consonants in the following data (a vowel harmony rule makes suffix vowels back after back vowels and front after front vowels, which you do not need to account for). State what the underlying forms are for all morphemes.
    
    My analysis:
    b seems to voice all of the consonants to the left:
    [ ] > [+voice] / _ C* b
    t seems to devoice all of the consonant to the left:
    [ ] > [-voice] / _ C* t
    
    {a,o:} > {e,ö:} / [+front] [ ]* _ [ ] [ ] #
    o: > ö: is [-back +front]
    a > e is [low,central] > [tense,middle,front], matrix [-low,-Central-+tense,]
    so it's becoming a front middle tense vowel
    the rule that it learns is needlessly verbose because it doesn't understand that some features are mutually exclusive
    
    
    
    This is going to be tricky to learn incrementally because a correct spreading rule is only learned for data points where there is also a vowel harmony.
    So you really have to do joint inference over everything.

Morphological analysis:
Inflection 0:	/  / + stem + /  /
Inflection 1:	/  / + stem + / b a n /
Inflection 2:	/  / + stem + / t o: l /
Inflection 3:	/  / + stem + / n a k /
Phonological rules:
[ +vowel ] ---> [ -back +middle -low +tense +front -central ] / [ +front ] [  ]* _ 
[  ] ---> [ +voice ] /  _ [ +bilabial ]
[ -sonorant ] ---> [ -voice ] /  _ [ -voice ]

If it understood the certain features are mutually exclusive, the spreading rule would be:
    [ +vowel ] ---> [ +front +tense +middle ] / [ +front ] [  ]* _ 
    +front implies [-Central -back]
    +middle implies [-low -high]


    ''',
#	noun	in N	from N	to N	gloss
    [
	(u"kalap",	u"kalabban",	u"kalapto:l",	u"kalapnak"), #	hat
	(u"ku:t",	u"ku:dban",	u"ku:tto:l",	u"ku:tnak"), #	well
	(u"ža:k",	u"ža:gban",	u"ža:kto:l",	u"ža:knak"), #	sack
	(u"re:s",	u"re:zben",	u"re:stö:l",	u"re:snek"), #	part
	(u"šro:f ",	u"šro:vban ",	u"šro:fto:l ",	u"šro:fnak"), #	screw
	(u"laka:š",	u"laka:žban",	u"laka:što:l",	u"laka:šnak"), #	apartment
	(u"ketret^s",	u"ketred^zben",	u"ketret^stö:l",	u"ketret^snek"), #	cage
	(u"test",	u"tezdben",	u"testtö:l",	u"testnek"), #	body
	(u"rab",	u"rabban",	u"rapto:l",	u"rabnak"), #	prisoner
	(u"ka:d",	u"ka:dban",	u"ka:tto:l",	u"ka:dnak"), #	tub
	(u"meleg",	u"melegben",	u"melektö:l",	u"melegnek"), #	warm
	(u"vi:z",	u"vi:zben",	u"vi:stö:l",	u"vi:znek"), #	water
	(u"vara:ž",	u"vara:žban",	u"vara:što:l",	u"vara:žnak"), #	magic
	(u"a:g^y",	u"a:g^yban",	u"a:k^yto:l",	u"a:g^ynak"), #	bed
	(u"sem",	u"semben",	u"semtö:l",	u"semnek"), #	eye
	(u"bün",	u"bünben",	u"büntö:l",	u"bünnek"), #	crime
	(u"toroñ",	u"toroñban",	u"toroñto:l",	u"toroñnak"), #	tower
	(u"fal",	u"falban",	u"falto:l",	u"falnak"), #	wall
	(u"ö:r",	u"ö:rben",	u"ö:rtö:l",	u"ö:rnek"), #	guard
	(u"sa:y",	u"sa:yban",	u"sa:yto:l",	u"sa:ynak") #	mouth
    ]))

underlyingProblems.append(Problem(
    '''
5: Kikuria
	Provide appropriate underlying representations and phonological rules which will account for the following data.
    i > e / _ [ ]* e
    u > o / _ [ ]* e
    My analysis:
    [+hi] > [-hi,+mid] / _ [-back]* e
    System discovers:
Morphological analysis:
Inflection 0:	/  / + stem + / a /
Inflection 1:	/  / + stem + / e r a /
Phonological rules:
[ +high ] ---> [ +middle ] /  _ [ -low ]* [ +middle ]
    ''',
	#verb	verb for
	[
            (u"suraaŋga",	u"suraaŋgera"), #	‘praise’
	    (u"taaŋgata",	u"taaŋgatera"), #	‘lead’
	    (u"baamba",	u"baambera"), #	‘fit a drum head’
	    (u"reenda",	u"reendera"), #	‘guard’
	    (u"rema",	u"remera"), #	‘cultivate’	
	    (u"hoora",	u"hoorera"), #	‘thresh’	
	    (u"roma",	u"romera"), #	‘bite’	
	    (u"sooka",	u"sookera"), #	‘respect’	
	    (u"tačora",	u"tačorera"), #	‘tear’
	    (u"siika",	u"seekera"), #	‘close’
	    (u"tiga",	u"tegera"), #	‘leave behind’
	    (u"ruga",	u"rogera"), #	‘cook’
	    (u"suka",	u"sokera"), #	‘plait’
	    (u"huuta",	u"hootera"), #	‘blow’
	    (u"riiŋga",	u"reeŋgera"), #	‘fold’
	    (u"siinda",	u"seendera")])) #	‘win’

underlyingProblems.append(Problem(
    '''
6: Farsi
Give the underlying forms for the following nouns, and say what phonological rule is necessary to explain the following data.
    
    Final solutions:
Morphological analysis:
Inflection 0:	/  / + stem + /  /
Inflection 1:	/  / + stem + / a n /
Phonological rules:
[  ] ---> Ø / [ +middle ] _ #

My analysis:
    +gan occurs only after /e/
    but not after every one:
     mæleke-an
     valede-an
     kæbire-an
     hamele-an
    contrast with:
     bačče-gan
     setare-gan
     bænde-gan
     azade-gan
     divane-gan

    ''',
	#singular	plural	gloss
    [
	(u"zæn",	u"zænan"), #	woman
	(u"læb",	u"læban"), #	lip
	(u"hæsud",	u"hæsudan"), #	envious
	(u"bæradær",	u"bæradæran"), #	brother
	(u"bozorg",	u"bozorgan"), #	big
	(u"mæleke",	u"mælekean"), #	queen
	(u"valede",	u"valedean"), #	mother
	(u"kæbire",	u"kæbirean"), #	big
	(u"ahu",	u"ahuan"), #	gazelle
	(u"hamele",	u"hamelean"), #	pregnant
	(u"bačče",	u"baččegan"), #	child
	(u"setare",	u"setaregan"), #	star
	(u"bænde",	u"bændegan"), #	slave
	(u"azade",	u"azadegan"), #	freeborn
	(u"divane",	u"divanegan")])) #	insane

underlyingProblems.append(Problem(
    '''7: Tibetan
Numbers between 11 and 19 are formed by placing the appropriate digit after the number 10, and multiples of 10 are formed by placing the appropriate multiplier before the number 10. What are the underlying forms of the basic numerals, and what phonological rule is involved in accounting for these data?
Final solution:
[ -nasal ] ---> Ø / # _ 
    ''',
    [
	u"ǰu",#	‘10’
	u"ǰig",#	‘1’
        u"ǰugǰig",#	‘11’
	u"ši",#	‘4’
	u"ǰubši",#	‘14’
        u"šibǰu",#	‘40’
	u"gu",#	‘9’
	u"ǰurgu",#	‘19’
	u"gubǰu",#	‘90’
	u"ŋa",#	‘5’
	u"ǰuŋa",#	‘15’
	u"ŋabǰu"],#	‘50’
    parameters = [10,1,11,4,14,40,9,19,90,5,15,50]))

underlyingProblems.append(Problem(
    '''
8: Makonde
Explain what phonological rules apply in the following examples (the acute accent in these example marks stress, whose position is predictable).
    My analysis:
    Second to last vowel is always stressed:
    V > [+stress] / _C*V#
    There's some kind of vowel harmony:
    o~a
    e~a
    {i,u} are unaffected
    Probably {o,e} exist underlying
    [+mid,-stress] > a
    So the stress rule applies first, and then if {o,e} was unstressed then it gets neutralized to [a]
    The system learns:
    Final solution:
Morphological analysis:
Inflection 0:	/  / + stem + / á ŋ g a /
Inflection 1:	/  / + stem + / í l e /
Inflection 2:	/  / + stem + / a /
Phonological rules:
[  ] ---> [ -highTone ] /  _ [  ]* [ +highTone ]
[ +middle -highTone ] ---> a /  _ [  ]
    ''',
	#repeated	past		imperative	gloss
	#imperative
    [
	(u"amáŋga",	u"amíle",		u"áma"),#		move
	(u"taváŋga",	u"tavíle",		u"táva"),#		wrap
	(u"akáŋga",		u"akíle",		u"áka"),#		hunt
	(u"patáŋga",	u"patíle",		u"póta"),#		twist
	(u"tatáŋga",		u"tatíle",		u"tóta"),#		sew
	(u"dabáŋga",	u"dabíle",		u"dóba"),#		get tired
	(u"aváŋga",		u"avíle",		u"óva"),#		miss
	(u"amáŋga",	u"amíle",		u"óma"),#		pierce
	(u"tapáŋga",	u"tapíle",		u"tépa"),#		bend
	(u"patáŋga",	u"patíle",		u"péta"),#		separate
	(u"aváŋga",		u"avíle",		u"éva"),#		separate
	(u"babáŋga",	u"babíle",		u"béba"),#		hold like a baby
	(u"utáŋga",		u"utíle",		u"úta"),#		smoke
	(u"lukáŋga",	u"lukíle",		u"lúka"),#		plait
	(u"lumáŋga",	u"lumíle",		u"lúma"),#		bite
	(u"uŋgáŋga",	u"uŋgíle",		u"úŋga"),#		tie
	(u"iváŋga",		u"ivíle",		u"íva"),#		steal
	(u"pitáŋga",		u"pitíle",		u"píta"),#		pass
	(u"imbáŋga",	u"imbíle",		u"ímba"),#		dig
	(u"limáŋga",	u"limíle",		u"líma")]))#		cultivate

underlyingProblems.append(Problem(
    '''
9: North Saami
Posit appropriate underlying forms and any rules needed to explain the following alternations. The emphasis heret should be on correctly identifying the underlying form: the exact nature of the changes seen here is a more advanced problem.
    My analysis:
    {h,g,b,ð} > t / _ #
    {ǰ} > š / _ #
    m > n / _ #
    Not affected: s
    ''',
	#Nominative sg.	Essive
    [
	(u"varit",	u"varihin"),#	“2 year old reindeer buck”
	(u"oahpis",	u"oahpisin"),#	“acquaintance”
	(u"čoarvuš",	u"čoarvušin"),#	“antlers & skullcap”
	(u"lottaaš",	u"lottaaǰin"),#	“small bird”
	(u"čuoivvat",	u"čuoivvagin"),#	“yellow-brown reindeer”
	(u"ahhkut",	u"ahhkubin"),#	“grandchild of woman”
	(u"suohkat",	u"suohkaðin"),#	“thick”
	(u"heeǰoš",	u"heeǰoǰin"),#	“poor guy”
	(u"aaǰǰut",	u"aaǰǰubin"),#	“grandchild of man”
	(u"bissobeahtset",	u"bissobeahtsehin"),#	“butt of gun”
	(u"čeahtsit",	u"čeahtsibin"),#	“children of elder brother of man”
	(u"yaaʔmin",	u"yaaʔmimin"),#	“death”
	(u"čuoivat",	u"čuoivagin"),#	“yellow-grey reindeer”
	(u"laageš",	u"laageǰin"),#	“mountain birch”
	(u"gahpir",	u"gahpirin"),#	“cap”
	(u"gaauhtsis",	u"gaauhtsisin"),#	“8 people”
	(u"aaslat",	u"aaslagin"),#	man’s name
	(u"baðoošgaattset",	u"baðoošgaattsebin"),#	“bird type”
	(u"ahhkit",	u"ahhkiðin"),#	“boring”
	(u"bahaanaalat",	u"bahaanaalagin"),#	“badly behaved”
	(u"beštor",	u"beštorin"),#	“bird type”
	(u"heevemeahhtun",	u"heevemeahhtunin"),#	“inappropriate”
	(u"beeǰot",	u"beeǰohin"),#	“white reindeer”
	(u"bissomeahtun",	u"bissomeahtumin"),#	“unstable”
	(u"laðas",	u"laðasin"),#	“something jointed”
	(u"heaiyusmielat",	u"heaiyusmielagin"),#	“unhappy”
	(u"heaŋkkan",	u"heaŋkkanin"),#	“hanger”
	(u"yaman",	u"yamanin")]))#	“something that makes noise”

underlyingProblems.append(Problem(
    '''
    Samoan: example from the textbook.
    ''',
    [
        (u"olo", u"oloia"),
        (u"lafo",u"lafoia"),
        (u"usu",u"usuia"),
        (u"taui",u"tauia"),
        (u"naumati",u"naumatia"),
        (u"lele",u"lelea"),
        (u"tafe",u"tafea"),
        (u"palepale",u"palepalea")
    ]))

underlyingProblems.append(Problem(
    '''
    Russian: devoicing of word final obscurant
    ''',
    [
        (u"vagon", u"vagona"),
        (u"glas", u"glaza"),
        (u"golos",u"golosa"),
        (u"ras", u"raza"),
        (u"les",u"lesa"),
        (u"porok",u"poroga"),
        (u"vrak",u"vraga"),
        (u"urok",u"uroka"),
        (u"tvet",u"tveta"),
        (u"prut",u"pruda"),
        (u"soldat",u"soldata"),
        (u"zavot",u"zavoda"),
        (u"xlep",u"xleba"),
        (u"grip",u"griba"),
        (u"trup",u"trupa")
    ]))

underlyingProblems.append(Problem(
    '''
    English verb inflections.
    ''',
    [(u"ro",u"rod",u"roz"),
     (u"lʊk",u"lʊkt",u"lʊks"),
     (u"æsk",u"æskt",u"æsks"),
     (u"wɛrk",u"wɛrkt",u"wɛrks"),
     (u"sim",u"simd",u"simz"),
     (u"liv",u"livd",u"livz"),
     (u"həg",u"həgd",u"həgz"),
     (u"kɩs",u"kɩst",u"kɩsəz"),
     (u"fɩš",u"fɩšt",u"fɩšəz"),
     (u"wet",u"wetəd",u"wets"),
     (u"græb",u"græbd",u"græbz")]))

interactingProblems = []

interactingProblems.append(Problem(
    '''1: Kerewe

What two tone rules are motivated by the following data; explain what order the rules apply in.
    Final solution:
Morphological analysis:
Inflection 0:	/ k u / + stem + / a /
Inflection 1:	/ k u / + stem + / a n a /
Inflection 2:	/ k u / + stem + / i l a /
Inflection 3:	/ k u / + stem + / i l a n a /
Inflection 4:	/ k u t ú / + stem + / a /
Inflection 5:	/ k u k í / + stem + / a /
Inflection 6:	/ k u t ú / + stem + / i l a /
Inflection 7:	/ k u k í t ú / + stem + / i l a /
Phonological rules:
[  ] ---> [ -highTone ] / [ +highTone ] [  ] _ 
[  ] ---> [ +highTone ] / [ +highTone ] [  ] _ [  ]
    ''',
    #to V	to V e.o	to V for	to V for e.o	to V us	to V it	to V for us	to V it for us
	[
            (u"kubala",	u"kubalana",	u"kubalila",	u"kubalilana", u"kutúbála",	u"kukíbála",	u"kutúbálila",	u"kukítúbalila"),#	“count”
	    (u"kugaya",	u"kugayana",	u"kugayila",	u"kugayilana", u"kutúgáya",	u"kukígáya",	u"kutúgáyila",	u"kukítúgayila"),#	“despise”
	    (u"kugula",	u"kugulana",	u"kugulila",	u"kugulilana", u"kutúgúla",	u"kukígúla",	u"kutúgúlila",	u"kukítúgulila"),#	“buy”
	    (u"kubála",	u"kubálána",	u"kubálíla",	u"kubálílana", u"kutúbála",	u"kukíbála",	u"kutúbálila",	u"kukítúbalila"),#	“kick”
	    (u"kulúma",	u"kulúmána",	u"kulúmíla",	u"kulúmílana", u"kutúlúma",	u"kukílúma",	u"kutúlúmila",	u"kukítúlumila"),#	“bite”
	    (u"kusúna",	u"kusúnána",	u"kusúníla",	u"kusúnílana", u"kutúsúna",	u"kukísúna",	u"kutúsúnila",	u"kukítúsunila"),#	“pinch”
	    (u"kulába",	u"kulábána",	u"kulábíla",	u"kulábílana", u"kutúlába",	u"kukílába",	u"kutúlábila",	u"kukítúlabila")]))#	“pass”

interactingProblems.append(Problem(
    '''2: Polish

What phonological rules are motivated by the following examples, and what order do those rules apply in?
Discovered by the system:
Final solution:
Morphological analysis:
Inflection 0:	/  / + stem + /  /
Inflection 1:	/  / + stem + / i /
Phonological rules:
o ---> u /  _ [ -nasal +voice ] #
[ -sonorant ] ---> [ -voice ] /  _ #
''',
    #singular	plural		singular	plural
	[
            (u"klup",	u"klubi"),#	‘club’
            (u"trup",	u"trupi"),#	‘corpse’
	    (u"dom",	u"domi"),#	‘house’
            (u"snop",	u"snopi"),#	‘sheaf’
	    (u"žwup",	u"žwobi"),#	‘crib’
            (u"trut",	u"trudi"),#	‘labor’
	    (u"dzvon",	u"dzvoni"),#	‘bell’
            (u"kot",	u"koti"),#	‘cat’
	    (u"lut",	u"lodi"),#	‘ice’
            (u"grus",	u"gruzi"),#	‘rubble’
	    (u"nos",	u"nosi"),#	‘nose’
            (u"vus",	u"vozi"),#	‘cart’
	    (u"wuk",	u"wugi"),#	‘lye’
            (u"wuk",	u"wuki"),#	‘bow’
	    (u"sok",	u"soki"),#	‘juice’
            (u"ruk",	u"rogi"),#	‘horn’
	    (u"bur",	u"bori"),#	‘forest’
            (u"vuw",	u"vowi"),#	‘ox’
	    (u"sul",	u"soli"),#	‘salt’
            (u"buy",	u"boyi"),#	‘fight’
	    (u"šum",	u"šumi"),#	‘noise’
            (u"žur",	u"žuri")]))#	‘soup’

interactingProblems.append(Problem(
    '''3: Ancient Greek

Discuss the phonological rules and underlying representations which are necessary to account for the following nouns; comment on the ordering of these phonological processes.

Greedy search discovers
    [ -sonorant ] ---> [ -aspirated -voice ] /  _ [ -voice ]
    So we unaspirate whenever the next sound is unvoiced, which bizarrely seems to work.
    obstructions also become you voiced whenever the next sound is unvoiced
it also looks like coronal is deleted in certain contexts.
    deleted after /s/?
    deleted: {t,d,n} / _ s
    ''',
#	nom. sg.	gen. sg.	dat. sg	dat. pl.
    [
	(u"hals",	u"halos",	u"hali",	u"halsi"),#	‘salt’
	(u"oys",	u"oyos",	u"oyi",	u"oysi"),#	‘sheep’
	(u"sus",	u"suos",	u"sui",	u"susi"),#	‘sow’
	(u"klo:ps",	u"klo:pos",	u"klo:pi",	u"klo:psi"),#	‘thief’
	(u"p^hle:ps",	u"p^hle:bos",	u"p^hle:bi",	u"p^hle:psi"),#	‘vein’
	(u"kate:lips",	u"kate:lip^hos",	u"kate:lip^hi",	u"kate:lipsi"),#	‘upper story’
	(u"p^hulaks",	u"p^hulakos",	u"p^hulaki",	u"p^hulaksi"),#	‘guard’
	(u"ayks",	u"aygos",	u"aygi",	u"ayksi"),#	‘goat’
	(u"salpiŋks",	u"salpiŋgos",	u"salpiŋgi",	u"salpiŋksi"),#	‘trumpet’
	(u"onuks",	u"onuk^hos",	u"onuk^hi",	u"onuksi"),#	‘nail’
	(u"t^he:s",	u"t^he:tos ",	u"t^he:ti ",	u"t^he:si"),#	‘serf’
	(u"k^haris",	u"k^haritos",	u"k^hariti",	u"k^harisi"),#	‘grace’
	(u"elpis",	u"elpidos",	u"elpidi",	u"elpisi"),#	‘hope’
	(u"korus",	u"korut^hos",	u"korut^hi",	u"korusi"),#	‘helmet’
	(u"ri:s",	u"ri:nos",	u"ri:ni",	u"ri:si"),#	‘nose’
	(u"delp^hi:s",	u"delp^hi:nos",	u"delp^hi:ni",	u"delp^hi:si")]))#	‘porpoise’

'''
Problem(
4: Shona
	Acute accent indicates H tone and unaccented vowels have L tone. Given the two sets of data immediately below, what tone rule do the following data motivate? There are alternations in the form of adjectives, e.g. kurefú, karefú, marefú all meaning “long”. Adjectives have an agreement prefix, hence ku-refú marks the form of the adjective in one grammatical class, and so on. In some cases, the agreement is realized purely as a change in the initial consonant of the adjective, i.e. gúrú ~ kúrú ~ húrú which need not be explained.

	bveni	‘baboon’	bveni pfúpi	‘short baboon’
	táfura	‘table’	táfura húrú	‘big table’
	šoko	‘word’	šoko bvúpi	‘short word’
	ɓadzá	‘hoe’	ɓadzá gúrú	‘big hoe’
	zigómaná	‘boy (aug.)’	zigómaná gúrú	‘big boy (aug.)’
	imbá	‘house’	imbá čéna	‘clean house’
	mhará	‘gazelle’	mhará čéna	‘clean gazelle’
	marí	‘money’	marí čéna	‘clean money’

	ɓáŋgá	‘knife’	ɓáŋga gúrú	‘big knife’
	ɗémó	‘axe’	ɗémo bvúpi	‘short axe’
	nhúmé	‘messenger’	nhúme pfúpi	‘short messenger’
	ǰírá	‘cloth’	ǰíra ǰéna	‘clean cloth’
	hárí	‘pot’	hári húrú	‘big pot’
	mbúndúdzí	‘worms’	mbúndúdzi húrú	‘big worms’
	fúma	‘wealth’	fúma čéna	‘clean wealth’
	nyíka	‘country’	nyíka húrú	‘big country’
	hákáta	‘bones’	hákáta pfúpi	‘short bones’
	ǰékéra	‘pumpkin’	ǰékéra gúrú	‘big pumpkin’

These data provide further illustration of the operation of this tone rule, which will help you to state the conditions on the rule correctly.

	guɗo	‘baboon’	gudo rákafá	‘the baboon died’
	ɓadzá	‘hoe’	ɓadzá rákawá	‘the hoe fell’
	nuŋgú	‘porcupine’	nuŋgú yákafá	‘the porcupine died’
	ɓáŋgá	‘knife’	ɓáŋga rákawá	‘the knife fell’
	nhúmé	‘messenger’	nhúme yákafá	‘the messenger died’
	búku	‘book’	búku rákawá	‘the book fell’
	mapfeni	‘baboons’	mapfeni makúrú	‘big baboons’
	mapadzá	‘hoes’	mapadzá makúrú	‘big hoes’
	mapáŋgá	‘knives’	mapáŋgá makúrú	‘big knives’
	nhúmé	‘messenger’	nhúmé ndefú	‘short messenger’
	matémó	‘axes’	matémó mapfúpi	‘short axes’
	mabúku	‘books’	mabúku mažínǰí	‘many books’
	čitóro	‘store’	čitóro čikúrú	‘big store’

	In the examples below, a second tone rule applies.

	guɗo	‘baboon’	guɗo refú	‘tall baboon’
	búku	‘book’	búku refú	‘long book’
	ɓadzá	‘hoe’	ɓadzá refú	‘long hoe’
	nuŋgú	‘porcupine’	nuŋgú ndefú	‘long porcupine’
	mašoko	‘words’	mašoko marefú	‘long words’
	kunyíka	‘to the land’	kunyíka kurefú	‘to the long land’
	mapadzá	‘hoes’	mapadzá márefú	‘long hoes’
	kamhará	‘gazelle (dim.)’	kamhará kárefú	‘long gazelle (dim.)’
	tunuŋgú	‘porcupines (dim.)’	tunuŋgú túrefú	‘long porcupines (dim.)’

	guɗo	‘baboon’	guɗo gobvú	‘thick baboon’
	búku	‘book’	búku gobvú	‘thick book’
	ɓadzá	‘hoe’	ɓadzá gobvú	‘thick hoe’
	makuɗo	‘baboons’	makuɗo makobvú	‘thick baboons’
	mapadzá	‘hoes’	mapadzá mákobvú	‘thick hoes’
	tsamba	‘letter’	tsamba nhete	‘thin letter’
	búku	‘book’	búku ɗete	‘thin book’
	ɓadzá	‘hoe’	badzá ɗéte	‘thin hoe’
	imbá	‘house’	imbá nhéte	‘thin house’

	What do the following examples show about these tone rules?

	ɓáŋgá	‘knife’	ɓáŋgá ɗéte	‘thin knife’
	ɗémó	‘axe’	ɗémó ɗéte	‘thin axe’
	murúmé	‘person’	murúmé mútete	‘thin person’
	kahúní	‘firewood (dim.)’	kahúní kárefú	‘long firewood’
	mačírá	‘clothes’	mačírá márefú	‘long clothes’
	hárí	‘pot’	hárí nhéte	‘thin pot’
'''

Problem(
'''5: Catalan

Give phonological rules which account for the following data, and indicate what ordering is necessary between these rules. For each adjective stem, state what the underlying form of the root is. Pay attention to the difference between surface [b,d,g] and [β,ð,ɣ], in terms of predictability.
''',		
#	masc	fem		masc	fem	
#	sing.	sing.		sing.	sing.
    [
	(u"əkely",	u"əkelyə"),#	‘that’
	(u"mal",	u"malə"),#	‘bad’
	(u"siβil",	u"siβilə"),#	‘civil’
	(u"əskerp",	u"əskerpə"),#	‘shy’
	(u"šop",	u"šopə"),#	‘drenched’
	(u"sɛk",	u"sɛkə"),#	‘dry’
	(u"əspɛs",	u"əspɛsə"),#	‘thick’
	(u"gros",	u"grosə"),#	‘large’
	(u"baš",	u"bašə"),#	‘short’
	(u"koš",	u"košə"),#	‘lame’
	(u"tot",	u"totə"),#	‘all’
	(u"brut",	u"brutə"),#	‘dirty’
	(u"pɔk",	u"pɔkə"),#	‘little’
	(u"prəsis",	u"prəsizə"),#	‘precise’
	(u"frənses",	u"frənsezə"),#	‘French’
	(u"gris",	u"grizə"),#	‘grey’
	(u"kəzat",	u"kəzaðə"),#	‘married’
	(u"bwit",	u"bwiðə"),#	‘empty’
	(u"rɔč",	u"rɔžə"),#	‘red’
	(u"boč",	u"božə"),#	‘crazy’
	(u"orp",	u"orβə"),#	‘blind’
	(u"lyark",	u"lyarɣə"),#	‘long’
	(u"sek",	u"seɣə"),#	‘blind’
	(u"fəšuk",	u"fəšuɣə"),#	‘heavy’
	(u"grok",	u"groɣə"),#	‘yellow’
	(u"puruk",	u"puruɣə"),#	‘fearful’
	(u"kandit",	u"kandiðə"),#	‘candid’
	(u"frɛt",	u"frɛðə"),#	‘cold’
	(u"səɣu",	u"səɣurə"),#	‘sure’
	(u"du",	u"durə"),#	‘hard’
	(u"səɣəðo",	u"səɣəðorə"),#	‘reaper’
	(u"kla",	u"klarə"),#	‘clear’
	(u"nu",	u"nuə"),#	‘nude’
	(u"kru",	u"kruə"),#	‘raw’
	(u"flɔñǰu",	u"flɔñǰə"),#	‘soft’
	(u"dropu",	u"dropə"),#	‘lazy’
	(u"əgzaktə",	u"əgzaktə"),#	‘exact’
	(u"əlβi",	u"əlβinə"),#	‘albino’
	(u"sa",	u"sanə"),#	‘healthy’
	(u"pla",	u"planə"),#	‘level’
	(u"bo",	u"bonə"),#	‘good’
	(u"sərɛ",	u"sərɛnə"),#	‘calm’
	(u"suβlim",	u"suβlimə"),#	‘sublime’
	(u"al",	u"altə"),#	‘tall’
	(u"fɔr",	u"fɔrtə"),#	‘strong’
	(u"kur",	u"kurtə"),#	‘short’
	(u"sor",	u"sorðə"),#	‘deaf’
	(u"bɛr",	u"bɛrðə"),#	‘green’
	(u"san",	u"santə"),#	‘saint’
	(u"kəlɛn",	u"kəlɛntə"),#	‘hot’
	(u"prufun",	u"prufundə"),#	‘deep’
	(u"fəkun",	u"fəkundə"),#	‘fertile’
	(u"dəsen",	u"dəsentə"),#	‘decent’
	(u"dulen",	u"dulentə"),#	‘bad’
	(u"əstuðian",	u"əstuðiantə"),#	‘student’
	(u"blaŋ",	u"blaŋkə")])#	‘white’

Problem(
    '''6: Finnish
Propose rules which will account for the following alternations. It would be best not to write a lot of rules which go directly from underlying forms to surface forms in one step; instead, propose a sequence of rules whose combined effect brings about the change in the underlying form. Pay attention to what consonants actually exist in the language.
    ''',
    [
	#genitive	nom.	nom.	ablative	essive	gloss
	#sing.	sing.	pl.	sing.	sing.	
	(u"kanadan",	u"kanada",	u"kanadat",	u"kanadalta",	u"kanadana"),#	Canada
	(u"kiryan",	u"kirya",	u"kiryat",	u"kiryalta",	u"kiryana"),#	book
	(u"aamun",	u"aamu",	u"aamut",	u"aamulta",	u"aamuna"),#	morning
	(u"talon",	u"talo",	u"talot",	u"talolta",	u"talona"),#	house
	(u"koiran",	u"koira",	u"koirat",	u"koiralta",	u"koirana"),#	dog
	(u"hüvæn",	u"hüvæ",	u"hüvæt",	u"hüvæltæ",	u"hüvænæ"),#	good
	(u"kuvan",	u"kuva",	u"kuvat",	u"kuvalta",	u"kuvana"),#	picture
	(u"lain",	u"laki",	u"lait",	u"lailta",	u"lakina"),#	roof
	(u"nælæn",	u"nælkæ",	u"nælæt",	u"nælæltæ",	u"nælkænæ"),#	hunger
	(u"yalan",	u"yalka",	u"yalat",	u"yalalta",	u"yalkana"),#	leg
	(u"leuan",	u"leuka",	u"leuat",	u"leualta",	u"leukana"),#	chin
	(u"paran",	u"parka",	u"parat",	u"paralta",	u"parkana"),#	poor
	(u"reiæn",	u"reikæ",	u"reiæt",	u"reiæltæ",	u"reikænæ"),#	hole
	(u"nahan",	u"nahka",	u"nahat",	u"nahalta",	u"nahkana"),#	hide
	(u"vihon",	u"vihko",	u"vihot",	u"viholta",	u"vihkona"),#	notebook
	(u"laihan",	u"laiha",	u"laihat",	u"laihalta",	u"laihana"),#	lean
	(u"avun",	u"apu",	u"avut",	u"avulta",	u"apuna"),#	help 
	(u"halvan",	u"halpa",	u"halvat",	u"halvalta",	u"halpana"),#	cheap
	(u"orvon",	u"orpo",	u"orvot",	u"orvolta",	u"orpona"),#	orphan
	(u"leivæn",	u"leipæ",	u"leivæt",	u"leivæltæ",	u"leipænæ"),#	bread
	(u"pæivæn",	u"pæivæ",	u"pæivæt ",	u"pæivæltæ ",	u"pæivænæ"),#	day
	(u"kilvan",	u"kilpa",	u"kilvat",	u"kilvalta",	u"kilpana"),#	competition
	(u"külvün",	u"külpü",	u"külvüt",	u"külvültæ",	u"külpünæ"),#	bath
	(u"tavan",	u"tapa",	u"tavat",	u"tavalta",	u"tapana"),#	manner
	(u"korvan",	u"korva",	u"korvat",	u"korvalta",	u"korvana"),#	ear
	(u"æidin",	u"æiti",	u"æidit",	u"æidiltæ",	u"æitinæ"),#	mother
	(u"kodin",	u"koti",	u"kodit",	u"kodilta",	u"kotina"),#	home
	(u"muodon",	u"muoto",	u"muodot",	u"muodolta",	u"muotona"),#	form
	(u"tædin",	u"tæti",	u"tædit",	u"tædiltæ",	u"tætinæ"),#	aunt
	(u"kadun",	u"katu",	u"kadut",	u"kadulta",	u"katuna"),#	street
	(u"maidon",	u"maito",	u"maidot",	u"maidolta",	u"maitona"),#	milk
	(u"pöüdæn",	u"pöütæ",	u"pöüdæt",	u"pöüdæltæ",	u"pöütænæ"),#	table
	(u"tehdün",	u"tehtü",	u"tehdüt",	u"tehdültæ",	u"tehtünæ"),#	made
	(u"læmmön",	u"læmpö",	u"læmmöt",	u"læmmöltæ",	u"læmpönæ"),#	warmth
	(u"laŋŋan",	u"laŋka",	u"laŋŋat",	u"laŋŋalta",	u"laŋkana"),#	thread
	(u"sæŋŋün",	u"sæŋkü",	u"sæŋŋüt",	u"sæŋŋültæ",	u"sæŋkünæ"),#	bed
	(u"hinnan",	u"hinta",	u"hinnat",	u"hinnalta",	u"hintana"),#	price
	(u"linnun",	u"lintu",	u"linnut",	u"linnulta",	u"lintuna"),#	bird
	(u"opinnon",	u"opinto",	u"opinnot",	u"opinnolta",	u"opintona"),#	study
	(u"rannan",	u"ranta",	u"rannat",	u"rannalta",	u"rantana"),#	shore
	(u"luonnon",	u"luonto",	u"luonnot",	u"luonnolta",	u"luontona"),#	nature
	(u"punnan",	u"punta",	u"punnat",	u"punnalta",	u"puntana"),#	pound
	(u"tunnin",	u"tunti",	u"tunnit",	u"tunnilta",	u"tuntina"),#	hour
	(u"kunnon",	u"kunto",	u"kunnot",	u"kunnolta",	u"kuntona"),#	condition
	(u"kannun",	u"kannu",	u"kannut",	u"kannulta",	u"kannuna"),#	can
	(u"linnan",	u"linna",	u"linnat",	u"linnalta",	u"linnana"),#	castle
	(u"tumman",	u"tumma",	u"tummat",	u"tummalta",	u"tummana"),#	dark
	(u"auriŋŋon",	u"auriŋko",	u"auriŋŋot",	u"auriŋŋolta",	u"auriŋkona"),#	sun
	(u"reŋŋin",	u"reŋki",	u"reŋŋit",	u"reŋŋiltæ",	u"reŋkinæ"),#	farm hand
	(u"vaŋŋin",	u"vaŋki",	u"vaŋŋit",	u"vaŋŋilta",	u"vaŋkina"),#	prisoner
	(u"kellon",	u"kello",	u"kellot",	u"kellolta",	u"kellona"),#	watch
	(u"kellan",	u"kelta",	u"kellat",	u"kellalta",	u"keltana"),#	yellow
	(u"sillan",	u"silta",	u"sillat",	u"sillalta",	u"siltana"),#	bridge
	(u"kullan",	u"kulta",	u"kullat ",	u"kullalta ",	u"kultana "),#	gold
	(u"virran",	u"virta",	u"virrat",	u"virralta",	u"virtana"),#	stream
	(u"parran",	u"parta",	u"parrat",	u"parralta",	u"partana")])#	beard

Problem(
    '''7: Korean
Provide rules which will account for the alternations in the stem final consonant in the following examples. State what underlying representation you are assuming for each noun.
    ''',
#	‘rice’	‘forest’	‘chestnut’	‘field’	‘sickle’	‘day’	‘face’	‘half’	
[
    (u"pamman",	u"summan",	u"pamman", u"pamman",	u"namman",	u"namman", u"namman",	u"pamman"),#	only N
	(u"pammaŋk^hɨm",	u"summaŋk^hɨm",	u"pammaŋk^hɨm", u"pammaŋk^hɨm",	u"nammaŋk^hɨm",	u"nammaŋk^hɨm", u"nammaŋk^hɨm",	u"pammaŋk^hɨm"),#	as much as N
	(u"pamnarɨm",	u"sumnarɨm",	u"pamnarɨm", u"pannarɨm",	u"nannarɨm",	u"nannarɨm", u"nannarɨm",	u"pannarɨm"),#	depending on N
	(u"pap",	u"sup",	u"pam", u"pat",	u"nat",	u"nat", u"nat",	u"pan"),#	N
	(u"papt’ero",	u"supt’ero",	u"pamtero", u"patt’ero",	u"natt’ero",	u"natt’ero", u"natt’ero",	u"pantero"),#	like N
	(u"papk’wa",	u"supk’wa",	u"pamkwa", u"pakk’wa",	u"nakk’wa",	u"nakk’wa", u"nakk’wa",	u"paŋkwa"),#	with N
	(u"papp’ota",	u"supp’ota",	u"pampota", u"papp’ota",	u"napp’ota",	u"napp’ota", u"napp’ota",	u"pampota"),#	more than N
	(u"papk’ači",	u"supk’ači",	u"pamk’ači", u"pakk’ači",	u"nakk’ači",	u"nakk’ači", u"nakk’ači",	u"paŋk’ači"),#	until N
	(u"papi",	u"sup^hi",	u"pami", u"pač^hi",	u"nasi",	u"nači", u"nač^hi",	u"pani"),#	N (nominative)
	(u"papɨn",	u"sup^hɨn",	u"pamɨn", u"pathɨn",	u"nasɨn",	u"načɨn", u"nač^hɨn",	u"panɨn"),#	N (topic)
	(u"pape",	u"sup^he",	u"pame", u"pathe",	u"nase",	u"nače", u"nač^he",	u"pane"),#	to N
	(u"papita",	u"sup^hita",	u"pamita", u"pač^hita",	u"nasita",	u"načita", u"nač^hita",	u"panita"),#	it is N
	(u"papɨro",	u"sup^hɨro",	u"pamɨro", u"pathɨro",	u"nasɨro",	u"načɨro", u"nač^hɨro",	u"panɨro")])#	using N


