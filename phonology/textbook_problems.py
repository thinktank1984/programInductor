# -*- coding: utf-8 -*-
from problems import *


# Oden's Introducing Phonology
Odden_Problems = []

Odden_Problems.append(Problem(
	u'''
Russian Odden 68-69''',
	[
	# Nominative sg		Genitive sg  		Gloss
	(u"vagon",			u"vagona"),			# wagon
	(u"avtomobil^y",	u"avtomobil^ya"),	# car
	(u"večer",			u"večera"),			# evening
	(u"muš",			u"muža"),			# husband
	(u"karandaš",		u"karandaša"),		# pencil
	(u"glas",			u"glaza"),			# eye
	(u"golos",			u"golosa"),			# voice
	(u"ras",			u"raza"),			# time
	(u"les",			u"lesa"),			# forest
	(u"porok",			u"poroga"),			# threshold
	(u"vrak",			u"vraga"),			# enemy
	(u"urok",			u"uroka"),			# lesson
	(u"porok",			u"poroka"),			# vice
	(u"t^svet",			u"t^sveta"),		# color
	(u"prut",			u"pruda"),			# pond
	(u"soldat",			u"soldata"),		# soldier
	(u"zavot",			u"zavoda"),			# factory
	(u"xlep",			u"xleba"),			# bread
	(u"grip",			u"griba"),			# mushroom
	(u"trup",			u"trupa")			# corpse
	], 
	solutions = [u'''
stem
stem + a
[-sonorant] -> [-voice] / _ #
	''']))

Odden_Problems.append(Problem(
	u'''
Finnish Odden 73-74	
	''',
	[ 
	# a. Nominative sg		Partitive sg  		Gloss
	(u"aamu",			u"aamua"),			# morning
	(u"hopea",			u"hopeaa"),			# silver
	(u"katto",			u"kattoa"),			# roof
	(u"kello",			u"kelloa"),			# clock
	(u"kirya",			u"kiryaa"),			# book
	(u"külmæ",			u"külmææ"),			# cold
	(u"koulu",			u"koulua"),			# school
	(u"lintu",			u"lintua"),			# bird
	(u"hüllü",			u"hüllüæ"),			# shelf
	(u"kömpelö",		u"kömpelöæ"),		# clumsy
	(u"nækö",			u"næköæ"),			# appearance

	# b. Nominative sg		Partitive sg  		Gloss
	(u"yoki",			u"yokea"),			# river
	(u"kivi",			u"kiveæ"),			# stone
	(u"muuri",			u"muuria"),			# wall
	(u"naapuri",		u"naapuria"),		# neighbor
	(u"nimi",			u"nimeæ"),			# name
	(u"kaappi",			u"kaappia"),		# chest of drawers
	(u"kaikki",			u"kaikkea"),		# all
	(u"kiirehti",		u"kiirehtiæ"),		# hurry
	(u"lehti",			u"lehteæ"),			# leaf
	(u"mæki",			u"mækeæ"),			# hill
	(u"ovi",			u"ovea"),			# door
	(u"posti",			u"postia"),			# mail
	(u"tukki",			u"tukkia"),			# log
	(u"æiti",			u"æitiæ"),			# mother
	(u"englanti",		u"englantia"),		# England
	(u"yærvi",			u"yærveæ"),			# lake
	(u"koski",			u"koskea"),			# waterfall
	(u"reki",			u"rekeæ"),			# sledge
	(u"væki",			u"vækeæ")			# people
	], 
	solutions = [u'''
stem
stem + æ
e ---> [ +high ] /  _ #
æ ---> [ +back ] / [ +back +continuant ] [  ]* _
	''']))

Odden_Problems.append(Problem(
	u'''
Kerewe Odden 76-77	
	''',
	[ 
	# Infinitive		1sg habitual		3sg habitual		Imperative		Gloss

	(u"kupaamba",		u"mpaamba",			u"apaamba",			u"paamba"),		# adorn
	(u"kupaaŋga",		u"mpaaŋga",			u"apaaŋga",			u"paaŋga"),		# line up
	(u"kupima",			u"mpima",			u"apima",			u"pima"),		# measure
	(u"kupuupa",		u"mpuupa",			u"apuupa",			u"puupa"),		# be light
	(u"kupekeča",		u"mpekeča",			u"apekeča",			u"pekeča"),		# make fire with stick
	(u"kupiinda",		u"mpiinda",			u"apiinda",			u"piinda"),		# be bent
	(u"kuhiiga",		u"mpiiga",			u"ahiiga",			u"hiiga"),		# hunt
	(u"kuheeka",		u"mpeeka",			u"aheeka",			u"heeka"),		# carry
	(u"kuhaaŋga",		u"mpaaŋga",			u"ahaaŋga",			u"haaŋga"),		# create
	(u"kuheeba",		u"mpeeba",			u"aheeba",			u"heeba"),		# guide
	(u"kuhiima",		u"mpiima",			u"ahiima",			u"hiima"),		# gasp
	(u"kuhuuha",		u"mpuuha",			u"ahuuha",			u"huuha")		# breath into
	], 
	solutions = [u'''
ku + stem + a
m + stem + a
a + stem + a
stem + a

# postnasal hardening
[-voice] -> p / [+nasal] _
	''']
	))


Odden_Problems.append(Problem(
	u'''
English Odden 77-78	
	''',
	[ 
	## Noun Plural Suffix

	# suffix [s]		
	(u"kæps",),		# caps
	(u"kæts",),		# cats
	(u"kaks",),		# cocks
	(u"pruwfs",),	# proofs

	# suffix [z]		
	(u"kæbz",),		# cabs
	(u"kædz",),		# cads
	(u"kagz",),		# cogs
	(u"hʊvz",),		# hooves
	(u"fliyz",),		# fleas
	(u"plæwz",),		# plows
	(u"pyṛez",),		# purees

	(u"klæmz",),		# clams
	(u"kænz",),		# cans
	(u"karz",),		# cars
	(u"gəlz",),		# gulls
	

	## 3sg Present Verbal Suffix

	# suffix [s]
	(u"slæps",),		# slaps
	(u"hɩts",),		# hits
	(u"powks",),		# pokes

	# suffix [z]
	(u"stæbz",),		# stabs
	(u"haydz",),		# hides
	(u"dɩgz",),		# digs
	(u"læfs",),		# laughs
	(u"pɩθs",),		# piths

	(u"slæmz",),		# slams
	(u"kænz",),		# cans
	(u"hæŋz",),		# hangs
	(u"θrayvz",),	# thrives
	(u"beyðz",),		# bathes
	(u"flayz",)		# flies

	], 
	solutions = [u'''
stem + z
[-sonorant] -> [-voice] / [-voice] _
	''']))


Odden_Problems.append(Problem(
	u'''
Jita Odden 79
	''',
	[ 
	(u"okuβuma",		# to hit
	 u"okuβumira",		# to hit for
	 u"okuβumana",		# to hit each other
	 u"okuβumirana",	# to hit for each other
	 u"okumuβúma",		# to hit him/her
	 u"okumuβúmira",	# to hit for him/her
	 u"okučiβúma",	# to hit it
	 u"okučiβúmira"),	# to hit for it
            

	(u"okusiβa",		# to block
	 u"okusiβira",		# to block for
	 u"okusiβana",		# to block each other
	 u"okusiβirana",	# to block for each other
     u"okumusíβa",		# to block him/her
	 u"okumusíβira",	# to block for him/her
	 u"okučisíβa",	# to block it
	 u"okučisíβira"),	# to block for it

	(u"okulúma",		# to bite
	 u"okulumíra",		# to bite for
	 u"okulumána",		# to bite each other
	 u"okulumírana",	# to bite for each other
         None,None,None,None),

	(u"okukúβa",		# to fold
	 u"okukuβíra",		# to fold for
	 u"okukuβána",		# to fold each other
	 u"okukuβírana",	# to fold for each other
         None,None,None,None)

	], 
	solutions = [u'''
oku + stem + a
oku + stem + ir + a
oku + stem + an + a
oku + stem + ir + an + a

oku + mú + stem + a
oku + mú + stem + ir + a
oku + čí + stem + a
oku + čí + stem + ir + a

# High tone shifting (check if the rule is right)
V > [+highTone]/[+highTone]C*_C
V > [-highTone]/_C*[+highTone]
	''']
	))




Odden_Problems.append(Problem(
	u'''
Korean Odden 81
	''',
	[ 
	# Imperative	Plain Present		Gloss

	(u"ana",		u"annɨnta"),		# hug
	(u"kama",		u"kamnɨnta"),		# wind
	(u"sinə",		u"sinnɨnta"),		# wear shoes
	(u"t̚atɨmə",		u"t̚atɨmnɨnta"),		# trim
	(u"nəmə",		u"nəmnɨnta"),		# overflow
	(u"nama",		u"namnɨnta"),		# remain
	(u"č^hama",		u"č^hamnɨnta"),		# endure
	(u"ipə",		u"imnɨnta"),		# put on
	(u"kupə", 		u"kumnɨnta"),		# bend
	(u"čəpə", 		u"čəmnɨnta"),		# fold
	(u"tatə",	 	u"tannɨnta"),		# close
	(u"put^hə", 	u"punnɨnta"),		# adhere
	(u"čəč^ha", 	u"čonnɨnta"),		# follow
	(u"məkə", 		u"məŋnɨnta"),		# eat
	(u"sək̚ə", 		u"səŋnɨnta"),		# mix
	(u"tak̚a", 		u"taŋnɨnta"),		# polish
	(u"čukə",	 	u"čuŋnɨnta"),		# die
	(u"ikə", 		u"iŋnɨnta")			# ripen
	],
	solutions = [u'''
stem + a or ə
stem + nɨnta

# no solution
	''']
	))


Odden_Problems.append(Problem(
	u'''
Koasati Odden 81
	''',
	[ 
	# Noun			1st-sg-pos("my") + N	Gloss
	(u"apahčá",		u"amapahčá"),			# shadow
	(u"asikčí",		u"amasikčí"),			# muscle
	(u"ilkanó",		u"amilkanó"),			# right side
	(u"ifá",		u"amifá"),				# dog
	(u"a:pó",		u"ama:pó"),				# grandmother
	(u"iskí",		u"amiskí"),				# mother
	(u"pačokkö́ka",	u"ampačokkö́ka"),		# chair
	(u"towá",		u"antowá"),				# onion
	(u"kastó",		u"aŋkastó"),			# flea
	(u"bayá:na",	u"ambayá:na"),			# stomach
	(u"tá:ta",		u"antá:ta"),			# father
	(u"čofkoní",	u"añčofkoní"),			# bone
	(u"kitiłká",	u"aŋkitiłká"),			# hair bangs
	(u"toní",		u"antoní")				# hip
	], 
	solutions = [u'''
# no solution
	''']
	))


Odden_Problems.append(Problem(
	u'''
Samoan Odden 85
	''',
	[ 
	# Simple		Perfective		Gloss
	(u"olo",		u"oloia"),		# rub
	(u"lafo",		u"lafoia"),		# cast
	(u"aŋa",		u"aŋaia"),		# face
	(u"usu",		u"usuia"),		# get up and go early
	(u"tau",		u"tauia"),		# reach a destination
	(u"taui",		u"tauia"),		# repay
	(u"sa:ʔili",	u"sa:ʔilia"),	# look for
	(u"vaŋai",		u"vaŋaia"),		# face each other
	(u"paʔi",		u"paʔia"),		# touch
	(u"naumati",	u"naumatia"),	# be waterless
	(u"sa:uni",		u"sa:unia"),	# prepare
	(u"seŋi",		u"seŋia"),		# be shy
	(u"lele",		u"lelea"),		# fly
	(u"suʔe",		u"suʔea"),		# uncover
	(u"taʔe",		u"taʔea"),		# smash
	(u"tafe",		u"tafea"),		# flow
	(u"ta:upule",	u"ta:upulea"),	# confer
	(u"palepale",	u"palepalea"),	# hold firm

	(u"tu:",		u"tu:lia"),		# stand
	(u"tau",		u"taulia"),		# cost
	(u"ʔalo",		u"ʔalofia"),	# avoid
	(u"oso",		u"osofia"),		# jump
	(u"sao",		u"saofia"),		# collect
	(u"asu",		u"asuŋia"),		# smoke
	(u"pole",		u"poleŋia"),	# be anxious
	(u"ifo",		u"ifoŋia"),		# bow down
	(u"ula",		u"ulaŋia"),		# mock
	(u"milo",		u"milosia"),	# twist
	(u"valu",		u"valusia"),	# scrape
	(u"vela",		u"velasia"),	# be cooked
	(u"api",		u"apitia"),		# be lodged
	(u"eʔe",		u"eʔetia"),		# be raised
	(u"lava:",		u"lava:tia"),	# be able
	(u"u:",			u"u:tia"),		# grip
	(u"puni",		u"punitia"),	# be blocked
	(u"siʔo",		u"siʔomia"),	# be enclosed
	(u"ŋalo",		u"ŋalomia"),	# forget
	(u"sopo",		u"sopoʔia"),	# go across

	(u"au",			u"aulia"),		# flow on
	(u"ma:tau",		u"ma:taulia"),	# observe
	(u"ili",		u"ilifia"),		# blow
	(u"ulu",		u"ulufia"),		# enter
	(u"taŋo",		u"taŋofia"),	# take hold
	(u"soa",		u"soaŋia"),		# have a friend
	(u"fesili",		u"fesiliŋia"),	# question
	(u"ʔote",		u"ʔoteŋia"),	# scold
	(u"tofu",		u"tofuŋia"),	# dive
	(u"laʔa",		u"laʔasia"),	# step
	(u"taŋi",		u"taŋisia"),	# cry
	(u"motu",		u"motusia"),	# break
	(u"mataʔu",		u"mataʔutia"),	# fear
	(u"sau",		u"sautia"),		# fall
	(u"oʔo",		u"oʔotia"),		# arrive
	(u"ufi",		u"ufitia"),		# cover
	(u"tanu",		u"tanumia"),	# cover up
	(u"moʔo",		u"moʔomia"),	# admire
	(u"tao",		u"taomia"),		# cover
	(u"fana",		u"fanaʔia")		# shoot
	], 
	solutions = [u'''
stem
stem + ia

# Vowel-cluster reduction
[ +vowel -back] -> 0 / [ +vowel -back ] _ 

# Final consonant deletion
 C -> 0 / _ #
	''']
	))

Odden_Problems.append(Problem(
	u'''
Palauan Odden 88
	''',
	[ 
	# Present middle	Future innovative	Future Conservative		Gloss
	(u"mədáŋəb",		u"dəŋəbáll",		u"dəŋóbl"),				# cover
	(u"mətéʔəb",		u"təʔəbáll",		u"təʔíbl"),				# pull out
	(u"məŋétəm",		u"ŋətəmáll",		u"ŋətóml"),				# lick
	(u"mətábək",		u"təbəkáll",		u"təbákl"),				# patch
	(u"məʔárəm",		u"ʔərəmáll",		u"ʔəróml"),				# taste
	(u"məsésəb",		u"səsəbáll",		u"səsóbl")				# burn
	], 
	solutions = [u'''
mə + stem
stem + al + l
stem + l

#final syllable stressed if ends in two consonants
#otherwise the second to last (penultimate) syllable stressed

# Unstressed vowel reduction
        [+vowel -highTone] -> ə
	''']
	))


Odden_Problems.append(Problem(
	u'''
Bukusu Odden 105
	''',
	[ 
	# Imperative	3pl pres		1sg pres		Gloss
	(u"ča",			u"βača",		u"ñǰa"),		# go
	(u"čexa",		u"βačexa",		u"ñǰexa"),		# laugh
	(u"čučuuŋga",	u"βačučuuŋga",	u"ñǰučuuŋga"),	# sieve
	(u"talaanda",	u"βatalaanda",	u"ndalaanda"),	# go around
	(u"teexa",		u"βateexa",		u"ndeexa"),		# cook
	(u"tiira",		u"βatiira",		u"ndiira"),		# get ahold of
	(u"piima",		u"βapiima",		u"mbiima"),		# weigh
	(u"pakala",		u"βapakala",	u"mbakala"),	# writhe in pain
	(u"ketulula",	u"βaketulula",	u"ŋgetulula"),	# pour out
	(u"kona",		u"βakona",		u"ŋgona"),		# pass the night
	(u"kula",		u"βakula",		u"ŋgula"),		# buy
	(u"kwa",		u"βakwa",		u"ŋgwa")		# fall 
	], 
	solutions = [u'''
stem
βa + stem
n + stem

# Postnasal voicing
[ -voice ] -> [ +voice ] / [ +nasal ] _
# Nasal place assimilation
[ +nasal ] -> αplace _[αplace]
	''']
	))


Odden_Problems.append(Problem(
	u'''
Lithuanian Odden 114
	''',
	[ 
	# a.
	# /at/
	(u"at-eiti",None),		# to arrive
	(u"at-imti",None),			# to take away
	(u"at-nešti",None),		# to bring
	(u"at-leisti",None),		# to forgive
	(u"at-likti",None),		# to complete
	(u"at-ko:pti",None),		# to rise
	(u"at-praši:ti",None),		# to ask
	(u"at-kurti",None),		# to reestablish

	# /ap/
	(None,u"ap-eiti"),		# to circumvent
	(None,u"ap-ieško:ti"),		# to search everywhere
	(None,u"ap-akti"),			# to become blind
	(None,u"ap-mo:ki:ti"),		# to train
	(None,u"ap-temdi:ti"),		# to obscure
	(None,u"ap-šaukti"),		# to proclaim

	# b.
	# /at/
	(u"ad-bekti",None),		# to run up
	(u"ad-gauti",None),		# to get back
	(u"ad-bukti",None),		# to become blunt
	(u"ad-gimti",None),		# to be born again

	# /ap/
	(None,u"ab-gauti"),		# to deceive
	(None,u"ab-ž^yureti"),		# to have a look at
	(None,u"ab-želti"),		# to become overgrown
	(None,u"ab-dauži:ti"),		# to damage
	(None,u"ab-draski:ti"),	# to tear


	], 
	solutions = [u'''
at + stem
ap + stem

# Voicing assimilation
[ -sonorant ] -> [ +voice ] / _ [ -sonorant +voice ]
	''']
	))


Odden_Problems.append(Problem(
	u'''
Armenian Odden 116
	''',
	[ 
	# a.
	(u"kert^ham",			# I will go
	u"k-asiem",				# I will say
	u"k-aniem",				# I will do
	u"k-akaniem",			# I will watch
	u"k-oxniem",			# I will bless
	u"k-urriem",			# I will swell
	None,None,None,None),

	# b.
	(u"kə-tam",				# I will give
	u"kə-kienam",			# I will exist
	u"gə-bəzzam",			# I will buzz
	u"gə-lam",				# I will cry
	u"gə-zəram",			# I will bray
	u"k^hə-t^huoyniem",		# I will allow
	u"k^hə-č^hap^hiem",		# I will measure
	u"g^hə-b^hieřiem",		# I will carry
	u"g^hə-g^huom",			# I will come
	u"g^hə-d^z^hieviem",	# I will form
	)
	], 
	solutions = [u'''
at + stem
ap + stem

# Voicing assimilation
[ -sonorant ] -> [ +voice ] / _ [ +sonorant +voice ]
	''']
	))

Odden_Problems.append(Problem(
	u'''
Yawelmani Odden 170
	''',
	[
	# Nonefuture	Imperative	Dubitative	Passive aorist	Gloss 
	(u"xathin",		u"xatk'a",	u"xatal",	u"xatit"),	# eat
	(u"dubhun",		u"dubk'a",	u"dubal",	u"dubut"),	# lead by hand
	(u"xilhin",		u"xilk'a",	u"xilal",	u"xilit"),	# tangle
	(u"k'oʔhin",	u"k'oʔk'o",	u"k'oʔol",	u"k'oʔit"),	# throw
	(u"doshin",		u"dosk'o",	u"do:sol",	u"do:sit"),	# report
	(u"ṣaphin",		u"ṣapk'a",	u"ṣa:pal",	u"ṣa:pit"),	# burn

	(u"lanhin",		u"lank'a",	u"la:nal",	u"la:nit"),	# hear
	(u"mek'hin",	u"mek'k'a",	u"me:k'al",	u"me:k'it"),	# swallow
	(u"wonhin",		u"wonk'o",	u"wo:nol",	u"wo:nit"),		# hide
	(u"p'axathin",	u"p'axatk'a",	u"p'axa:tal",	u"p'axa:tit"),	# mourn
	(u"hiwethin",	u"hiwetk'a",	u"hiwe:tal",	u"hiwe:tit"),	#walk


	], 
	solutions = [u'''

	''']
	))
