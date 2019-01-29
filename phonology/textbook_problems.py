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
	(u"kæps"),		# caps
	(u"kæts"),		# cats
	(u"kaks"),		# cocks
	(u"pruwfs"),	# proofs

	# suffix [z]		
	(u"kæbz"),		# cabs
	(u"kædz"),		# cads
	(u"kagz"),		# cogs
	(u"hʊvz"),		# hooves
	(u"fliyz"),		# fleas
	(u"plæwz"),		# plows
	(u"pyṛez"),		# purees

	(u"klæmz"),		# clams
	(u"kænz"),		# cans
	(u"karz"),		# cars
	(u"gəlz"),		# gulls
	

	## 3sg Present Verbal Suffix

	# suffix [s]
	(u"slæps"),		# slaps
	(u"hɩts"),		# hits
	(u"powks"),		# pokes

	# suffix [z]
	(u"stæbz"),		# stabs
	(u"haydz"),		# hides
	(u"dɩgz"),		# digs
	(u"læfs"),		# laughs
	(u"pɩθs"),		# piths

	(u"slæmz"),		# slams
	(u"kænz"),		# cans
	(u"hæŋz"),		# hangs
	(u"θrayvz"),	# thrives
	(u"beyðz"),		# bathes
	(u"flayz")		# flies

	], 
	solutions = [u'''
stem + s
stem + z
[-sonorant] -> [-voice] / [-voice] _
	''']))


Odden_Problems.append(Problem(
	u'''
Jita Odden 79
	''',
	[ 
	(u"okuβuma"),		# to hit
	(u"okuβumira"),		# to hit for
	(u"okuβumana"),		# to hit each other
	(u"okuβumirana"),	# to hit for each other

	(u"okusiβa"),		# to block
	(u"okusiβira"),		# to block for
	(u"okusiβana"),		# to block each other
	(u"okusiβirana"),	# to block for each other

	(u"okulúma"),		# to bite
	(u"okulumíra"),		# to bite for
	(u"okulumána"),		# to bite each other
	(u"okulumírana"),	# to bite for each other

	(u"okukúβa"),		# to fold
	(u"okukuβíra"),		# to fold for
	(u"okukuβána"),		# to fold each other
	(u"okukuβírana"),	# to fold for each other


	(u"okumuβúma"),		# to hit him/her
	(u"okumuβúmira"),	# to hit for him/her
	(u"okučiβúma"),		# to hit it
	(u"okučiβúmira"),	# to hit for it

	(u"okumusíβa"),		# to block him/her
	(u"okumusíβira"),	# to block for him/her
	(u"okučisíβa"),		# to block it
	(u"okučisíβira")	# to block for it
	], 
	solutions = [u'''
oku + stem + a
oku + stem + ir + a
oku + stem + an + a
oku + stem + ir + an + a

oku + mu + stem + a
oku + mu + stem + ir + a
oku + či + stem + a
oku + či + stem + ir + a

V > [+highTone]/[+highTone]C*_
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
	(u"kupə" 		u"kumnɨnta"),		# bend
	(u"čəpə", 		u"cəm̌nɨnta"),		# fold
	(u"tatə",	 	u"tannɨnta"),		# close
	(u"put^hə", 	u"punnɨnta"),		# adhere
	(u"čəč^ha", 	u"čonnɨnta"),		# follow
	(u"məkə", 		u"məŋnɨnta"),		# eat
	(u"sək̚ə", 		u"səŋnɨnta"),		# mix
	(u"tak̚a", 		u"taŋnɨnta"),		# polish
	(u"čukə",	 	u"čuŋnɨnta"),		# die
	(u"ikə", 		u"iŋnɨnta"),		# ripen

	], 
	solutions = [u'''
stem + a or ə
stem + nɨnta
# no solution
	''']
	))

