from rule import *
from latex import *
from parseSPE import *


import fileinput

for l in fileinput.input():
    r = parseRule(l)
    print r.latex()

    
