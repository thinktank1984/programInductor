from rule import *
from latex import *
from parseSPE import *


import fileinput

for l in fileinput.input():
    if len(l.strip()) == 0:
        continue
    
    r = parseRule(l)
    print r.latex()

    
