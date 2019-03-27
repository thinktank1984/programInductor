from textbook_problems import *
from problems import *
from os import system
import os

names = []
for name,problem in Problem.named.iteritems():
    if "Tibetan" in name: continue
    if "Kevin" in name: continue
    
    if problem.parameters and "alternations" in problem.parameters: continue
    

    if os.path.exists("experimentOutputs/%s_incremental_disableClean=False_features=sophisticated_geometry=True.p"%name):
        continue

    print(name)
    names.append(name)

def ke():
    system("pkill -9 pypy")
    system("pkill -9 python")
    system("pkill -9 lt-cegis")
    system("rm  -rf ~/.sketch/tmp/* /scratch/ellisk/*")
    
for n in names:
    ke()
    system("sleep 5")
    ke()
    system("sleep 600")

    system("./launchers/ug0 %s"%n)
    system("sleep 1d 5h")
    ke()
    
