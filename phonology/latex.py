# -*- coding: utf-8 -*-

from features import featureMap,tokenize

from morph import *
from problems import *

import cPickle as pickle
import os

latexMap = {
    u"ɨ": '1',
    u"ɯ": 'W',
    u"ɩ": '\\textiota',
    u"ə": '@',
    u"ɛ": 'E',
    u"æ": '\\ae',
    # rounded vowels
    u"ü": '\\"u',
    u"ʊ": 'U',
    u"ö": '\\"o',
    u"ɔ": 'O',
    #possibly missing are umlauts

    # consonance
    u"y": 'j',
    u"p|": 'p\\textcorner',
    u"p^h": 'p\\super h',
    u"β": 'B',
    u"m̥": '\\r*m',
    u"θ": 'T',
    u"d^z": 'd\\super z',
    u"t|": 't\\textcorner',
    u"t^s": 't\\super s',
    u"t^h": 't\\super h',
    u"ṭ": '\\.*t',
    u"ḍ": '\\.*d',
    u"ð": 'D',
    u"ǰ": 'd\\super Z',#'\\v{j}',
    u"ž": 'Z',#'\\v{z}',
    u"n̥": '\\r*n',
    u"ñ": '\\~n',
    u"š": 'S',#'\\v{s}',
    u"č": 't\\super S',#'\\v{c}',
    u"č^h": 't\\super S\\super h',#'\\v{c}\\super h',
    u"k|": 'k\\textcorner',
    u"k^h": 'k\\super h',
    u"k^y": 'k\\super j',
    u"x": 'x',
    u"χ": 'x',
    u"x^y": 'x\\super j',
    u"g^y": 'g\\super j',
    u"ɣ": 'G',
    u"ŋ": 'N',
    u"N": '\\;N',
    u"G": '\\;G',
    u"ʔ": 'P',
    u"r̃": '\\~r',
    u"r̥̃": '\\r*{\\~r}',
    u"ř": '\\v{r}'
}

# tone stuff
for v in featureMap:
    if "vowel" in featureMap[v]:
        latexMap[v + u"́"] = '\\\'' + latexMap.get(v,v)
        latexMap[v + u"`"] = '\\\'=' + latexMap.get(v,v)
        latexMap[v + u"¯"] = '\\=' + latexMap.get(v,v)
        latexMap[v + u":"] = latexMap.get(v,v) + ':'
        latexMap[v + u"̌"] = '\\|x{' + latexMap.get(v,v) + '}'
        latexMap[v + u"̃"] = '\\~' + latexMap.get(v,v)
        latexMap[u"̌" + v] =  '\\|x{' + latexMap.get(v,v) + '}'


def latexWord(w):
    if w == None: return " -- "
    if not isinstance(w,Morph): w = Morph(w)
    return "\\textipa{" + "".join([ latexMap.get(p,p) for p in w.phonemes ]) + "}"

def latexMatrix(m):
    r = "\\begin{tabular}{%s}\n"%("c"*len(m[0]))
    r += "\\\\\n".join([ " & ".join([latexWord(w) for w in l ])
                         for l in m ])
    r += "\n\\end{tabular}\n"
    return r

def latexSolutionAndProblem(path):
    with open(path,'rb') as handle: solution = pickle.load(handle)
    if isinstance(solution,list): solution = solution[0]

    # figure out which problem it corresponds to
    problem = None
    f = path.split('/')[-1][:-2]
    if f.startswith('alternation'):
        problem = alternationProblems[int(f.split('_')[-2]) - 1]
    elif f.startswith('matrix'):
        problemNumber = int(f.split('_')[-1])
        if problemNumber < 50: problem = underlyingProblems[problemNumber - 1]
        elif problemNumber < 70: problem = interactingProblems[problemNumber - 50 - 1]
        elif problemNumber < 80: problem = sevenProblems[problemNumber - 70 - 1]
    if problem == None:
        print "Could not find the problem for path",path
        assert False

    if problem.parameters == None:
        r = "\\begin{longtable}{%s}\\toprule\\\\\n"%("l"*len(solution.prefixes) + "|l")
        r += " & ".join([ ("$\\varnothing$" if len(p) == 0 else latexWord(p)) + "$+$stem$+$" + ("$\\varnothing$$" if len(s) == 0 else latexWord(s))
                          for p,s in zip(solution.prefixes, solution.suffixes) ] + ["UR"])
        r += "\n\\\\ \\midrule\n"
        for j in range(len(problem.data)):
            if j < len(solution.underlyingForms): ur = solution.underlyingForms[j]
            else: ur = None
            r += " & ".join([ latexWord(x) for x in problem.data[j] ] + [latexWord(ur)])
            r += "\\\\\n"
        r += "\\bottomrule\\end{longtable}"

    else: assert False

    r += '''\n\\begin{tabular}{l}\\emph{Rules: }\\\\
%s
\\end{tabular}'''%("\\\\".join([ r.latex() for r in solution.rules ]))
    return r

LATEXPRELUDE = '''
\\documentclass{article}
\\usepackage{tipa}
\\usepackage{booktabs}
\\usepackage{amssymb}
\\usepackage{longtable}
\\begin{document}

'''

LATEXEPILOGUE = '''

\\end{document}
'''

def exportLatexDocument(source, path):
    with open(path,'w') as handle:
        handle.write(LATEXPRELUDE + source + LATEXEPILOGUE)
    os.system('pdflatex %s'%path)

if __name__ == "__main__":
    exportLatexDocument(latexSolutionAndProblem("pickles/matrix_55.p"),"../../phonologyPaper/test.tex")

    
            

        
