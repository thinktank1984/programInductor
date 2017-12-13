# -*- coding: utf-8 -*-

from features import featureMap,tokenize

#from solution import *
from morph import *
from problems import *

import cPickle as pickle
import os

latexMap = {
    u"ɨ": '1',
    u"ɯ": 'W',
    u"ɩ": '\\textiota ',
    u"ə": '@',
    u"ɛ": 'E',
    u"ʌ": '2',
    u"æ": '\\ae ',
    # rounded vowels
    u"ü": '\\"u',
    u"ʊ": 'U',
    u"ö": '\\"o',
    u"ɔ": 'O',
    #possibly missing are umlauts

    # consonance
    u"ṛ": "\\.*r",
    u"n^y": "n\\super j",
    u"r^y": "r\\super j",
    u"s^y": "s\\super j",
    u"z^y": "z\\super j",
    u"b^y": "b\\super j",
    u"t^s^y": "r\\super s \\super j",
    u"d^y": "d\\super j",
    u"ħ": "\\textcrh ",
    u"ʕ": 'Q',
    u"m^y": "m\\super j",
    u"t^y": "t\\super j",
    u"ṇ": '\\.*n',
    u"l^y": "l\\super j",
    u"v^y": "v\\super j",
    u"š^y": "S\\super j",
    u"y": 'j',
    u"p|": 'p\\textcorner ',
    u"p^h": 'p\\super h',
    u"β": 'B',
    u"m̥": '\\r*m',
    u"θ": 'T',
    u"d^z": 'd\\super z',
    u"t|": 't\\textcorner ',
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
    u"k|": 'k\\textcorner ',
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
        latexMap[v + u"́"] = '\\\'{' + latexMap.get(v,v) + '}'
        latexMap[v + u"`"] = '\\\'={' + latexMap.get(v,v) + '}'
        latexMap[v + u"¯"] = '\\={' + latexMap.get(v,v) + '}'
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
    solution = loadPickle(path)
    if isinstance(solution,list): solution = solution[0]
    if isinstance(solution, list): return "(invalid solution)"

    # figure out which problem it corresponds to
    problem = None
    f = path.split('/')[-1][:-2]
    if f.startswith('alternation'):
        problem = alternationProblems[int(f.split('_')[-1]) - 1]
    elif f.startswith('matrix'):
        problemNumber = int(f.split('_')[-1])
        if problemNumber < 50: problem = underlyingProblems[problemNumber - 1]
        elif problemNumber < 70: problem = interactingProblems[problemNumber - 50 - 1]
        elif problemNumber < 80: problem = sevenProblems[problemNumber - 70 - 1]
    if problem == None:
        print "Could not find the problem for path",path
        assert False

    r = "\\emph{%s:}\\\\"%(problem.languageName.replace('-','--'))
    
    if problem.parameters == None:
        r += "\\begin{longtable}{%s}\\toprule\n"%("l"*len(solution.prefixes) + "|l")
        r += " & ".join([ ("$\\varnothing$" if len(p) == 0 else latexWord(p)) + " $+$stem$+$ " + ("$\\varnothing$" if len(s) == 0 else latexWord(s))
                          for p,s in zip(solution.prefixes, solution.suffixes) ] + ["UR"])
        r += "\n\\\\ \\midrule\n"
        for j in range(len(problem.data)):
            if j < len(solution.underlyingForms): ur = solution.underlyingForms[j]
            else: ur = None
            r += " & ".join([ latexWord(x) for x in problem.data[j] ] + [latexWord(ur)])
            r += "\\\\\n"
        r += "\\bottomrule\\end{longtable}"

        rules = solution.rules

    elif "Numbers between" in problem.description:
        r += "\\begin{longtable}{%s}\\toprule\n"%("ll")
        r += " & ".join([ "Number","Surface form"])
        r += "\n\\\\ \\midrule\n"
        for j in range(len(problem.data)):
            r += str(problem.parameters[j]) + " & "
            r += latexWord(problem.data[j])
            r += "\\\\\n"
        r += "\\bottomrule\\end{longtable}"
        rules = [solution]

    elif "alternations" in problem.parameters:
        assert str(solution.__class__) == 'solution.AlternationSolution'
        
        r += "\\begin{longtable}{%s}\\toprule\n"%("ll")
        r += " & ".join([ "Surface form","UR"])
        r += "\n\\\\ \\midrule\n"
        for x in problem.data:
            r += latexWord(x) + "&"
            r += latexWord(solution.applySubstitution(Morph(x)))
            r += "\\\\\n"
        r += "\\bottomrule\\end{longtable}\n\n"
        r += "\\begin{longtable}{ll}\\toprule\n"
        r += "\\emph{The surface form...}&\\emph{Is underlyingly...}"
        r += "\n\\\\ \\midrule\n"
        for k,v in solution.substitution.iteritems():
            r += latexWord(k) + "&" + latexWord(v)
            r += "\\\\\n"
        r += "\\bottomrule\\end{longtable}\n\n"

        rules = solution.rules
        
        

    r += '''\n\\begin{tabular}{l}\\emph{Rules: }\\\\
%s
\\end{tabular}'''%("\\\\".join([ r.latex() for r in rules if not r.doesNothing() ]))
    return r

def latexFeatures():
    features = list(set(f for _,v in featureMap.iteritems() for f in v))[:2]
    n = len(featureMap)
    r = "\\begin{longtable}{%s}\\toprule\n"%("c|" + "l"*len(features))
    r += "&".join([""] + features)
    r += "\n\\\\ \\midrule\n"
    for p,fs in featureMap.iteritems():
        r += " & ".join([latexWord(p)] + \
                        [ "$+$" if f in fs else "$-$" for f in features ])
        r += "\\\\\n"
    r += "\\bottomrule\\end{longtable}"
    print r
    return r
    
LATEXPRELUDE = '''
\\documentclass{article}
\\usepackage[margin = 0.1cm]{geometry}
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
    if '/' in path: directory = "/".join(path.split("/")[:-1])
    else: directory = "."
    #os.system('pdflatex %s -output-directory %s'%(path,directory))

if __name__ == "__main__":
    source = "\n\n\\pagebreak\n\n".join([ latexSolutionAndProblem("pickles/alternation_%d.p"%j)
                                          for j in range(1,11+1) ] + \
                                        [ latexSolutionAndProblem("pickles/matrix_%d.p"%j)
                                          for j in range(1,15) ] + \
                                        [ latexSolutionAndProblem("pickles/matrix_%d.p"%j)
                                          for j in [51,52,53,55] ] + [])
                                        #[ latexFeatures() ])
    exportLatexDocument(source,"../../phonologyPaper/test.tex")

    
            

        
