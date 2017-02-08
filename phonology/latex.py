# -*- coding: utf-8 -*-

from features import featureMap,tokenize

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
    u"ǰ": '\\|x{j}',
    u"ž": '\\|x{z}',
    u"n̥": '\\r*n',
    u"ñ": '\\~n',
    u"š": '\\|x{s}',
    u"č": '\\|x{c}',
    u"č^h": '\\|x{c}\\super h',
    u"k|": 'k\\textcorner',
    u"k^h": 'k\\super h',
    u"k^y": 'k\\super y',
    u"x": 'x',
    u"χ": 'x',
    u"x^y": 'x\\super y',
    u"g^y": 'g\\super y',
    u"ɣ": 'G',
    u"ŋ": 'N',
    u"N": '\\;N',
    u"G": '\\;G',
    u"ʔ": 'P',
    u"r̃": '\\~r',
    u"r̥̃": '\\r*{\\~r}',
    u"ř": '\\|x{r}'
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
    return "\\textipa{" + "".join([ latexMap.get(p,p) for p in tokenize(w) ]) + "}"

def latexMatrix(m):
    r = "\\begin{tabular}{%s}\n"%("c"*len(m[0]))
    r += "\\\\\n".join([ " & ".join([latexWord(w) for w in l ])
                         for l in m ])
    r += "\n\\end{tabular}\n"
    return r
