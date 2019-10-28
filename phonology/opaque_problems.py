# -*- coding: utf-8 -*-
from problems import *
import random

def load_problem(name, path, substitution=None):
    import codecs
    data = []
    with codecs.open(path, encoding='utf-8') as handle:
        for l in handle:
            l = l.replace(u"γ",u"ɣ").replace(u"dʒ",u"ǰ").replace(u"tʃ",u"č")
            if substitution:
                for k,v in substitution.iteritems():
                    l = l.replace(k,v)
            l = l.strip().split("\t")
            data.append(tuple(l))
    random.shuffle(data)
    return Problem(u"interaction Kevin %s"%name, data)

load_problem("1b","opaque/dataset1-B4.txt")
load_problem("1cb","opaque/dataset1-CB4.txt")
load_problem("1f","opaque/dataset1-F4.txt")
load_problem("1cf","opaque/dataset1-CF4.txt")

load_problem("2b","opaque/dataset2-B.txt")
load_problem("2cb","opaque/dataset2-CB.txt")
load_problem("2f","opaque/dataset2-F.txt")
load_problem("2cf","opaque/dataset2-CF.txt")

load_problem("3b","opaque/Faroese_NEW_B.txt",
             {u"z": u"ð"})
load_problem("3cb","opaque/Faroese_OLD_CB.txt",
             {u"z": u"ð"})
