# -*- coding: utf-8 -*-

def compose(f,g):
    return lambda x: f(g(x))
