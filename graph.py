## graph.py

import string
import random as ran

class Vertex(object):

    def __init__(self, value=''):

        self.value = value

    def __repr__(self):

        return 'Vertex(%s)' % repr(self.value)

    __str__ = __repr__


class Edge(tuple):
    '''
    Should replace this with 'arc' for directed graph.
    '''

    def __new__(cls, e1, e2):

        return tuple.__new__(cls, (e1, e2))

    def __repr__(self):

        return 'Edge(%s, %s)' % (repr(self[0]), repr(self[1]))

    __str__ = __repr__


class UndirectedGraph(dict):

    def __init__(self, es=[], vs=[]):

        for vertex in vs:
            self.add_vertex(vertex)

        for edge in es:
            self.add_edge(edge)

    def add_vertex(self, v):

        self[v] = {}

    def add_edge(self, e):

        a, b = e
        self[a][b] = e
        self[b][a] = e


class DirectedGraph(dict):

    def __init__(self, vs=[], es=[]):

        for vertex in vs:
            self.add_vertex(vertex)
        for edge in es:
            self.add_edge(edge)

    def add_vertex(self, v):

        self[v] = {}

    def add_edge(self, e):

        a, b = e
        self[a][b] = e

    def add_alphabet_vertices(self):

        for char in string.lowercase:
            self.add_vertex(Vertex(char))

    def add_random_edges(self, threshold):

        for i in self:
            for j in self:
                if i != j and ran.random() >= threshold:
                    self.add_edge(Edge(i,j))





if __name__ == '__main__':

    v = Vertex('v')
    w = Vertex('w')
    vs = [v, w]

    e = Edge(v,w)
    es = [e]

    ug = UndirectedGraph(es, vs)


    threshold = 0.95

    dg = DirectedGraph()
    dg.add_alphabet_vertices()
    dg.add_random_edges(threshold)



















