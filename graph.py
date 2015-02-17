## graph.py

class Vertex(object):

    def __init__(self, value=''):

        self.value = value

    def __repr__(self):

        return 'Vertex(%s)' % repr(self.value)

    __str__ = __repr__


class Edge(tuple):

    def __new__(cls, e1, e2):

        return tuple.__new__(cls, (e1, e2))

    def __repr__(self):

        return 'Edge(%s, %s)' % (repr(self[0]), repr(self[1]))

    __str__ = __repr__


class Graph(dict):

    def __init__(self, es=[], vs=[]):

        for vertex in vs:
            self.add_vertex(vertex)

        for edge in es:
            self.add_edge(edge)

    def add_vertex(self, vertex):

        self[v] = {}

    def add_edge(self, edge):

        v, w = e
        self[v][w] = e
        self[w][v] = e




if __name__ == '__main__':

    v = Vertex('v')
    w = Vertex('w')
    x = Vertex('x')
    y = Vertex('y')
    z = Vertex('z')

    e = Edge(v,w)
    f = Edge(v,x)
    g = Edge(v,y)
    h = Edge(v,z)
    i = Edge(w,x)
    j = Edge(w,y)
    k = Edge(w,z)
    l = Edge(x,y)
    m = Edge(x,z)
    n = Edge(y,z)

    vs = [v,w,x,y,z]
    es = [e,f,g,h,i,j,k,l,m,n]

    g = Graph(vs, es)



















