class Primitive():
    def __init__(self):
        self.halfedge = None
        self.index = -1

    def __str__(self) -> str:
        return str(self.index)

    def __repr__(self) -> str:
        return str(self)


class Halfedge(Primitive):
    def __init__(self):
        # note parent constructor is replaced
        self.vertex = None
        self.edge = None
        self.face = None
        # self.corner = None
        self.next = None
        self.twin = None
        self.index = -1  # an ID between 0 and |H| - 1, where |H| is the number of halfedges in a mesh

    def prev(self):
        # TODO: Q2 -- complete this function
        """ Return previous halfedge """
        return self.next.next

    def tip_vertex(self):
        # TODO: Q2 -- complete this function
        """ Return vertex on the tip of the halfedge """
        return self.next.vertex

    def serialize(self):
        return (
            self.index,
            self.vertex.index,
            self.edge.index,
            self.face.index,
            self.next.index,
            self.twin.index,
        )


class Edge(Primitive):
    """ initialization: assign halfedge and index (see Primitive) """
    
    def two_vertices(self):
        # TODO: Q2 -- complete this function
        """return the two incident vertices of the edge
        note that the incident vertices are ambiguous to ordering
        """
        return (self.halfedge.vertex, self.halfedge.tip_vertex())


class Face(Primitive):
    """ initialization: assign halfedge and index (see Primitive) """
    def adjacentHalfedges(self):
        # TODO: Q2 -- complete this function
        # Return ONLY the halfedges for which this face is assigned to. Be careful not to return the twins! 
        """ Return iterator of adjacent halfedges """
        return [self.halfedge, self.halfedge.next, self.halfedge.prev()]

    # Map is overpowered
    def adjacentVertices(self):
        # TODO: Q2 -- complete this function
        # Return all the vertices which are contained in this face 
        """ Return iterator of adjacent vertices """
        def getVertex(he):
            return he.vertex
        return list(map(getVertex, self.adjacentHalfedges()))

    def adjacentEdges(self):
        # TODO: Q2 -- complete this function
        # Return all the edges which make up this face 
        """ Return iterator of adjacent edges """
        def getEdge(he):
            return he.edge
        return list(map(getEdge, self.adjacentHalfedges()))
    
    def adjacentFaces(self):
        # TODO: Q2 -- complete this function
        # Return all the faces which share an edge with this face
        """ Return iterator of adjacent faces """
        def getAdjacentFace(he):
            return he.twin.face
        return list(map(getAdjacentFace, self.adjacentHalfedges()))

class Vertex(Primitive):
    """ initialization: assign halfedge and index (see Primitive) """
    
    def degree(self):
        # TODO: Q2 -- complete this function
        """ Return vertex degree: # of incident edges """
        return len(self.adjacentHalfedges())

    def isIsolated(self) -> bool:
        return self.halfedge is None

    def adjacentHalfedges(self):
        # TODO: Q2 -- complete this function
        # Return ONLY the halfedges for which this vertex is assigned to. Be careful not to return the twins! 
        """ Return iterator of adjacent halfedges """
        halfEdges = [self.halfedge] # Manifold assumption guarantees each vertex has a halfedge
        curHalfEdge = self.halfedge.twin.next # Manifold assumption guarantees .twin.next is defined
        while curHalfEdge != self.halfedge:
            halfEdges.append(curHalfEdge)
            curHalfEdge = curHalfEdge.twin.next # Manifold assumption guarantees .twin.next is defined
        return halfEdges
    
    def adjacentVertices(self):
        # TODO: Q2 -- complete this function
        # Return all the vertices which are connected to this vertex by an edge 
        """ Return iterator of adjacent vertices """
        def getAdjacentVertex(he):
            return he.tip_vertex()
        return list(map(getAdjacentVertex, self.adjacentHalfedges()))

    def adjacentEdges(self):
        # TODO: Q2 -- complete this function
        # Return all the edges which this vertex is contained in 
        """ Return iterator of adjacent edges """
        def getEdge(he):
            return he.edge
        return list(map(getEdge, self.adjacentHalfedges()))
    
    def adjacentFaces(self):
        # TODO: Q2 -- complete this function
        # Return all the faces which this vertex is contained in 
        """ Return iterator of adjacent faces """
        def getFace(he):
            return he.face
        return list(map(getFace, self.adjacentHalfedges()))
