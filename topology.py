import numpy as np
from collections import defaultdict
from primitive import Halfedge, Vertex, Edge, Face

class ElemCollection(dict):
    """This dict wrapper keeps track of the number of uniquly allocated elements
    so that each element has an unambiguous id in the lifetime of the mesh (even after edits)
    """
    def __init__(self, cons_f):
        super().__init__()
        self.cons_f = cons_f
        self.n_allocations = 0

    def allocate(self):
        elem = self.cons_f()
        i = self.n_allocations
        elem.index = i
        self[i] = elem
        self.n_allocations += 1
        return elem

    def fill_vacant(self, elem_id):
        """an element was previously deleted, and is now re-inserted"""
        assert elem_id not in self
        elem = self.cons_f()
        elem.index = elem_id
        self[elem_id] = elem
        return elem

    def compactify_keys(self):
        """fill the holes in index keys"""
        store = dict()
        for i, (_, elem) in enumerate(sorted(self.items())):
            store[i] = elem
            elem.index = i
        self.clear()
        self.update(store)

class Topology():
    def __init__(self):
        self.halfedges = ElemCollection(Halfedge)
        self.vertices = ElemCollection(Vertex)
        self.edges = ElemCollection(Edge)
        self.faces = ElemCollection(Face)

    def build(self, n_vertices, indices) -> bool:
        # TODO: Q1 -- complete this function  
        """
        This will be the primary function for generating your halfedge data structure. As the name suggests, the central element of 
        this data structure will be the Halfedge. Halfedges are related to each other through two operations `twin` and `next`. 
        Halfedge.twin returns the halfedge the shares the same edge as the current Halfedge, but is oppositely oriented
        (e.g. if halfedge H points from vertex A to vertex B, then H.twin points from vertex B to vertex A). Halfedge.next returns the
        next halfedge within the same triangle in the same orientation (e.g. given triangle ABC, if halfedge H goes A->B, then H.next 
        goes B -> C). With these properties alone, every halfedge can be associated with a specific face, vertex, and edge. Thus, in 
        your implementation every halfedge H should be assigned a Face, Vertex, and Edge element as attributes. Likewise, every Face, 
        Vertex, and Edge element should be assigned a halfedge H. Note that this relationship is not 1:1, so that there are multiple valid
        halfedges you can assign to Face, Vertex, and Edge. The choice is not important. As long as the orientation of the elements are 
        consistent across the mesh, then your implementation should work. 
        
        ======== VERY IMPORTANT =======
        In order for your implementation to pass our checks, you MUST allocate faces/halfedges in the following order 
        for {face array} in `indices` array:
            - If a {face array} contains vertex indices [i,j,k], then allocate halfedges/edges in the order (i,j), (j,k), (k, i)
            - If an edge has already been encountered, then set the new halfedge as the `twin` of the existing halfedge 
        """
        for _ in range(n_vertices):
            self.vertices.allocate()

        halfedgesSoFar = dict()
        for face in indices:
            f = self.faces.allocate()

            crntHalfEdges = []
            for _ in range(3):
                newHalfEdge = self.halfedges.allocate()
                crntHalfEdges.append(newHalfEdge)

            for i in range(3):
                vertexIndex = face[i]
                j = (i + 1) % 3
                he = crntHalfEdges[i]
                he.next = crntHalfEdges[j]
                v = self.vertices[vertexIndex]
                he.vertex = v
                v.halfedge = he
                f.halfedge = he
                he.face = f
                
                twinCoord = (face[j], face[i])
                if twinCoord in halfedgesSoFar:
                    twin = halfedgesSoFar[twinCoord]
                    twin.twin = he
                    he.edge = twin.edge
                    he.twin = twin
                else:
                    e = self.edges.allocate()
                    he.edge = e
                    e.halfedge = he
                    halfedgesSoFar[(face[i], face[j])] = he
        # self.thorough_check()
        return True

    def compactify_keys(self):
        self.halfedges.compactify_keys()
        self.vertices.compactify_keys()
        self.edges.compactify_keys()
        self.faces.compactify_keys()

    def export_halfedge_serialization(self):
        """
        this provides the unique, unambiguous serialization of the halfedge topology
        i.e. one can reconstruct the mesh connectivity from this information alone
        It can be used to track history, etc.
        """
        data = []
        for _, he in sorted(self.halfedges.items()):
            data.append(he.serialize())
        data = np.array(data, dtype=np.int32)
        return data

    def export_face_connectivity(self):
        face_indices = []
        for inx, face in self.faces.items():
            vs_of_this_face = []
            if face.halfedge is None:
                continue
            for vtx in face.adjacentVertices():
                vs_of_this_face.append(vtx.index)
            assert len(vs_of_this_face) == 3
            face_indices.append(vs_of_this_face)
        return face_indices

    def export_edge_connectivity(self):
        conn = []
        for _, edge in self.edges.items():
            if edge.halfedge is None:
                continue
            v1 = edge.halfedge.vertex
            v2 = edge.halfedge.next.next.vertex
            conn.append([v1.index, v2.index])
        return conn

    def hasNonManifoldVertices(self):
        # TODO: Q3 -- return True if any non-manifold vertices found, False otherwise
        
        # Method:
        # - vertex.adjacentHalfedges() cycles around a fan by starting at vertex.halfedge
        #   and continuing to add halfedges prev.twin.next until the starting halfedge is reached
        # - If any halfedge has this vertex and is not reached through this cycle, then
        #   there must be a second fan and the vertex is therefore nonmanifold
        fans = dict()
        for vertexIdx in self.vertices:
            vertex = self.vertices[vertexIdx]
            fans[vertex] = vertex.adjacentHalfedges()
        for halfEdgeIdx in self.halfedges:
            halfEdge = self.halfedges[halfEdgeIdx] 
            if not halfEdge in fans[halfEdge.vertex]:
                return True
        return False

    
    def hasNonManifoldEdges(self):
        # TODO: Q3 -- return True if any non-manifold edges found, False otherwise 

        # Methodology:
        # - A face being associated with an edge is equivalent to a
        #   unique halfedge being associated with an edge since faces
        #   are represented by halfedges that form a 3-cycle
        # - Thus, counting the halfedges that have a given edge is
        #   equivalent to counting the number of faces adjacent to an edge
        
        # Steps:
        # - Count halfedges that have each edge
        # - If any edge has a number of associated halfedges < 1 or > 2, 
        #   that edge is nonmanifold

        edges = dict()
        for edgeIdx in self.edges:
            edges[edgeIdx] = []
        for halfedgeIdx in self.halfedges:
            halfedge = self.halfedges[halfedgeIdx]
            if not halfedge.face.index in edges[halfedge.edge.index]:
                edges[halfedge.edge.index].append(halfedge.face.index)
        for edgeIdx in edges:
            if len(edges[edgeIdx]) < 1 or len(edges[edgeIdx]) > 2:
                return True
        return False

    def thorough_check(self):
        # Check full halfedge coverage across all mesh elements  
        if len(self.halfedges) == 0 or len(self.vertices) == 0 or len(self.faces) == 0 or len(self.edges) == 0: 
            print(f"Warning: Topology is incomplete. You need to allocate halfedge, vertex, face, and edge elements in order for the checker to work.")
            return 
            
        def check_indexing(src_dict):
            for inx, v in src_dict.items():
                assert inx == v.index

        check_indexing(self.halfedges)
        check_indexing(self.vertices)
        check_indexing(self.edges)
        check_indexing(self.faces)

        # 2. check edges
        self._check_verts()
        self._check_edges()
        self._check_faces()
  
    def _check_verts(self):
        encountered_halfedges = []
        for inx, v in self.vertices.items():
            hes = []
            for he in v.adjacentHalfedges():
                # if he.vertex != v:
                #     return False, he, v
                assert he.vertex == v
                hes.append(he)
            encountered_halfedges.extend([elem.index for elem in hes])
        encountered_halfedges = set(encountered_halfedges)
        assert encountered_halfedges == set(self.halfedges.keys()), "must cover all halfedges"
        # return True, True, True
    def _check_edges(self):
        encountered_halfedges = []
        for inx, e in self.edges.items():
            he = e.halfedge
            twin = he.twin

            hes = [he, twin]
            n = len(hes)

            for i, he in enumerate(hes):
                assert he.edge == e
                assert he.twin == hes[(i + 1) % n]

            encountered_halfedges.extend([elem.index for elem in hes])

        encountered_halfedges = set(encountered_halfedges)
        assert encountered_halfedges == set(self.halfedges.keys()), "must cover all halfedges"

    def _check_faces(self):
        encountered_halfedges = []
        for inx, f in self.faces.items():
            hes = []
            for he in f.adjacentHalfedges():
                hes.append(he)

            n = len(hes)
            for i, he in enumerate(hes):
                assert he.face == f
                assert he.next == hes[(i + 1) % n]

            encountered_halfedges.extend([elem.index for elem in hes])

        encountered_halfedges = set(encountered_halfedges)
        target_halfedges = {
            k for k, v in self.halfedges.items()
        }
        assert encountered_halfedges == target_halfedges, \
            f"must cover all halfedges"