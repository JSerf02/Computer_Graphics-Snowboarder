from . import Halfedge, Edge, Vertex, Face, Topology, Mesh
import numpy as np


class MeshEdit():
    def __init__(self):
        pass

    def apply(self):
        raise NotImplementedError()

    def inverse(self):
        raise NotImplementedError()

class LaplacianSmoothing(MeshEdit):
    def __init__(self, mesh: Mesh, n_iter: int):
        self.mesh = mesh
        self.n_iter = n_iter
    
    def _apply(self): 
        newVertexPositions = []
        for vertexIdx in self.mesh.topology.vertices:
            vertex = self.mesh.topology.vertices[vertexIdx] 
            adjacentVertices = list(map(self.mesh.get_3d_pos, vertex.adjacentVertices()))
            average = [0, 0, 0]
            for adjVertex in adjacentVertices:
                for i in range(3):
                    average[i] += adjVertex[i]
            average = np.array(average) * 1 / len(adjacentVertices) 
            newVertexPositions.append(average)
        
        for idx in range(len(newVertexPositions)):
            for i in range(3):
                self.mesh.vertices[idx][i] = newVertexPositions[idx][i]
        
        return True 

    def apply(self):
        res = True
        for _ in range(self.n_iter):
            res = res and self._apply()
            if not res:
                break
        return res

# TODO: Q6 -- complete this 
class EdgeCollapse(MeshEdit):
    def __init__(self, mesh: Mesh, e_id: int):
        self.mesh = mesh
        self.e_id = e_id

    def apply(self):
        return next(do_collapse(self.mesh, self.e_id))
  
# TODO: Q6 -- complete this 
def do_collapse(mesh, e_id):
    topology = mesh.topology
    e = topology.edges[e_id]

    c = e.halfedge.vertex
    d = e.halfedge.tip_vertex()

    cd = e.halfedge
    dc = cd.twin

    bc = dc.next.twin
    ca = cd.prev().twin

    cAdjHalfedges = c.adjacentHalfedges()

    facesToDelete = []
    halfedgesToDelete = []
    edgesToDelete = []
    
    # Plan to delete faces acd and bcd
    facesToDelete.append(cd.face)
    facesToDelete.append(dc.face)

    # Plan to delete halfedges in face acd
    halfedgesToDelete.append(cd)
    halfedgesToDelete.append(cd.next)
    halfedgesToDelete.append(cd.prev())

    # Plan to delete halfedges in face bcd
    halfedgesToDelete.append(dc)
    halfedgesToDelete.append(dc.next)
    halfedgesToDelete.append(dc.prev())

    # Plan to delete edges cd, bc, and ca
    edgesToDelete.append(e)
    edgesToDelete.append(bc.edge)
    edgesToDelete.append(ca.edge)

    # Join halfedge bc with halfedge db
    bc.twin = dc.prev().twin
    bc.twin.twin = bc
    bc.edge = bc.twin.edge

    # Join halfedge ca with halfedge ad
    ca.twin = cd.next.twin
    ca.twin.twin = ca 
    ca.edge = ca.twin.edge

    # Merge vertex d into vertex c
    for halfedge in cAdjHalfedges:
        halfedge.vertex = d
    
    # Move vertex c to average(c, d)
    mesh.vertices[d.index] = np.mean(np.array([mesh.vertices[c.index], mesh.vertices[d.index]]), axis=0)
    
    # Make sure vertices and edges point to halfedges that aren't getting removed
    # - Since we are only removing halfedges from faces that are getting removed, 
    #   there will never be issues with faces pointing to deleted halfedges
    for halfedge in halfedgesToDelete:
        if halfedge.vertex.index != c.index:
            while halfedge.vertex.halfedge in halfedgesToDelete:
                halfedge.vertex.halfedge = halfedge.vertex.halfedge.twin.next
        if not halfedge.edge in edgesToDelete:
            while halfedge.edge.halfedge in halfedgesToDelete:
                halfedge.edge.halfedge = halfedge.edge.halfedge.twin

    # Delete previously determined faces, edges, and halfedges 
    # - These deletions were saved for the end so the order of computation doesn't make a difference
    while len(facesToDelete) > 0:
        idx = facesToDelete.pop().index
        del topology.faces[idx]
    while len(edgesToDelete) > 0:
        idx = edgesToDelete.pop().index
        del topology.edges[idx]
    while len(halfedgesToDelete) > 0:
        idx = halfedgesToDelete.pop().index
        del topology.halfedges[idx]

    # Delete vertex c
    mesh.vertices = np.delete(mesh.vertices, c.index, 0)
    del topology.vertices[c.index]

    # Realign indices to remove bad keys
    topology.halfedges.compactify_keys()
    topology.vertices.compactify_keys()
    topology.faces.compactify_keys()

    mesh.indices = topology.export_face_connectivity()
    yield True # Wooo, the hell is over!
  
# TODO: Extra credit -- complete this 
class EdgeCollapseWithLink(MeshEdit):
    def __init__(self, mesh: Mesh, e_id: int):
        self.mesh = mesh
        self.e_id = e_id
        event = do_collapse_with_link(mesh, e_id)
        self.do_able = next(event)
        if self.do_able is False:
            return
        self.event = event

    def apply(self):
        return next(self.event)

# TODO: Extra credit -- complete this 
def do_collapse_with_link(mesh, e_id):
    topology = mesh.topology
    e = topology.edges[e_id]

    c = e.halfedge.vertex
    d = e.halfedge.tip_vertex()

    yield linkCondition(mesh, c, d, e)

    yield next(do_collapse(mesh, e_id))

def linkCondition(mesh: Mesh, c: Vertex, d: Vertex, cd: Edge):
    cVertices = c.adjacentVertices()
    dVertices = d.adjacentVertices()

    cIntersectD = []
    for vertex in cVertices:
        if vertex in dVertices:
            cIntersectD.append(vertex)

    cdVertices = []
    cdFaces = [cd.halfedge.face, cd.halfedge.twin.face]

    for face in cdFaces:
        faceVertices = face.adjacentVertices()
        for vertex in faceVertices:
            if vertex.index != c.index and vertex.index != d.index:
                cdVertices.append(vertex)
    if len(cIntersectD) != len(cdVertices):
        return False
    for vertex in cIntersectD:
        if not vertex in cdVertices:
            return False
    return True