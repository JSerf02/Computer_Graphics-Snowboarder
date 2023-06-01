import numpy as np
from primitive import Halfedge, Edge, Vertex, Face
from topology import Topology

"""
NOTE: We will NOT deal with boundary loops 
"""

class Mesh:
    def __init__(self, vertices, face_indices):
        self.vertices = vertices
        self.indices = face_indices 
        self.topology = Topology()
        self.topology.build(len(vertices), face_indices)

    def export_soup(self):
        init_n = len(self.vertices)
        face_conn = np.array(self.topology.export_face_connectivity(), dtype=np.uint32)
        edge_conn = np.array(self.topology.export_edge_connectivity(), dtype=np.uint32)

        old_inds = np.array(sorted(self.topology.vertices.keys()))
        new_inds = np.arange(len(old_inds), dtype=int)
        vertices = self.vertices[old_inds]
        A = np.zeros(init_n, dtype=np.uint32)
        A[old_inds] = new_inds

        face_conn = A[face_conn]
        edge_conn = A[edge_conn]
        return vertices, face_conn, edge_conn

    # TODO: Q4
    def get_3d_pos(self, v: Vertex):
        """ Given a vertex primitive, return the position coordinates """
        return self.vertices[v.index]

    # TODO: Q4
    def vector(self, h: Halfedge):
        """ Given a halfedge primitive, return the vector """
        start = self.get_3d_pos(h.vertex)
        end = self.get_3d_pos(h.tip_vertex())
        return np.array([end[0] - start[0], end[1] - start[1], end[2] - start[2]])

    # TODO: Q4
    def faceNormal(self, f: Face):
        """ Given a face primitive, compute the unit normal """
        normal = np.cross(self.vector(f.halfedge), self.vector(f.halfedge.prev()) * -1)
        normalLength = np.linalg.norm(normal)
        if normalLength == 0 or normalLength == 1:
            return normal
        return normal * (1 / normalLength)
    
    # TODO: Q5
    def smoothMesh(self, n=5):
        """ Laplacian smooth mesh n times """
        from . import LaplacianSmoothing
        LaplacianSmoothing(self, n).apply()
        self.export_obj("p5.obj")
    
    # TODO: Q6
    def collapse(self, edgelist):
        """ Edge collapse without link condition check """
        from . import EdgeCollapse
        for edge in edgelist:
            edt = EdgeCollapse(self, edge)
            edt.apply()

        # for i in range(len(self.vertices)):
        #     self.vertices[i] = better_round(self.vertices[i])
            # self.vertices[i] = np.round(self.vertices[i], 6)
        self.export_obj("p6.obj")
          
    # TODO: Extra credit 
    def collapse_with_link_condition(self, edgelist):
        """ Extra credit: collapse with link condition check """
        from . import EdgeCollapseWithLink
        for edge in edgelist:
            edt = EdgeCollapseWithLink(self, edge)
            if edt.do_able:
                edt.apply() 
        self.export_obj("ec.obj")
        
    def view(self):
        """ Mesh viewer using polyscope """
        import polyscope as ps 
        ps.init() 
        ps_mesh = ps.register_surface_mesh("mesh", self.vertices, self.indices, edge_width=1)
        ps.show() 
    
    def export_obj(self, path):
        vertices, faces, edges = self.export_soup()
        with open(path, 'w') as f:
            for vi, v in enumerate(vertices):
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            for face_id in range(len(faces) - 1):
                f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            for edge in edges:
                f.write("\ne %d %d" % (edge[0] + 1, edge[1] + 1))
