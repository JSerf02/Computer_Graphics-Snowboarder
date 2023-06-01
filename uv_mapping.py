import math
import numpy as np
from mesh import Mesh
from topology import Topology
from primitive import Vertex, Face, Edge, Halfedge
from p3_io import PolygonSoup

class NumpyHelpers:
    def magnitude(array : np.ndarray):
        return np.linalg.norm(array)

    def round_coords(coords, n = 5):
        return list(map((lambda coord: round(coord, n)), coords))

    def trunc_coords(coords, n = 5):
        def truncate(f, n): # https://stackoverflow.com/questions/783897/how-to-truncate-float-values
            '''Truncates/pads a float f to n decimal places without rounding'''
            s = '{}'.format(f)
            if 'e' in s or 'E' in s:
                return '{0:.{1}f}'.format(f, n)
            i, p, d = s.partition('.')
            return '.'.join([i, (d+'0'*n)[:n]])
        return list(map((lambda coord: truncate(coord, n)), coords))
    
    def to_tuple(array : np.ndarray, n = 5, trunc=False):
        arr = array.reshape(1, -1)[0]
        return tuple(NumpyHelpers.trunc_coords(arr, n)) if trunc else tuple(NumpyHelpers.round_coords(arr, n)) 
    
    
# Adds seams to a mesh (terminology based on Blender)
# - A seam is an edge that is duplicated so each halfedge of that edge gets its own edge
#   - Conceptually, you can imagine this as cutting along an edge with scissors.
# - Whenever 2 pairs of duplicate edges share a vertex, that vertex will be split into 2 
#   - For example. imagine that you just took scissors and cut two edges of a cube 
#     that share a corner. This behavior would have you also cut that corner so you
#     can freely fold the new face formed by those 2 cut edges.
class Seams:
    # Duplicates an edge so each halfedge can have its own edge
    # - "Cuts the edge with scissors"
    def add_edge_seam(mesh: Mesh, edge_idx):
        edge1 = mesh.topology.edges[edge_idx]
        
        # Can't create a second edge if there are not 2 halfedges
        if edge1.halfedge == None or edge1.halfedge.twin == None:
            return
        
        # Create the new edge and distribute halfedges accordingly
        edge2 = mesh.topology.edges.allocate()
        edge2.halfedge = edge1.halfedge.twin
        edge1.halfedge.twin = None
        edge2.halfedge.edge = edge2
        edge2.halfedge.twin = None
    
    # Duplicates corners shared between edge seams
    # - "Cuts the corner with scissors so you can freely fold the face"
    def add_vertex_seams(mesh : Mesh):
        vertex_halfedges = dict()
        for halfedge_idx in mesh.topology.halfedges:
            halfedge = mesh.topology.halfedges[halfedge_idx]
            if not halfedge.vertex.index in vertex_halfedges:
                vertex_halfedges[halfedge.vertex.index] = []
            vertex_halfedges[halfedge.vertex.index].append(halfedge_idx)
        
        # Vertices are split whenever 2 seams share a vertex
        for halfedge_idx in mesh.topology.halfedges:
            halfedge = mesh.topology.halfedges[halfedge_idx]
            
            # Make sure the current halfedge is a part of a seam
            if halfedge.twin != None:
                continue

            # Check if there are any halfedges that are seams that start on the next vertex
            base = halfedge.next
            cur = base
            nexts = [cur]
            res = True
            while cur.twin != None:
                cur = cur.twin.next
                if cur == base:
                    res = False
                    break
                nexts.append(cur)
            
            # If there are no other seams or the only other halfedge that is a seam 
            # was originally the current halfedge's twin, do not proceed
            if not res or (mesh.vertices[cur.next.vertex.index] == mesh.vertices[halfedge.vertex.index]).all():
                continue

            # Make sure the old vertex does not point to a halfedge that will be assigned
            # a new vertex
            # - Also makes sure corners are not split multiple times for the same seams
            if base.vertex.halfedge in nexts:
                res = False
                for he_idx in vertex_halfedges[base.vertex.index]:
                    if not mesh.topology.halfedges[he_idx] in nexts:
                        base.vertex.halfedge = mesh.topology.halfedges[he_idx]
                        res = True
                        break
                if not res:
                    continue
            
             # Add a new vertex 
            new_vtx = mesh.topology.vertices.allocate()
            new_vtx.halfedge = base
            new_vertices = list(mesh.vertices)
            new_vertices.append(mesh.vertices[new_vtx.halfedge.vertex.index])
            mesh.vertices = np.array(new_vertices)

            # For each halfedge between the two seams, set the halfedge's vertex
            # to the new vertex
            prev_vtx_idx = base.vertex.index
            vertex_halfedges[new_vtx.index] = []
            cur2 = base
            while cur2 != cur:
                vertex_halfedges[prev_vtx_idx].remove(cur2.index)
                vertex_halfedges[new_vtx.index].append(cur2.index)
                cur2.vertex = new_vtx
                cur2 = cur2.twin.next
            vertex_halfedges[prev_vtx_idx].remove(cur.index)
            vertex_halfedges[new_vtx.index].append(cur.index)
            cur.vertex = new_vtx
    
    # Adds multple seams to a mesh 
    def add_seams(mesh : Mesh, edge_indices):
        for edge_idx in edge_indices:
            Seams.add_edge_seam(mesh, edge_idx)
        Seams.add_vertex_seams(mesh)
    
    # Finds the edges with the inputted vertices for each inputted pair of vertices
    def get_edges_from_vertices(mesh : Mesh, vertex_pairs):
        vertices_map = dict()
        for vtx_idx in mesh.topology.vertices:
            vertices_map[NumpyHelpers.to_tuple(np.array([float(mesh.vertices[vtx_idx][0]), float(mesh.vertices[vtx_idx][1]), float(mesh.vertices[vtx_idx][2])]), 5)] = vtx_idx
        
        edges = []
        for pair in vertex_pairs:
            vertex0 = np.array(NumpyHelpers.round_coords(pair[0]))
            vertex1 = np.array(NumpyHelpers.round_coords(pair[1]))

            if np.allclose(vertex0, vertex1, atol=1e-5):
                continue
            
            idx = vertex0

            # Make sure floating point errors don't cause this to fail
            if not tuple(idx) in vertices_map:
                res = False
                def generate_patterns(n = 3):
                    if n <= 0:
                        return []
                    if n == 1:
                        return [[0], [1], [-1]]
                    prev = generate_patterns(n - 1)
                    return list(map(lambda p: [0] + p, prev)) + list(map(lambda p: [1] + p, prev)) + list(map(lambda p: [-1] + p, prev))
                patterns = generate_patterns()
                for pattern in patterns:
                    np_pattern = np.array(tuple(pattern))
                    test_idx = idx + np_pattern * 1e-5
                    if NumpyHelpers.to_tuple(test_idx) in vertices_map:
                        idx = test_idx
                        res = True
                        break
                assert(res)
            
            top_vertex = mesh.topology.vertices[vertices_map[NumpyHelpers.to_tuple(idx)]]
            
            start_halfedge = top_vertex.halfedge
            cur_halfedge = top_vertex.halfedge
            while not np.allclose(np.array(NumpyHelpers.round_coords(mesh.vertices[cur_halfedge.tip_vertex().index])), vertex1, atol=1e-5):
                cur_halfedge = cur_halfedge.twin.next
                assert(cur_halfedge != start_halfedge)
            edges.append(cur_halfedge.edge.index)
        return edges
    
    # Finds edges that contain each inputted pair of vertices and adds seams on 
    # each of these edges
    def add_seams_vertices(mesh : Mesh, vertex_pairs):
        Seams.add_seams(mesh, Seams.get_edges_from_vertices(mesh, vertex_pairs))
        
                
class Least_Square_Conformal_Map:
    # Turns a triangle into local coordinates using an orthonormal basis
    # [A, B, C] |=> [(0, 0), (1, 0), ( ((C - A) . (B - A)) / |B - A|^2, |(C - A) - this.x * (B - A)| / |B - A| )]
    def convert_triangle_to_local(input_vertices):
        vertices = list(map((lambda vertex: np.array(vertex)), input_vertices))

        local_vertices = [(0, 0), (1, 0)]

        second_vec = vertices[1] - vertices[0]
        mag_second_vec = NumpyHelpers.magnitude(second_vec)

        third_vec = vertices[2] - vertices[0]
        local_x = np.dot(third_vec, second_vec) / (mag_second_vec * mag_second_vec)
        third_vec_global_y = third_vec - local_x * second_vec
        local_y = NumpyHelpers.magnitude(third_vec_global_y) / mag_second_vec
        local_vertices.append((local_x, local_y))
        
        return local_vertices

    def double_triangle_area(vertices):
        result =  vertices[0][0] * vertices[1][1] - vertices[0][1] * vertices[1][0]
        result += vertices[1][0] * vertices[2][1] - vertices[1][1] * vertices[2][0]
        result += vertices[2][0] * vertices[0][1] - vertices[2][1] * vertices[0][0]
        return result

    def get_map(mesh : Mesh):
        # Sort the vertices to find 2 vertices that are far apart. Place them at the end
        sorted_vertices = [vtx_idx for vtx_idx in mesh.topology.vertices]
        sorted_vertices.sort(key=lambda vtx_idx: NumpyHelpers.to_tuple(np.array(mesh.vertices[vtx_idx])))
        sorted_vertices = sorted_vertices[1:] + [sorted_vertices[0]] 

        # Store which index each regular vertex is in the sorted vertices list
        num_vertices = len(sorted_vertices)
        sorted_vertices_map = dict()
        for i in range(num_vertices):
            cur_vtx_idx = sorted_vertices[i]
            sorted_vertices_map[cur_vtx_idx] = i
        
        # Create matrices M_1 and M_2 using the local coordinates of the vertices
        # of each face in the mesh in terms off an orthonormal basis
        # - M_1[face][vtx] = 
        #   - 0 if vtx is not in face
        #   - (x_3 - x_2) / sqrt(2A) if vtx is the first index in the face
        #      - A is the area of the face
        #   - (x_1 - x_3) / sqrt(2A) if vtx is the second index in the face
        #   - (x_2 - x_1) / sqrt(2A) if vtx is the third index in the face
        # - M_2 is defined identically to M_1 except with y instead of x
        M_1 = np.zeros((len(mesh.topology.faces), num_vertices), dtype=np.float64)
        M_2 = np.zeros((len(mesh.topology.faces), num_vertices), dtype=np.float64)
        for idx in range(len(mesh.topology.faces)):
            # Gets the local coordinates of the vertices of the current face
            face = mesh.topology.faces[idx]
            face_vertices = [vertex.index for vertex in face.adjacentVertices()]
            global_vertices = list(map((lambda vtx_idx : mesh.vertices[vtx_idx]), face_vertices))
            local_vertices = Least_Square_Conformal_Map.convert_triangle_to_local(global_vertices)

            # Calculates sqrt(2A)
            sqrt_of_double_area = math.sqrt(Least_Square_Conformal_Map.double_triangle_area(local_vertices))
            
            # Determines which indices in the matrix these vertices correspond to
            # - The vertices are ordered according to the sorted vertices list
            idx_0 = sorted_vertices_map[face_vertices[0]]
            idx_1 = sorted_vertices_map[face_vertices[1]]
            idx_2 = sorted_vertices_map[face_vertices[2]]

            # Sets the current indices of M_1 and M_2
            M_1[idx][idx_0] = (local_vertices[2][0] - local_vertices[1][0]) / sqrt_of_double_area
            M_2[idx][idx_0] = (local_vertices[2][1] - local_vertices[1][1]) / sqrt_of_double_area

            M_1[idx][idx_1] = (local_vertices[0][0] - local_vertices[2][0]) / sqrt_of_double_area
            M_2[idx][idx_1] = (local_vertices[0][1] - local_vertices[2][1]) / sqrt_of_double_area

            M_1[idx][idx_2] = (local_vertices[1][0] - local_vertices[0][0]) / sqrt_of_double_area
            M_2[idx][idx_2] = (local_vertices[1][1] - local_vertices[0][1]) / sqrt_of_double_area

        # Splits M_1 and M_2 each into 2 matrices: One that contains the 2 fixed points
        # and one that does not
        M_1_f = M_1[:, :-2]
        M_2_f = M_2[:, :-2]
        M_1_p = M_1[:, -2:]
        M_2_p = M_2[:, -2:]
        
        # Solves the least squares equation using these matrices to get uv coordinates 
        # for the non-fixed points
        # - Always maps the 2 far apart points to (0, 0) and (1, 1)
        A = np.block([[M_1_f, -M_2_f], [M_2_f, M_1_f]])
        b = -1 * np.block([[M_1_p, -M_2_p], [M_2_p, M_1_p]]) @ np.array([[0, 1, 0, 1]]).T
        x = np.linalg.lstsq(A, b)[0]

        # Stores all of the uv coordinates in a dict for easier access
        uvs = dict()
        uvs[sorted_vertices[num_vertices - 2]] = (0, 0)
        uvs[sorted_vertices[num_vertices - 1]] = (1, 1)
        for i in range(num_vertices - 2):
            uvs[sorted_vertices[i]] = (x[i][0], x[i + num_vertices - 2][0])

        # Normalizes the uv coordinates so u and v are both in [0, 1]
        min_val = min([min(val[0], val[1]) for val in uvs.values()])
        max_val = max([max(val[0], val[1]) for val in uvs.values()]) - min_val 
        for i in range(num_vertices):
            uvs[i] = (round((uvs[i][0] - min_val) / max_val, 5), round((uvs[i][1] - min_val) / max_val, 5))
        
        return uvs

# A simple interface for importing, uv-mapping, and exporting meshes
class UV_Map:
    # Exports a uv map (vertices, uvs, and faces)
    def export(mesh : Mesh, uvs, output_name):
        path = output_name + ".obj"
        with open(path, 'w') as f:
            # Export sorted vertices with duplicates removed 
            sorted_vertices = list(map(NumpyHelpers.to_tuple, mesh.vertices))
            sorted_vertices.sort()
            sorted_vertices = [*set(sorted_vertices)]
            for v in sorted_vertices:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            
            # Store the indices of each vertex in the sorted list for face calculations
            vertex_indices = {}
            for idx in range(len(mesh.vertices)):
                for sorted_idx in range(len(sorted_vertices)):
                    if NumpyHelpers.to_tuple(mesh.vertices[idx]) == sorted_vertices[sorted_idx]:
                        vertex_indices[idx] = sorted_idx
                        break
            
            # Export sorted uvs with duplicates removed
            sorted_uvs = list(uvs.values())
            sorted_uvs.sort()
            sorted_uvs = [*set(sorted_uvs)]
            for uv in sorted_uvs:
                f.write("vt %f %f\n" % (uv[0], uv[1]))

            # Store the indices of each uv in the sorted list for face calculations
            uv_indices = {}
            for idx in range(len(uvs)):
                for sorted_idx in range(len(sorted_uvs)):
                    uv_val = sorted_uvs[sorted_idx]
                    if uvs[idx] == uv_val:
                        uv_indices[idx] = sorted_idx
                        break
            
            # Export all faces
            for face_idx in mesh.topology.faces:
                face = mesh.topology.faces[face_idx]
                halfedge = face.halfedge
                export_face = []
                for _ in range(3):
                    vtx = vertex_indices[halfedge.vertex.index]
                    uv = uv_indices[halfedge.vertex.index]
                    export_face.append((vtx + 1, uv + 1))
                    halfedge = halfedge.next
                f.write("f %d/%d %d/%d %d/%d\n" % (export_face[0][0], export_face[0][1], export_face[1][0], export_face[1][1], export_face[2][0], export_face[2][1]))

    # Imports a mesh, adds edge seams for inputted edges, creates a UV map with LSCM, and exports the mesh
    def map(input_name, seam_edge_indices=None, output_name=None, uv_scale=1, uv_offset=(0, 0)):
        # Default file name
        if(output_name == None):
            output_name = input_name + "_mapped"
        
        input_path = input_name + ".obj"
        soup = PolygonSoup.from_obj(input_path)
        mesh = Mesh(soup.vertices, soup.indices)

        if seam_edge_indices != None:
            Seams.add_seams(mesh, seam_edge_indices)
        uvs = Least_Square_Conformal_Map.get_map(mesh)
        uvs = dict(map(lambda item : (item[0], (item[1][0] * uv_scale + uv_offset[0], item[1][1] * uv_scale + uv_offset[1])), uvs.items()))
        UV_Map.export(mesh, uvs, output_name)
    
    # Imports a mesh, adds edge seams for inputed vertex pairs, creates a UV map with LSCM, and exports the mesh
    def map_with_vertices(input_name, seam_vertex_pairs=None, output_name=None, just_print=False, uv_scale=1, uv_offset=(0, 0)):
        if(output_name == None):
            output_name = input_name + "_mapped"
        input_path = input_name + ".obj"
        soup = PolygonSoup.from_obj(input_path)
        mesh = Mesh(soup.vertices, soup.indices)
        if just_print:
            edges = Seams.get_edges_from_vertices(mesh, seam_vertex_pairs)
            print(edges)
            f = open(output_name + ".txt", "w")
            f.write(str(edges))
            f.close()
            return edges
        if seam_vertex_pairs != None:
            Seams.add_seams_vertices(mesh, seam_vertex_pairs)
        uvs = Least_Square_Conformal_Map.get_map(mesh)
        uvs = dict(map(lambda item : (item[0], (item[1][0] * uv_scale + uv_offset[0], item[1][1] * uv_scale + uv_offset[1])), uvs.items()))
        UV_Map.export(mesh, uvs, output_name)

