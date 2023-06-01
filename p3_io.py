import numpy as np

class PolygonSoup():
    """
    We define a triangular polygon soup as a collection of spatial points (vertices),
    and their connectivity information.
    The fields are:
        vertices: [N, 3]
        indices: [M, 3] where each row is the indices of the 3 vertices that make up a face

    This can be read from and written to in various file formats e.g obj, stl
    """
    def __init__(self, vertices, indices):
        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.uint32)
        assert vertices.shape[1] == 3, "`vertices` must be an Nx3 array"
        assert indices.shape[1] == 3, "`faces` must be an Mx3 array"
        self.vertices = vertices
        self.indices = indices

    @classmethod
    def from_obj(cls, fname):
        """
        # An obj file looks like:
        v 0.123 0.234 0.345
        vn 0.707 0.000 0.707
        f 1 2 3
        # each line could be vertex_index/texture_index. Only need vertex_index
        f 3/1 4/2 5/3

        We can recompute normals "vn" ourselves
        """
        vertices = []
        indices = []
        with open(fname, "r") as f_handle:
            for line in f_handle:
                line = line.strip()
                tokens = line.split(" ")
                identifier = tokens[0]

                if identifier == "v":
                    vertices.append(
                        [float(tokens[1]), float(tokens[2]), float(tokens[3])]
                    )
                elif identifier == "f":
                    assert len(tokens) == 4,\
                        f"only triangle meshes are supported, got face index {line}"

                    face_indices = []
                    for i in range(3):
                        inx = tokens[1 + i].split("/")[0]  # throw away texture index, etc
                        inx = int(inx)
                        # NOTE obj index is 1-based
                        # theoretically negatives are allowed in the spec; but hell
                        assert (inx > 0), "index should be positive"
                        face_indices.append(inx - 1)
                    indices.append(face_indices)

        return cls(vertices, indices)

    def to_obj(self, fname):
        with open(fname, "w") as f_handle:
            for vi, v in enumerate(self.vertices):
                f_handle.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            for face_id in range(len(self.indices) - 1):
                f_handle.write("f %d %d %d\n" % (self.indices[face_id][0] + 1, self.indices[face_id][1] + 1, self.indices[face_id][2] + 1))
            f_handle.write("f %d %d %d" % (self.indices[-1][0] + 1, self.indices[-1][1] + 1, self.indices[-1][2] + 1))
