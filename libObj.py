import numpy as np

class libObj:
    def __init__(self, filepath):
        with open(filepath) as f:
            self.lines = f.readlines()
            pnts = [[float(y) for y in x.split()[1:]] for x in self.lines if x.startswith("v ")]
            uvs = [[float(y) for y in x.split()[1:]] for x in self.lines if x.startswith("vt ")]
            normals = [[float(y) for y in x.split()[1:]] for x in self.lines if x.startswith("vn ")]
            self.pnts = np.array(pnts)
            self.uvs = np.array(uvs) if len(uvs) else None
            self.normals = np.array(normals) if len(normals) else None
            self.faceLines = [[y for y in x.split()[1:]] for x in self.lines if x.startswith("f ")]
            self.faces = [[int(y.split("/")[0])-1 for y in x]for x in self.faceLines]
            self.uvFaces = [[int(y.split("/")[1])-1 for y in x]for x in self.faceLines] if len(uvs) else None
            self.normalFaces = [[int(y.split("/")[2])-1 for y in x]for x in self.faceLines] if len(normals) else None

            # to convert uv idx to vtx idx
            self.uvIdToVtxId = np.zeros(len(self.pnts), dtype=int)
            for i in range(len(self.faces)):
                for j in range(len(self.faces[i])):
                    self.uvIdToVtxId[self.faces[i][j]] = self.uvFaces[i][j]

    def save(self, filepath, pnts=None):
        with open(filepath, "w") as f:
            if pnts is None:
                pnts = self.pnts
            for v in pnts:
                f.write("v %f %f %f\n"%(v[0], v[1], v[2]))
            if self.uvs is not None:
                for uv in self.uvs:
                    f.write("vt %f %f\n"%(uv[0], uv[1]))
            if self.normals is not None:
                for n in self.normals:
                    f.write("n %f %f %f\n"%(n[0], n[1], n[2]))
            s = [[str(x+1) for x in y] for y in self.faces]
            if self.uvs is not None or self.normals is not None:
                s = [[x+"/" for x in y] for y in s]
                if self.uvs is not None:
                    s = [[s[i][j]+str(self.uvFaces[i][j]+1) for j in range(len(s[i]))] for i in range(len(s))]
                s = [[x+"/" for x in y] for y in s]
                if self.normals is not None:
                    s = [[s[i][j]+str(self.normalFaces[i][j]+1) for j in range(len(s[i]))] for i in range(len(s))]
            s = ['f %s\n'%' '.join(x) for x in s]
            f.writelines(s)
    
    def setPnts(self, pnts):
        pnts = np.array(pnts)
        if pnts.shape != self.pnts.shape:
            print("setPnts() - shape mismatch")
        else:
            self.pnts = pnts
