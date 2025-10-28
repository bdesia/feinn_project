class BaseSolver:
    
    def __init__(self, mesh, bcs, geoparams, matfld):
        self.mesh = mesh
        self.bcs = bcs
        self.geoparams = geoparams
        self.matfld = matfld

class nlfea(BaseSolver):

    def run(self):
    