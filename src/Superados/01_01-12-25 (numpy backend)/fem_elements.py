
import numpy as np
from utils import QuadShapeFunctions, QuadShapeDerivatives, GaussQuad

class MasterElement:
    ' Base class for master finite elements'

    def __init__(self, id, nodes, material = None):
        self.id = id                                                            # Element ID   
        self.nodes = nodes                                                      # List of nodes 
        self.material = material                                                # Material object


class QuadElement(MasterElement):
    """
    Quadrilateral element in 2D for continua
    Supports both Total Lagrangian formulation for geometric nonlinearity
    and geometrically linear analysis.
    """
    def __init__(self, id: int, nodes: list, material= None, thickness=1.0):
        super().__init__(id, nodes, material)
        self.id = id                                                            # Element ID   
        self.nodes = nodes                                                      # List of nodes connectivity
        self.material = material                                                # Material object: could be defined later
        self.thickness = thickness                                              # Element thickness (just in case of 2D plane stress)

        self.nnode = len(nodes)                                                 # Number of nodes
        self.ngp = 2 if self.nnode == 4 else 3                                  # Number of Gauss points 
        self.ndof_local = self.nnode * 2                                        # Number of local dofs

        self.dofs = self._get_dof_indices()                                     # Local dof indices in global system

        # Define element node order for plotting edges
        if self.nnode == 4:
            self.edge_order = [0, 1, 2, 3, 0]  # Q4: top-right, top-left, bottom-left, bottom-right, close
        elif self.nnode == 8:
            self.edge_order = [0, 4, 1, 5, 2, 6, 3, 7, 0]  # Q8: top-right, mid-top, top-left, mid-left, ...
        elif self.nnode == 9:
            self.edge_order = [0, 4, 1, 5, 2, 6, 3, 7, 0]  # Q9: same as Q8 for edges, center node (8) ignored
        
        # self.X = np.array([node.coordinates for node in nodes])                 # Original coordinates [4, 2]
    
    # --------------------------------------
    #  AUXILIARY METHODS
    # --------------------------------------

    def _get_dof_indices(self):
        dofs = []
        for node in self.nodes:
            dofs.extend([2 * (node - 1), 2 * (node - 1) + 1])
        return dofs

    def get_nodal_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        ' Get nodal coordinates from global coordinate array '
        self.X = np.array([coordinates[node - 1] for node in self.nodes])       # [nnode, 2-coord]

    def reduced_integration(self) -> int:
        ' Set element to use reduced integration (1x1) '
        self.ngp = 1 if self.nnode == 4 else 2

    def get_local_disp(self, global_disp: np.ndarray) -> np.ndarray:
        return global_disp[self.dofs].reshape(self.nnode, 2)                         # [nnode, 2-dofs]

    def compute_jacobian(self, r: float, s: float):
        dHrs = QuadShapeDerivatives(r,s,self.nnode)                             # [2, nnode]
        jac = dHrs @ self.X
        detJ = jac[0,0] * jac[1,1] - jac[0,1] * jac[1,0]            
        if detJ <= 0:
            raise ValueError(f"Jacobian singular or negative: detJ = {detJ:.2e} in element {self.id}")
        invJ = np.array([[jac[1,1], -jac[0,1]],
                        [-jac[1,0], jac[0,0]]]) / detJ
        dHdX = invJ @ dHrs                                                      # [2, nnode]  
        return dHdX, detJ

    # --------------------------------------
    #  GEOMETRICALLY LINEAR METHODS
    # --------------------------------------

    def _compute_B_matrix_lin(self, r: float, s: float):
        dHdX, detJ = self.compute_jacobian(r, s)
        Bl0 = np.zeros((3, self.ndof_local))
        Bl0[0, 0::2] = dHdX[0, :]        # ∂N_i/∂x → ux dofs
        Bl0[1, 1::2] = dHdX[1, :]        # ∂N_i/∂y → uy dofs
        Bl0[2, 0::2] = dHdX[1, :]        # ∂N_i/∂y → shear
        Bl0[2, 1::2] = dHdX[0, :]        # ∂N_i/∂x → shear
            
        return Bl0, detJ  # [3, 2*nnode]

    def compute_infinitesimal_strain(self, local_disp: np.ndarray, r: float, s: float) -> np.ndarray:
        Bl0, _ = self._compute_B_matrix_lin(r, s)
        u_local_flat = local_disp.flatten()                         # [2*nnode] 
        epsilon = Bl0 @ u_local_flat
        return epsilon                                              # [3] Voigt notation [exx, eyy, 2exy]

    def compute_linear_stiff(self) -> np.ndarray:
        points, weights = GaussQuad(self.ngp)
        K_local = np.zeros((self.ndof_local, self.ndof_local))
        for r_i, w_i in zip(points, weights):
            for r_j, w_j in zip(points, weights):
                Bl0, detJ = self._compute_B_matrix_lin(r_i, r_j)
                DDSDDE = self.material.get_constitutive_matrix()
                K_local += Bl0.T @ DDSDDE @ Bl0 * w_i * w_j * detJ * self.thickness
        return K_local

    def compute_linear_intfor(self, global_disp: np.ndarray) -> np.ndarray:
        points, weights = GaussQuad(self.ngp)
        local_disp = self.get_local_disp(global_disp)
        Fint_local = np.zeros(self.ndof_local)
        for r_i, w_i in zip(points, weights):
             for r_j, w_j in zip(points, weights):
                Bl0, detJ = self._compute_B_matrix_lin(r_i, r_j)
                epsilon = self.compute_infinitesimal_strain(local_disp, r_i, r_j)
                sigma_vec = self.material.compute_stress(epsilon)
                Fint_local += Bl0.T @ sigma_vec * w_i * w_j * detJ * self.thickness
        return Fint_local
    
    # --------------------------------------
    #  GEOMETRICALLY NON-LINEAR METHODS: TOTAL LAGRANGIAN FORMULATION
    # --------------------------------------

    def compute_displacement_gradient(self, local_disp: np.ndarray, r: float, s: float) -> np.ndarray:
        dHdX, _ = self.compute_jacobian(r, s)
        return (dHdX @ local_disp).T                                  # Displacement Gradient[2, 2] 

    def compute_deformation_gradient(self, grad_u: np.ndarray) -> np.ndarray:
        return np.eye(2) + grad_u                               # Deformation Gradient [2, 2]    

    def compute_green_lagrange_strain(self, F: np.ndarray) -> np.ndarray:
        C = F.T @ F
        E = 0.5 * (C - np.eye(2))
        return np.array([E[0, 0], E[1, 1], 2 * E[0, 1]])           # Voigt notation [exx, eyy, 2exy]

    def _compute_B_matrix_nl(self, r: float, s: float, grad_u: np.ndarray):
        dHdX, _ = self.compute_jacobian(r, s)
        
        Bl1 = np.zeros((3, self.ndof_local))
        Bl1[0, 0::2] = grad_u[0,0] * dHdX[0, :]
        Bl1[0, 1::2] = grad_u[1,0] * dHdX[0, :]
        Bl1[1, 0::2] = grad_u[0,1] * dHdX[1, :]
        Bl1[1, 1::2] = grad_u[1,1] * dHdX[1, :]
        Bl1[2, 0::2] = grad_u[0,0] * dHdX[1, :] + grad_u[0,1] * dHdX[0, :]
        Bl1[2, 1::2] = grad_u[1,0] * dHdX[1, :] + grad_u[1,1] * dHdX[0, :]

        Bnl = np.zeros((4, self.ndof_local))
        Bnl[0, 0::2] = dHdX[0, :]
        Bnl[1, 0::2] = dHdX[1, :]
        Bnl[2, 1::2] = dHdX[0, :]
        Bnl[3, 1::2] = dHdX[1, :]

        return Bl1, Bnl

    def compute_nonlinear_stiff(self, global_disp: np.ndarray) -> np.ndarray:
        local_disp = self.get_local_disp(global_disp)
        points, weights = GaussQuad(self.ngp)
        K_local = np.zeros((self.ndof_local, self.ndof_local))
        for r_i, w_i in zip(points, weights):
            for r_j, w_j in zip(points, weights):
                # Displacement-strain interpolation matrices
                Bl0, detJ = self._compute_B_matrix_lin(r_i, r_j)
                grad_u = self.compute_displacement_gradient(local_disp, r_i, r_j)
                Bl1, Bnl = self._compute_B_matrix_nl(r_i, r_j, grad_u)
                F = self.compute_deformation_gradient(grad_u)
                # Material stiffness
                Bl = Bl0 + Bl1
                DDSDDE = self.material.get_constitutive_matrix()
                Kmat_local = Bl.T @ DDSDDE @ Bl * w_i * w_j * detJ * self.thickness
                # Geometric stiffness
                E_vec = self.compute_green_lagrange_strain(F)
                S_vec = self.material.compute_pk2_stress(E_vec) 
                S_mat = np.array([[S_vec[0], S_vec[2], 0, 0],
                                  [S_vec[2], S_vec[1], 0, 0],
                                  [0, 0, S_vec[0], S_vec[2]],
                                  [0, 0, S_vec[2], S_vec[1]]])
                Kgeo_local = Bnl.T @ S_mat @ Bnl * w_i * w_j * detJ * self.thickness
                # Overall stiffness
                K_local += Kmat_local + Kgeo_local
        return K_local

    def compute_nonlinear_intfor(self, global_disp: np.ndarray) -> np.ndarray:
        local_disp = self.get_local_disp(global_disp)
        points, weights = GaussQuad(self.ngp)
        Fint_local = np.zeros(self.ndof_local)
        for r_i, w_i in zip(points, weights):
            for r_j, w_j in zip(points, weights):
                # Displacement-strain interpolation matrices
                Bl0, detJ = self._compute_B_matrix_lin(r_i, r_j)
                grad_u = self.compute_displacement_gradient(local_disp, r_i, r_j)
                Bl1, _ = self._compute_B_matrix_nl(r_i, r_j, grad_u)
                Bl = Bl0 + Bl1
                F = self.compute_deformation_gradient(grad_u)
                E_vec = self.compute_green_lagrange_strain(F)
                pk2_vec = self.material.compute_pk2_stress(E_vec)
                Fint_local += Bl.T @ pk2_vec * w_i * w_j * detJ * self.thickness
        return Fint_local

    # --------------------------------------
    #  EXTERNAL LOADS METHODS
    # --------------------------------------
    def compute_body_loads(self, fvec: np.ndarray) -> np.ndarray:
        points, weights = GaussQuad(self.ngp)
        Fbody = np.zeros(self.ndof_local)
        Nshape = np.zeros((2, self.ndof_local))
        for r_i, w_i in zip(points, weights):
            for r_j, w_j in zip(points, weights):
                H = QuadShapeFunctions(r_i, r_j, self.nnode)
                _, detJ = self.compute_jacobian(r_i, r_j)

                Nshape[0, 0::2] = H 
                Nshape[1, 1::2] = H

                Fbody += Nshape.T @ fvec * w_i * w_j * detJ * self.thickness
        return Fbody

    def compute_edge_load(self, side: int, fsup: np.ndarray, reference = 'local') -> np.ndarray:
        """
        Compute equivalent nodal forces due to edge traction (surface load).
        
        Args:
            side (int): 1=top, 2=left, 3=right, 4=bottom
            fsup (array): 
                If reference='local': [ft, fn] 
                    ft: tangential traction (positive in edge direction: counterclockwise)
                    fn: normal traction (positive outward)
                If reference='global': [fx, fy]
                    fx: traction in global X direction
                    fy: traction in global Y direction
            reference (str): 'local' or 'global'

        Returns:
            Fedge_local (np.array): [ndof_local,] local force vector
        """

        if reference not in ['local', 'global']:
            raise ValueError("reference must be 'local' or 'global'")

        side = int(side)

        # Edge node mapping (local 0-based indices)
        edge_map = {
            1: ([0, 1], ('s',  1.0)),  # top
            2: ([1, 2], ('r', -1.0)),  # left
            3: ([2, 3], ('s', -1.0)),  # right
            4: ([3, 0], ('r',  1.0))   # bottom
        }
        if side not in edge_map:
            raise ValueError(f"Invalid side {side}. Use 1,2,3,4.")
        edge_nodes, fixed_param = edge_map[side]
        i, j = edge_nodes

        # Edge geometry
        x1, y1 = self.X[i]
        x2, y2 = self.X[j]
        dx = x1 - x2
        dy = y1 - y2
        L = np.sqrt(dx**2 + dy**2)
        if L < 1e-12:
            raise ValueError(f"Zero-length edge in element {self.id}, side {side}")

        # Traction in global coords
        if reference == 'local':
            tx, ty = dx / L, dy / L         # Unit tangent (from node j to i → counterclockwise)
            nx, ny = -ty, tx                # Outward unit normal: rotate 90° CCW
            ft, fn = fsup[0], fsup[1]       # Traction vector
            fx = ft * tx + fn * nx
            fy = ft * ty + fn * ny
        else:
            fx, fy = fsup[0], fsup[1]

        t_global = np.array([fx, fy])  # [2]

        # Gauss points on edge
        points, weights = GaussQuad(self.ngp-1)

        # Integration loop
        Fedge = np.zeros(self.ndof_local)
        param_name, param_val = fixed_param

        for gp, w in zip(points, weights):
            r = param_val if param_name == 'r' else gp
            s = param_val if param_name == 's' else gp

            # Shape functions
            H = QuadShapeFunctions(r, s, self.nnode).flatten() # [nnode]
            dHrs = QuadShapeDerivatives(r, s, self.nnode)      # [2, nnode]

            # 2D Jacobian
            J = dHrs @ self.X                                  # [2,2]
            detJ_edge = J[1,1] if param_name == 'r' else J[0,0]  # ds or dr direction

            if detJ_edge < 1e-12:
                raise ValueError(f"Near-singular edge Jacobian in elem {self.id}, side {side}")

            # N matrix: [2, ndof_local]
            N_mat = np.zeros((2, self.ndof_local))
            N_mat[0, 0::2] = H   # ux
            N_mat[1, 1::2] = H   # uy

            # Contribute to force
            dF = N_mat.T @ t_global * w * detJ_edge
            Fedge += dF

        return Fedge
