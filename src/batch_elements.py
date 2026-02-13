
import numpy as np
import torch
from utils import LineShapeDerivatives, LineShapeFunctions, QuadShapeFunctions, QuadShapeDerivatives, GaussQuad
from typing import List, Tuple

class MasterElement:
    ' Base class for master finite elements'

    def __init__(self, ids: List[int], nodes_list: List[List[int]], material=None, device='cpu'):
            self.ids = torch.tensor(ids, dtype=torch.int, device=device)                # Element IDs    (nelem,)
            self.nodes = torch.tensor(nodes_list, dtype=torch.int64, device=device)     # List of nodes (nelem, nnode)
            self.material = material                                                    # Material object: same material for all elements in batch
            self.device = device

            self.nelem = len(ids)                   # Number of elements in batch
            self.nnode = self.nodes.shape[1]        # Number of nodes per element (assumes all elements have same number of nodes)
            self.ndof_local = self.nnode * 2        # Number of local dofs per element
            self.dofs = self._get_dof_indices()     # Local dof indices in global system   (nelem, ndof_local)

    # --------------------------------------
    #  METHODS TO RUN BEFORE ANALYSIS (ONCE)
    # --------------------------------------

    def _get_dof_indices(self):
        '''Compute dofs indexs for each element'''
        node_offsets = 2 * (self.nodes - 1)                               # (nelem, nnode)
        dofs = torch.stack([node_offsets, node_offsets + 1], dim=-1)      # (nelem, nnode, 2)
        return dofs.view(self.nelem, self.ndof_local)                     # (nelem, ndof_local)

    def _assign_nodal_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        ''' Get nodal coordinates from global coordinate tensor '''
        node_indexs = self.nodes - 1
        self.X = coordinates[node_indexs]  # (nelem, nnode, 2)

    def _vectorized_gauss_points(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """ Returns r, s, w for all ngp² points, shape (ngp²,) """
            gp1d, w1d = GaussQuad(self.ngp)
            gp1d = gp1d.to(self.device)
            w1d = w1d.to(self.device)

            r, s = torch.meshgrid(gp1d, gp1d, indexing='ij')
            r = r.flatten()   # (ngp²,)
            s = s.flatten()
            w = torch.outer(w1d, w1d).flatten()   # (ngp²,)

            return r, s, w
  
    def reduced_integration(self) -> None:
        ' Set element to use reduced integration'
        self.ngp -= 1

    # --------------------------------------
    #  METHODS TO RUN IN EACH ITERATION
    # --------------------------------------

    def get_local_disp(self, global_disp: torch.Tensor) -> torch.Tensor:
        """ u_global: (ndof_total,) → returns (nelem, nnode, 2) """
        return global_disp[self.dofs].view(self.nelem, self.nnode, 2)  # (nelem, nnode, 2)

class LineElement(MasterElement):
    """
    Line element in 2D for load computation (e.g., distributed loads).
    Supports 2, 3, or 4 nodes.
    """
    def __init__(self, ids: List[int], nodes: List[List[int]], material=None, device='cpu'):
        super().__init__(ids, nodes, material, device)

        if self.nnode not in [2, 3, 4]:
            raise ValueError("Unsupported number of nodes. Supported: 2, 3, 4")
        self.ngp = 2 if self.nnode == 2 else 3                                  # Number of Gauss points (default full integration)

        # Define element node order for plotting edges
        if self.nnode == 2:
            self.edge_order = [0, 1]            # SEG2 element: start node, end node
        elif self.nnode == 3:
            self.edge_order = [0, 2, 1]         # SEG3 element: start, middle, end
        elif self.nnode == 4:
            self.edge_order = [0, 1, 2, 3]      # SEG4 element: start, internal1, internal2, end

    def compute_jacobian(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:   
        """
        Compute Jacobian at given points r.

        Args:
            r (torch.Tensor): Natural coordinates of shape (ngp,)

        Returns:
            J (torch.Tensor): Jacobian vectors of shape (nelem, ngp, 2)
            detJ (torch.Tensor): Jacobian determinants of shape (nelem, ngp)
        """
        dHr = LineShapeDerivatives(r, self.nnode).squeeze(-2)       # (ngp, nnode)
        J = torch.einsum('gn,enj->egj', dHr, self.X)                # (nelem, ngp, 2)
        detJ = torch.norm(J, dim=-1)                                # (nelem, ngp)
        if (detJ < 1e-10).any():
            bad_elems = torch.where((detJ < 1e-10).any(dim=1))[0]
            raise ValueError(f"Non-positive Jacobian in line elements {bad_elems.tolist()}")

        return J, detJ  # J (nelem, ngp, 2), detJ (nelem, ngp)
    
    def compute_line_load(self, flin: torch.Tensor, reference: str = 'local') -> torch.Tensor:
        """
        Compute equivalent nodal forces due to distributed line load (force per unit length).
        
        Args:
            flin (torch.Tensor): Load vector of shape (nelem, 2) or (2,) broadcasted
                If reference='local': fx=axial (tangent), fy=transversal (normal, 90° CCW)
                If reference='global': fx=global X, fy=global Y
            reference (str): 'local' or 'global' (default: 'local')

        Returns:
            Fline (torch.Tensor): Local force vector of shape (nelem, ndof_local)
        """
        if reference not in ['local', 'global']:
            raise ValueError("reference must be 'local' or 'global'")

        # Gauss points on the line
        gpoint, weight = GaussQuad(self.ngp)
        gpoint = gpoint.to(self.device)
        weight = weight.to(self.device)

        # Shape functions (pass dummy s=0 since unused in LineShapeFunctions)
        H = LineShapeFunctions(gpoint, self.nnode)  # [ngp, nnode]

        # Jacobian at Gauss points
        J, detJ = self.compute_jacobian(gpoint)  # J: [nelem, ngp, 2], detJ: [nelem, ngp]

        # Compute traction at each GP
        if reference == 'local':
            t_gp = J / detJ.unsqueeze(-1)                               # Unit tangent at each GP: [nelem, ngp, 2]
            n_gp = torch.stack([-t_gp[..., 1], t_gp[..., 0]], dim=-1)   # Unit normal (90° CCW rotation): [nelem, ngp, 2]
            ft = flin[:, 0].unsqueeze(-1)  # (nelem, 1)
            fn = flin[:, 1].unsqueeze(-1)  # (nelem, 1)
            traction_gp = ft.unsqueeze(-1) * t_gp + fn.unsqueeze(-1) * n_gp  # [nelem, ngp, 2]
        else:
            traction_gp = flin.unsqueeze(0).repeat(1, self.ngp, 1)  # Constant in global: [nelem, ngp, 2]

        # N matrix: (nelem, ngp, 2, ndof_local)
        N_edge = torch.zeros(self.nelem, self.ngp, 2, self.ndof_local, dtype=torch.float64, device=self.device)
        N_edge[:, :, 0, 0::2] = H.unsqueeze(0)  # ux components
        N_edge[:, :, 1, 1::2] = H.unsqueeze(0)  # uy components
        # Integration: ∫ N^T t ds = sum_gp N^T @ traction_gp * w * detJ
        Fline = torch.einsum('egji, egj, eg -> ei',
                             N_edge, traction_gp, weight.unsqueeze(0) * detJ)

        return Fline  # (nelem, ndof_local)

class QuadElement(MasterElement):
    """
    Quadrilateral element in 2D for continua
    Supports both Total Lagrangian formulation for geometric nonlinearity
    and geometrically linear analysis.
    """
    def __init__(self, ids: List[int], nodes_list: List[List[int]], material=None, thickness=1.0, device='cpu'):
        super().__init__(ids, nodes_list, material, device)
        self.thickness = thickness              # Assume same thickness for all elements in batch
        self.ngp = 2 if self.nnode == 4 else 3  # Number of Gauss points

        # Define element node order for plotting edges
        if self.nnode == 4:
            self.edge_order = [0, 1, 2, 3, 0]  # Q4
        elif self.nnode == 8:
            self.edge_order = [0, 4, 1, 5, 2, 6, 3, 7, 0]  # Q8
        elif self.nnode == 9:
            self.edge_order = [0, 4, 1, 5, 2, 6, 3, 7, 0]  # Q9

        # Precomputed constants
        self.r_gp = None
        self.s_gp = None
        self.dHdX = None
        self.weighted_dV = None
        self.Bl0 = None
        self.ngp2 = None
    
    # --------------------------------------
    #  AUXILIARY METHODS
    # --------------------------------------

    def _precompute_constants(self):
        """Pre-compute constant values: vectorize Gauss Points, dHdX, detJ, Bl0."""
        self.r_gp, self.s_gp, weights = self._vectorized_gauss_points()
        self.ngp2 = self.r_gp.shape[0]
        self.dHdX, detJ = self.compute_jacobian(self.r_gp, self.s_gp)               # (nelem, ngp², 2, nnode), (nelem, ngp²)
        self.Bl0 = self._compute_B_matrix_lin(self.r_gp, self.s_gp)                 # (nelem, ngp², 3, ndof_local)
        self.weighted_dV = weights.unsqueeze(0) * detJ * self.thickness             # (nelem, ngp²)

    def compute_jacobian(self, r: torch.Tensor, s: torch.Tensor):   
        dHrs = QuadShapeDerivatives(r, s, self.nnode)                           # (ngp², 2, nnode)
        J = torch.einsum('gij,ejk->egik', dHrs, self.X)                         # (nelem, ngp², 2, 2)
        detJ = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]        # (nelem, ngp²)
        if (detJ <= 0).any():
            bad_elems = torch.where((detJ <= 0).any(dim=1))[0]
            raise ValueError(f"Negative Jacobian in elements {bad_elems.tolist()}")

        # Inverse Jacobian: (nelem, ngp², 2, 2)
        invJ = torch.zeros_like(J)
        invJ[..., 0, 0] =  J[..., 1, 1]
        invJ[..., 0, 1] = -J[..., 0, 1]
        invJ[..., 1, 0] = -J[..., 1, 0]
        invJ[..., 1, 1] =  J[..., 0, 0]
        invJ = invJ / detJ.unsqueeze(-1).unsqueeze(-1)

        # Physical derivatives: dH/dX = invJ @ dHrs → (nelem, ngp², 2, nnode)
        dHdX = torch.einsum('egij,gjk->egik', invJ, dHrs)

        return dHdX, detJ

    def _vectorized_material(self):
        """Prepare the material for vectorized evaluation over nelem elements and npoints Gauss points per element."""
        if self.material.is_vectorized:
            return  # Already vectorized
        else:
            self.material.vectorize(self.nelem, self.ngp2)

    # --------------------------------------
    #  GEOMETRICALLY LINEAR METHODS
    # --------------------------------------

    def _compute_B_matrix_lin(self, r: torch.Tensor, s: torch.Tensor):
        Bl0 = torch.zeros(self.nelem, self.ngp2, 3, self.ndof_local, device=self.device)
        Bl0[:, :, 0, 0::2] = self.dHdX[:, :, 0, :]  # ∂N/∂x → ux
        Bl0[:, :, 1, 1::2] = self.dHdX[:, :, 1, :]  # ∂N/∂y → uy
        Bl0[:, :, 2, 0::2] = self.dHdX[:, :, 1, :]  # γxy from uy,x
        Bl0[:, :, 2, 1::2] = self.dHdX[:, :, 0, :]  # γxy from ux,y

        return Bl0

    def compute_infinitesimal_strain(self, local_disp: torch.Tensor) -> torch.Tensor:
        epsilon = torch.einsum('egij,ej->egi', self.Bl0, local_disp.view(self.nelem, self.ndof_local))  # (nelem, ngp², 3)
        return epsilon  # [nelem, ngp², 3] Voigt notation [exx, eyy, 2exy]

    def compute_linear_stiff(self) -> torch.Tensor:
        DDSDDE = self.material.get_constitutive_matrix()                    # (nelem, ngp², 3, 3)
        K_local = torch.einsum('egij, egik, egkl, eg -> ejl', self.Bl0, DDSDDE, self.Bl0, self.weighted_dV)
        return K_local  # (nelem, ndof_local, ndof_local)

    def compute_linear_intfor(self, global_disp: torch.Tensor) -> torch.Tensor:
        local_disp = self.get_local_disp(global_disp)                           # (nelem, nnode, 2)
        epsilon = self.compute_infinitesimal_strain(local_disp)                 # Usa precomputado
        sigma_vec = self.material.compute_stress(epsilon)                       # (nelem, ngp², 3)
        Fint_local = torch.einsum('egij,egi,eg->ej', self.Bl0, sigma_vec, self.weighted_dV)
        return Fint_local

    # --------------------------------------
    #  GEOMETRICALLY NON-LINEAR METHODS: TOTAL LAGRANGIAN FORMULATION
    # --------------------------------------

    def compute_displacement_gradient(self, local_disp: torch.Tensor) -> torch.Tensor:
        return torch.einsum('egij,ejk->egki', self.dHdX, local_disp)  # (nelem, ngp², 2, 2)

    def compute_deformation_gradient(self, grad_u: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(2, device=self.device).unsqueeze(0).unsqueeze(0)  # (1,1,2,2)
        return eye.expand(self.nelem, grad_u.shape[1], -1, -1) + grad_u  # (nelem, ngp², 2, 2)

    def compute_green_lagrange_strain(self, F: torch.Tensor) -> torch.Tensor:
        C = torch.einsum('egij,egjk->egik', F.transpose(2,3), F)                    # (nelem, ngp², 2, 2)
        E = 0.5 * (C - torch.eye(2, device=self.device).unsqueeze(0).unsqueeze(0))  # (nelem, ngp², 2, 2)
        return torch.stack([E[...,0,0], E[...,1,1], 2*E[...,0,1]], dim=-1)          # (nelem, ngp², 3)

    def _compute_B_matrix_nl(self, grad_u: torch.Tensor):
        Bl1 = torch.zeros(self.nelem, self.ngp2, 3, self.ndof_local, device=self.device)
        Bl1[:, :, 0, 0::2] = grad_u[:, :, 0, 0].unsqueeze(-1) * self.dHdX[:, :, 0, :]
        Bl1[:, :, 0, 1::2] = grad_u[:, :, 1, 0].unsqueeze(-1) * self.dHdX[:, :, 0, :]
        Bl1[:, :, 1, 0::2] = grad_u[:, :, 0, 1].unsqueeze(-1) * self.dHdX[:, :, 1, :]
        Bl1[:, :, 1, 1::2] = grad_u[:, :, 1, 1].unsqueeze(-1) * self.dHdX[:, :, 1, :]
        Bl1[:, :, 2, 0::2] = grad_u[:, :, 0, 0].unsqueeze(-1) * self.dHdX[:, :, 1, :] + grad_u[:, :, 0, 1].unsqueeze(-1) * self.dHdX[:, :, 0, :]
        Bl1[:, :, 2, 1::2] = grad_u[:, :, 1, 0].unsqueeze(-1) * self.dHdX[:, :, 1, :] + grad_u[:, :, 1, 1].unsqueeze(-1) * self.dHdX[:, :, 0, :]

        Bnl = torch.zeros(self.nelem, self.ngp2, 4, self.ndof_local, device=self.device)
        Bnl[:, :, 0, 0::2] = self.dHdX[:, :, 0, :]
        Bnl[:, :, 1, 0::2] = self.dHdX[:, :, 1, :]
        Bnl[:, :, 2, 1::2] = self.dHdX[:, :, 0, :]
        Bnl[:, :, 3, 1::2] = self.dHdX[:, :, 1, :]

        return Bl1, Bnl

    def compute_nonlinear_stiff(self, global_disp: torch.Tensor) -> torch.Tensor:
        local_disp = self.get_local_disp(global_disp)                           # (nelem, nnode, 2)
        grad_u = self.compute_displacement_gradient(local_disp)
        Bl1, Bnl = self._compute_B_matrix_nl(grad_u)
        F = self.compute_deformation_gradient(grad_u)
        # Material stiffness
        Bl = self.Bl0 + Bl1
        DDSDDE = self.material.get_constitutive_matrix()                # (nelem, ngp², 3,3)
        Kmat_local = torch.einsum('egij, egik, egkl, eg -> ejl', Bl, DDSDDE, Bl, self.weighted_dV)
        # Geometric stiffness
        E_vec = self.compute_green_lagrange_strain(F)
        S_vec = self.material.compute_pk2_stress(E_vec)
        S_mat = torch.zeros(self.nelem, self.r_gp.shape[0], 4, 4, dtype=torch.float64, device=self.device)
        S_mat[:, :, 0, 0] = S_vec[:, :, 0]
        S_mat[:, :, 1, 1] = S_vec[:, :, 1]
        S_mat[:, :, 0, 1] = S_mat[:, :, 1, 0] = S_vec[:, :, 2]
        S_mat[:, :, 2, 2] = S_vec[:, :, 0]
        S_mat[:, :, 3, 3] = S_vec[:, :, 1]
        S_mat[:, :, 2, 3] = S_mat[:, :, 3, 2] = S_vec[:, :, 2]

        Kgeo_local = torch.einsum('egij,egik,egkl,eg->ejl',
                                  Bnl, S_mat, Bnl, self.weighted_dV)
        K_local = Kmat_local + Kgeo_local
        return K_local

    def compute_nonlinear_intfor(self, global_disp: torch.Tensor) -> torch.Tensor:
        local_disp = self.get_local_disp(global_disp)
        grad_u = self.compute_displacement_gradient(local_disp)
        Bl1, _ = self._compute_B_matrix_nl(grad_u=grad_u)
        Bl = self.Bl0 + Bl1
        F = self.compute_deformation_gradient(grad_u)
        E_vec = self.compute_green_lagrange_strain(F)
        pk2_vec = self.material.compute_pk2_stress(E_vec)
        Fint_local = torch.einsum('egij,egi,eg->ej', Bl, pk2_vec, self.weighted_dV)
        return Fint_local  # (nelem, ndof_local)

    # --------------------------------------
    #  EXTERNAL LOADS METHODS
    # --------------------------------------
    def compute_body_loads(self, fvec: torch.Tensor) -> torch.Tensor:

        H = QuadShapeFunctions(self.r_gp, self.s_gp, self.nnode)    # (ngp², nnode)
        
        Nshape = torch.zeros(self.nelem, self.ngp2, 2, self.ndof_local, dtype=torch.float64, device=self.device)
        Nshape[:, :, 0, 0::2] = H.unsqueeze(0)
        Nshape[:, :, 1, 1::2] = H.unsqueeze(0)

        Fbody = torch.einsum('egji, ej, eg -> ei', Nshape, fvec, self.weighted_dV)
        return Fbody  # (nelem, ndof_local)

    def compute_edge_load(self, sides: List[int], fsup: torch.Tensor, reference='local') -> torch.Tensor:
        """
        Compute equivalent nodal forces due to edge traction (surface load).
        
        Args:
            sides (int or List[int]): Side(s) for each element, 1=top, 2=left, 3=right, 4=bottom. If int, same for all.
            fsup (torch.Tensor): (nelem, 2) or (2,) broadcasted
                If reference='local': [ft, fn] 
                    ft: tangential traction (positive in edge direction: counterclockwise)
                    fn: normal traction (positive outward)
                If reference='global': [fx, fy]
            reference (str): 'local' or 'global'

        Returns:
            Fedge_local (torch.Tensor): (nelem, ndof_local)
        """
        if reference not in ['local', 'global']:
            raise ValueError("reference must be 'local' or 'global'")

        if isinstance(sides, int):
            sides = [sides] * self.nelem
        sides = torch.tensor(sides, device=self.device)  # (nelem,)

        if fsup.ndim == 1:
            fsup = fsup.unsqueeze(0).expand(self.nelem, -1)  # (nelem, 2)

        # Edge node mapping (local 0-based indices)
        edge_map = {
            1: ([0, 1], ('s',  1.0)),  # top
            2: ([1, 2], ('r', -1.0)),  # left
            3: ([2, 3], ('s', -1.0)),  # right
            4: ([3, 0], ('r',  1.0))   # bottom
        }

        Fedge = torch.zeros(self.nelem, self.ndof_local, device=self.device)

        # Gauss points on edge
        gpoint, weight = GaussQuad(self.ngp - 1)
        gpoint = gpoint.to(self.device)
        weight = weight.to(self.device)
        n_gp_edge = gpoint.shape[0]

        for e in range(self.nelem):
            side = sides[e].item()
            if side not in edge_map:
                raise ValueError(f"Invalid side {side} for elem {e}")
            edge_nodes, fixed_param = edge_map[side]
            i, j = edge_nodes

            # Edge geometry
            p1 = self.X[e, i]  # (2,)
            p2 = self.X[e, j]  # (2,)
            edge_vec = p1 - p2
            L = torch.norm(edge_vec)

            if L < 1e-12:
                raise ValueError(f"Zero-length edge in element {self.ids[e].item()}, side {side}")

            # Traction in global coords
            if reference == 'local':
                t = edge_vec / L  # Unit tangent
                n = torch.stack([-t[1], t[0]])  # Outward unit normal
                ft, fn = fsup[e, 0], fsup[e, 1]
                traction = ft * t + fn * n
            else:
                traction = fsup[e]  # [fx, fy]

            # Integration loop
            param_name, param_val = fixed_param

            r = param_val * torch.ones_like(gpoint) if param_name == 'r' else gpoint
            s = param_val * torch.ones_like(gpoint) if param_name == 's' else gpoint

            # Shape functions
            H = QuadShapeFunctions(r, s, self.nnode)  # [n_gp_edge, nnode]
            dHrs = QuadShapeDerivatives(r, s, self.nnode)  # [n_gp_edge, 2, nnode]
            # 2D Jacobian
            J_edge = torch.einsum('gij,jk->gik', dHrs, self.X[e])  # (n_gp_edge, 2, 2)
            detJ_edge = torch.norm(J_edge[:, :, 1 if param_name == 's' else 0], dim=1)  # (n_gp_edge,)
            if torch.any(detJ_edge < 1e-12):
                raise ValueError(f"Near-singular edge Jacobian in elem {self.ids[e].item()}, side {side}")
            # N matrix: (n_gp_edge, 2, ndof_local)
            N_edge = torch.zeros(n_gp_edge, 2, self.ndof_local, dtype=torch.float64, device=self.device)
            N_edge[:, 0, 0::2] = H  # ux
            N_edge[:, 1, 1::2] = H  # uy
            # Integration: ∫ N^T t ds = sum_gp N^T @ traction * w * detJ_edge
            Fedge[e] = torch.einsum('gji, j, g -> i',
                                     N_edge, traction, weight * detJ_edge)

        return Fedge  # (nelem, ndof_local)
    