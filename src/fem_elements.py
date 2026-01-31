
import numpy as np
import torch
from fem_utils import QuadShapeFunctions, QuadShapeDerivatives, GaussQuad
from typing import Tuple

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
    def __init__(self, id: int, nodes: list, material= None, thickness=1.0, device='cpu'):
        super().__init__(id, nodes, material)
        self.id = id                                                            # Element ID   
        self.nodes = nodes                                                      # List of nodes connectivity
        self.material = material                                                # Material object: could be defined later
        self.thickness = thickness                                              # Element thickness (just in case of 2D plane stress)
        self.device = device

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



    # --------------------------------------
    #  AUXILIARY METHODS
    # --------------------------------------

    def plot_element(self, plot_local_nodes_label=True,
                 plot_local_edges_label=True,
                 plot_global_nodes_label=False):
        """
        Shows element edges and optionally labels for local/global nodes and edge numbers.
        """
        import matplotlib.pyplot as plt

        # Create figure and axis properly
        fig, ax = plt.subplots(figsize=(8, 8))

        # Extract coordinates using the predefined edge_order (closes the loop automatically)
        Xo = self.X[self.edge_order, 0]   # self.X is (nnode, 2) array with reference coordinates
        Yo = self.X[self.edge_order, 1]

        # Plot element outline (original configuration)
        ax.plot(Xo, Yo, 'k-', linewidth=2, label=f'Element {self.id}')

        # Plot nodes (only the actual nodes, not the closing point)
        ax.plot(self.X[:, 0], self.X[:, 1], 'ko', markersize=8)

        # Local node labels (0, 1, 2, ...)
        if plot_local_nodes_label:
            for i, node in enumerate(self.nodes):
                ax.text(self.X[i, 0], self.X[i, 1], str(i+1),
                        color='red', fontsize=11, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="red", alpha=0.8))

        # Global node labels (node.id)
        if plot_global_nodes_label:
            for i, node in enumerate(self.nodes):
                ax.text(self.X[i, 0] - 0.03 * (Xo.max() - Xo.min() or 1), 
                        self.X[i, 1] + 0.03 * (Yo.max() - Yo.min() or 1),
                        str(node),
                        color='blue', fontsize=10,
                        ha='center', va='bottom')

        # Edge labels (centered on each edge)
        edge_order_4 = [0, 1, 2, 3, 0]
        if plot_local_edges_label:
            for edge in range(4):
                i1 = edge_order_4[edge]
                i2 = edge_order_4[edge + 1]
                xm = (self.X[i1, 0] + self.X[i2, 0]) / 2
                ym = (self.X[i1, 1] + self.X[i2, 1]) / 2
                 # Adjust the position to shift the text to the right and above
                text_offset_x = 0.1  # Offset to the right
                text_offset_y = 0.1  # Offset above
                ax.text(xm + text_offset_x, ym + text_offset_y, str(edge + 1),
                color='darkgreen', fontsize=9, fontweight='bold',
                ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="darkgreen", alpha=0.8))

        # Final plot adjustments
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Quadrilateral Element:{self.id} (Q{self.nnode})', pad=20)

        plt.show()

    # def rotate_order(self, sense='clockwise', delta_pos=1):
    #     pass

    def _get_dof_indices(self):
        dofs = []
        for node in self.nodes:
            dofs.extend([2 * (node - 1), 2 * (node - 1) + 1])
        return torch.tensor(dofs, dtype=torch.long, device=self.device)
    
    def get_nodal_coordinates(self, coordinates: np.ndarray) -> torch.Tensor:
        ''' Get nodal coordinates from global coordinate array '''

        node_indices = np.array(self.nodes) - 1         # [nnode,]
        nodal_coords_np = coordinates[node_indices]     # Shape: (nnode, 2)
        self.X = torch.tensor(nodal_coords_np, dtype=torch.float64, device=self.device) 
        return self.X

    def reduced_integration(self) -> int:
        ' Set element to use reduced integration (1x1) '
        self.ngp = 1 if self.nnode == 4 else 2

    def get_local_disp(self, global_disp: torch.Tensor) -> torch.Tensor:
        """ u_global: (ndof_total,) → returns (nnode, 2) """
        return global_disp[self.dofs].reshape(self.nnode, 2)    # [nnode, 2-dofs]

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

    def compute_jacobian(self, r: torch.tensor, s: torch.tensor):   
        dHrs = QuadShapeDerivatives(r, s, self.nnode)   # [ngp², 2, nnode]
        J = torch.einsum('gij,jk->gik', dHrs, self.X)   # (ngp², 2, 2)
        detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        if (detJ <= 0).any():
            bad = torch.where(detJ <= 0)[0]
            raise ValueError(f"Negative Jacobian in element {self.id} at GPs {bad.tolist()}")

        # Inverse Jacobian: (ngp², 2, 2)
        invJ = torch.zeros_like(J)
        invJ[:, 0, 0] =  J[:, 1, 1]
        invJ[:, 0, 1] = -J[:, 0, 1]
        invJ[:, 1, 0] = -J[:, 1, 0]
        invJ[:, 1, 1] =  J[:, 0, 0]
        invJ = invJ / detJ.unsqueeze(-1).unsqueeze(-1)

        # Physical derivatives: dH/dX = invJ @ dHrs → (ngp², 2, nnode)

        dHdX = torch.einsum('gij,gjk->gik', invJ, dHrs)
                                                     # [2, nnode]  

        return dHdX, detJ

    # --------------------------------------
    #  GEOMETRICALLY LINEAR METHODS
    # --------------------------------------

    def _compute_B_matrix_lin(self, r: torch.tensor, s: torch.tensor):
        dHdX, detJ = self.compute_jacobian(r, s)
        ngp2 = r.shape[0]

        Bl0 = torch.zeros(ngp2, 3, self.ndof_local, device=self.device)
        Bl0[:, 0, 0::2] = dHdX[:, 0, :]   # ∂N/∂x → ux
        Bl0[:, 1, 1::2] = dHdX[:, 1, :]   # ∂N/∂y → uy
        Bl0[:, 2, 0::2] = dHdX[:, 1, :]   # γxy from uy,x
        Bl0[:, 2, 1::2] = dHdX[:, 0, :]   # γxy from ux,y

        return Bl0, detJ  # Linear B-matrix: (ngp², 3, ndof)

    def compute_infinitesimal_strain(self, local_disp: torch.tensor, r: torch.tensor, s: torch.tensor) -> torch.tensor:
        Bl0, _ = self._compute_B_matrix_lin(r, s)
        epsilon = torch.einsum('gij,j->gi', Bl0, local_disp.flatten())
        return epsilon                                              # [3] Voigt notation [exx, eyy, 2exy]

    def compute_linear_stiff(self) -> torch.tensor:
        r_gp, s_gp, weights = self._vectorized_gauss_points()
        Bl0, detJ = self._compute_B_matrix_lin(r_gp, s_gp)

        DDSDDE = self.material.get_constitutive_matrix()        # incluir deformacion para calcular tensor algoritmico

        K_local = torch.einsum('gij, ik, gkl, g -> jl', Bl0, DDSDDE, Bl0, weights * detJ * self.thickness)
        return K_local  # (ndof, ndof)

    def compute_linear_intfor(self, global_disp: torch.tensor) -> torch.tensor:
        r_gp, s_gp, weights = self._vectorized_gauss_points()
        Bl0, detJ = self._compute_B_matrix_lin(r_gp, s_gp)
        local_disp = self.get_local_disp(global_disp)
        epsilon = self.compute_infinitesimal_strain(local_disp, r_gp, s_gp)
        sigma_vec = self.material.compute_stress(epsilon)
        Fint_local = torch.einsum('gij,gi,g->j', Bl0, sigma_vec, weights * detJ * self.thickness)

        return Fint_local
    
    # --------------------------------------
    #  GEOMETRICALLY NON-LINEAR METHODS: TOTAL LAGRANGIAN FORMULATION
    # --------------------------------------

    def compute_displacement_gradient(self, local_disp: torch.tensor, r: torch.tensor, s: torch.tensor) -> torch.tensor:
        dHdX, _ = self.compute_jacobian(r, s)
        return torch.einsum('gij,jk->gki', dHdX, local_disp)   # (ngp², 2, 2)                                 # Displacement Gradient[2, 2] 

    def compute_deformation_gradient(self, grad_u: torch.tensor) -> torch.tensor:
        return torch.eye(2, device=self.device).unsqueeze(0) + grad_u  # (ngp², 2, 2)                             # Deformation Gradient [2, 2]    

    def compute_green_lagrange_strain(self, F: torch.tensor) -> torch.tensor:
        C = torch.einsum('gij,gjk->gik', F.transpose(1,2), F)
        E = 0.5 * (C - torch.eye(2, device=self.device))
        return torch.stack([E[:,0,0], E[:,1,1], 2*E[:,0,1]], dim=1)  # (ngp², 3)           # Voigt notation [exx, eyy, 2exy]

    def _compute_B_matrix_nl(self, r: torch.tensor, s: torch.tensor, grad_u: torch.tensor):
        dHdX, _ = self.compute_jacobian(r, s)
        ngp2 = r.shape[0]

        Bl1 = torch.zeros(ngp2, 3, self.ndof_local, device=self.device)
        Bl1[:, 0, 0::2] = grad_u[:, 0, 0].unsqueeze(1) * dHdX[:, 0, :]
        Bl1[:, 0, 1::2] = grad_u[:, 1, 0].unsqueeze(1) * dHdX[:, 0, :]
        Bl1[:, 1, 0::2] = grad_u[:, 0, 1].unsqueeze(1) * dHdX[:, 1, :]
        Bl1[:, 1, 1::2] = grad_u[:, 1, 1].unsqueeze(1) * dHdX[:, 1, :]
        Bl1[:, 2, 0::2] = grad_u[:, 0, 0].unsqueeze(1) * dHdX[:, 1, :] + grad_u[:, 0, 1].unsqueeze(1) * dHdX[:, 0, :]
        Bl1[:, 2, 1::2] = grad_u[:, 1, 0].unsqueeze(1) * dHdX[:, 1, :] + grad_u[:, 1, 1].unsqueeze(1) * dHdX[:, 0, :]

        Bnl = torch.zeros(ngp2, 4, self.ndof_local, device=self.device)
        Bnl[:, 0, 0::2] = dHdX[:, 0, :]
        Bnl[:, 1, 0::2] = dHdX[:, 1, :]
        Bnl[:, 2, 1::2] = dHdX[:, 0, :]
        Bnl[:, 3, 1::2] = dHdX[:, 1, :]

        return Bl1, Bnl

    def compute_nonlinear_stiff(self, global_disp: torch.Tensor) -> torch.Tensor:
        r_gp, s_gp, weights = self._vectorized_gauss_points()
        ngp2 = r_gp.shape[0]
        Bl0, detJ = self._compute_B_matrix_lin(r_gp, s_gp)
        local_disp = self.get_local_disp(global_disp)
        grad_u = self.compute_displacement_gradient(local_disp, r_gp, s_gp)
        Bl1, Bnl = self._compute_B_matrix_nl(r_gp, s_gp, grad_u)
        F = self.compute_deformation_gradient(grad_u)
        # Material stiffness
        Bl = Bl0 + Bl1

        DDSDDE = self.material.get_constitutive_matrix()

        Kmat_local = torch.einsum('gij, ik, gkl, g -> jl', Bl, DDSDDE, Bl, weights * detJ * self.thickness)
        
        # Geometric stiffness
        E_vec = self.compute_green_lagrange_strain(F)
        S_vec = self.material.compute_pk2_stress(E_vec) 
        S_mat = torch.zeros(ngp2, 4, 4, dtype=torch.float64, device=self.device)
        S_mat[:, 0, 0] = S_vec[:, 0]
        S_mat[:, 1, 1] = S_vec[:, 1]
        S_mat[:, 0, 1] = S_mat[:, 1, 0] = S_vec[:, 2]
        S_mat[:, 2, 2] = S_vec[:, 0]
        S_mat[:, 3, 3] = S_vec[:, 1]
        S_mat[:, 2, 3] = S_mat[:, 3, 2] = S_vec[:, 2]

        Kgeo_local = torch.einsum('gij,gik,gkl,g->jl',
                            Bnl, S_mat, Bnl, weights * detJ * self.thickness)
        # Overall stiffness
        K_local = Kmat_local + Kgeo_local
        
        return K_local  # (ndof_local, ndof_local)

    def compute_nonlinear_intfor(self, global_disp: torch.tensor) -> torch.tensor:
        r_gp, s_gp, weights = self._vectorized_gauss_points()
        Bl0, detJ = self._compute_B_matrix_lin(r_gp, s_gp)
        local_disp = self.get_local_disp(global_disp)
        grad_u = self.compute_displacement_gradient(local_disp, r_gp, s_gp)
        Bl1, _ = self._compute_B_matrix_nl(r_gp, s_gp, grad_u)
        Bl = Bl0 + Bl1
        F = self.compute_deformation_gradient(grad_u)
        E_vec = self.compute_green_lagrange_strain(F)
        pk2_vec = self.material.compute_pk2_stress(E_vec)
        Fint_local = torch.einsum('gij,gi,g->j', Bl, pk2_vec, weights * detJ * self.thickness)
        
        return Fint_local

    # --------------------------------------
    #  EXTERNAL LOADS METHODS
    # --------------------------------------
    def compute_body_loads(self, fvec: torch.tensor) -> torch.tensor:
        r_gp, s_gp, weights = self._vectorized_gauss_points()
        ngp2 = r_gp.shape[0]
        H = QuadShapeFunctions(r_gp, s_gp, self.nnode)
        _, detJ = self.compute_jacobian(r_gp, s_gp)
        
        Nshape = torch.zeros(ngp2, 2, self.ndof_local, dtype=torch.float64, device=self.device)
        Nshape[0, 0::2] = H 
        Nshape[1, 1::2] = H

        Fbody = torch.einsum('gji, j, g -> i', Nshape, fvec, weights * detJ * self.thickness)
        return Fbody

    def compute_edge_load(self, side: int, fsup: torch.tensor, reference = 'local') -> torch.tensor:
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
        p1 = self.X[i]      # (2,)
        p2 = self.X[j]      # (2,)
        edge_vec = p1 - p2
        L = torch.norm(edge_vec)          # sqrt(dx² + dy²)

        if L < 1e-12:
            raise ValueError(f"Zero-length edge in element {self.id}, side {side}")

        # Traction in global coords
        if reference == 'local':
            t = edge_vec / L                                # Unit tangent (from node j to i → counterclockwise)
            n = torch.stack([-t[1], t[0]])                  # Outward unit normal: rotate 90° CCW
            ft, fn = fsup[0], fsup[1]
            traction = ft * t + fn * n
        else:
            traction = fsup  # already [fx, fy]

        # Gauss points on edge
        gpoint, weight = GaussQuad(self.ngp - 1)
        gpoint = gpoint.to(self.device)
        weight = weight.to(self.device)
        n_gp_edge = gpoint.shape[0]

        # Integration loop
        param_name, param_val = fixed_param

        r = param_val * torch.ones_like(gpoint) if param_name == 'r' else gpoint
        s = param_val * torch.ones_like(gpoint) if param_name == 's' else gpoint

        # Shape functions
        H = QuadShapeFunctions(r, s, self.nnode)   # [ngp², 2, nnode]
        dHrs = QuadShapeDerivatives(r, s, self.nnode)   # [ngp², 2, nnode]
        # 2D Jacobian
        J_edge = torch.einsum('gij,jk->gik', dHrs, self.X)   # (ngp, 2, 2)
        detJ_edge = torch.norm(J_edge[:, :, 1 if param_name == 's' else 0], dim=1)  # ds or dr direction
        if torch.any(detJ_edge < 1e-12):
            raise ValueError(f"Near-singular edge Jacobian in elem {self.id}, side {side}")
        # N matrix: (n_gp_edge, 2, ndof_local)
        N_edge = torch.zeros(n_gp_edge, 2, self.ndof_local, dtype=torch.float64, device=self.device)
        N_edge[:, 0, 0::2] = H  # ux
        N_edge[:, 1, 1::2] = H  # uy
        # Integration: ∫ N^T t ds = sum_gp N^T @ traction * w * detJ_edge
        Fedge = torch.einsum('gji, j, g -> i',
                            N_edge, traction, weight * detJ_edge)

        return Fedge