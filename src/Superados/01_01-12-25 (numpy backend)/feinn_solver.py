import numpy as np
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Boundary Conditions classes
# ----------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class BoundaryCondition:
    dof: int
    value: float

    def __post_init__(self) -> None:
            """Validate dof after construction."""
            if self.dof not in (1, 2):
                raise ValueError(f"Invalid dof {self.dof}. Must be 1 (ux) or 2 (uy).")

class BoundaryConditions(dict):
    """
    Dictionary-like container that only accepts BoundaryCondition objects.

    Example
    -------
    bcs = BoundaryConditions()
    bcs["fixed_bottom"] = BoundaryCondition(dof=1, value=0.0)  # ux = 0
    bcs["fixed_left"]   = BoundaryCondition(dof=2, value=0.0)  # uy = 0
    """
    def __setitem__(self, key: str, value: BoundaryCondition) -> None:
        if not isinstance(value, BoundaryCondition):
            raise TypeError("Value must be an instance of BoundaryCondition")
        super().__setitem__(key, value)

# ----------------------------------------------------------------------
# Body Loads classes
# ----------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class BodyLoad:
    """
    Body load (force per unit volume) applied to a group of elements.

    Attributes
    ----------
    bx : float
        Body force in X-direction ([F]/[L³])
    by : float
        Body force in Y-direction ([F]/[L³])
    """
    bx: float
    by: float

    @property
    def vector(self) -> np.ndarray:
        """Return the load as a NumPy array [bx, by]."""
        return np.array([self.bx, self.by], dtype=float)
    
class BodyLoads(dict):
    """Only accepts BodyLoad objects."""
    def __setitem__(self, key: str, value: BodyLoad) -> None:
        if not isinstance(value, BodyLoad):
            raise TypeError("Value must be an instance of BodyLoad")
        super().__setitem__(key, value)


# ----------------------------------------------------------------------
# Edge Loads classes
# ----------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class EdgeLoad:
    """
    Surface traction applied to a specific side of an element.

    Attributes
    ----------
    side : int
        Element side number (1-based, element-dependent)
    fnormal : float
        Normal component of the traction
    ftangential : float
        Tangential component of the traction
    reference : str
        'local'  - components are given in the element local system
        'global' - components are given in the global X-Y system
    """

    side: int
    fnormal: float
    ftangential: float
    reference: str = "local"

    def __post_init__(self) -> None:
        if self.reference not in {"local", "global"}:
            raise ValueError(
                f"Invalid reference '{self.reference}'. Must be 'local' or 'global'."
            )

    @property
    def vector(self) -> np.ndarray:
        """Return the load as a NumPy array [ftangential, fnormal]."""
        return np.array([self.ftangential, self.fnormal], dtype=float)

class EdgeLoads(dict):
    """Only accepts EdgeLoad objects."""
    def __setitem__(self, key: str, value: EdgeLoad) -> None:
        if not isinstance(value, EdgeLoad):
            raise TypeError("Value must be an instance of EdgeLoad")
        super().__setitem__(key, value)

# ----------------------------------------------------------------------
# Nodal Loads classes
# ----------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class NodalLoad:
    """
    Concentrated force applied to a node.

    Attributes
    ----------
    fx : float
        Force in X-direction
    fy : float
        Force in Y-direction
    """
    fx: float
    fy: float

    def vector(self) -> np.ndarray:
        """Return the load as a NumPy array [fx, fy]."""
        return np.array([self.fx, self.fy], dtype=float)

class NodalLoads(dict):
    """Only accepts NodalLoad objects."""
    def __setitem__(self, key: str, value: NodalLoad) -> None:
        if not isinstance(value, NodalLoad):
            raise TypeError("Value must be an instance of NodalLoad")
        super().__setitem__(key, value)

# ----------------------------------------------------------------------
# Solver classes
# ----------------------------------------------------------------------

class BaseSolver:
    "Base class for different types of solvers for 2D problems"

    def __init__(self,  mesh: object, 
                        bcs: BoundaryConditions, 
                        matfld: dict, 
                        body_loads: BodyLoads = None, 
                        edge_loads: EdgeLoads  = None, 
                        nodal_loads: NodalLoads  = None,
                        verbose: bool = False):
        """
        Parameters (mandatory)
        ----------
        mesh : UniformQuadMesh2D
            Finite element mesh with nodes and elements.
        bcs : dict
            {'group_name': BoundaryCondition(), ...}
            Dirichlet boundary conditions by node groups.
        matfld: dict
            {'group_name': Material(), ...}
            Material fields by group of elements.
        
        Parameters (if requered)
        ----------
        body_loads: dict
            {'group_name': BodyLoad(), ...}
            Body loads by group of elements.
        edge_loads: dict
            {'group_name': EdgeLoad(), ...}
            Edge loads by group of elements.
        nodal_loads: dict
            {'group_name': NodalLoad(), ...}
            Nodal loads by group of nodes.   
        """
        self.mesh = mesh
        self.bcs = bcs                          # Diritchlet Boundary conditions. [node_id, dof (1=u,2=v), value]  
        self.matfld = matfld                    # Where elements material properties are assigned
        self.body_loads = body_loads            # Body forces (if any)
        self.edge_loads = edge_loads            # Surface tractions (if any)
        self.nodal_loads = nodal_loads          # Concentrated forces (if any)
        self.formulation = 'infinitesimal'      # "infinitesimal" / "TLF"
        self.coordinates = mesh.coordinates
        self.elements = mesh.elements

        self.verbose = verbose

        # === MESH ATTRIBUTES ===
        self.nnod = mesh.nnod               # Number of nodes
        self.nelem = mesh.nelem             # Number of elements
        self.ndof = self.nnod * 2           # Total number of DOFs  

        # === APPLY MATERIALS ===
        self._assign_matfld(matfld)
        self._check_material()              # Checks if all elements have an assigned material

        # === INICIALIZE SOLUTION AND SOLVER DATA ===
        self.udisp = np.zeros(self.ndof)   # Displacement vector initialization

        # === APPLY DIRICHLET BOUNDARY CONDITIONS ===
        self.free_dofs = self._apply_dirichlet_bcs()
        self.fixed_dofs = set(range(self.ndof)) - set(self.free_dofs)

        # === ASSEMBLE EXTERNAL FORCE VECTOR ===
        self.Fext_total = np.zeros(self.ndof)
        self._apply_body_loads()
        self._apply_edge_loads()
        self._apply_nodal_loads()

        # Current state
        self.load_factor = 0.0                              # Current λ (0 ≤ λ ≤ 1 or more)
        self.Fext = np.zeros(self.ndof)                 # Current external load = λ * Fext_total
        self.history = {
            'load_factor': [0.0],
            'displacement_error': [0.0],
            'residual_error': [0.0],
            'energetic_error': [0.0],
            'iterations': [0]
        }
    # ========================================
    # BASE METHODS
    # ========================================

    def _reinit(self):
        self.udisp.fill(0.0)
        self.load_factor = 0.0
        self.Fext.fill(0.0)
        self.history = {
            'load_factor': [0.0],
            'displacement_error': [0.0],
            'residual_error': [0.0],
            'energetic_error': [0.0],
            'iterations': [0]
        }

    def _assign_matfld(self, matfld):
        """
        Assigns material to elements based on element groups.
        Group 'all' is used as default (fallback).
        """
        if 'all' not in matfld:
            raise ValueError("Group 'all' is MANDATORY in matfld")

        default_material = matfld['all']
        for elem in self.mesh.elements:
            elem.material = default_material  # Default: 'all'

        for group_name, material in matfld.items():
            if group_name == 'all':
                continue
            if group_name not in self.mesh.element_groups:
                print(f"[matfld] Warning: group '{group_name}' does not exist in mesh")
                continue
            for elem_id in self.mesh.element_groups[group_name]:
                self.mesh.elements[elem_id - 1].material = material

        if self.verbose:
            print(f"[matfld] Assigned: {list(matfld.keys())}")
        
    def _check_material(self):
        """
        Checks that every element has a material assigned.
        Raises error if any element has material = None.
        """
        missing = [elem.id for elem in self.mesh.elements if elem.material is None]
        if missing:
            raise ValueError(f"Elements without material: {missing}")
        if self.verbose:
            print(f"[matfld] All {self.nelem} elements have assigned material")

    def _apply_dirichlet_bcs(self):
        """
        Apply Dirichlet BCs from self.bcs.
        Expected format:
            bcs = {
                'fixed_bottom': BoundaryCondition(dof=0, value=0.0),    # ux = 0
                'bottom': [ BoundaryCondition(dof=1, value=0.0),        # ux = 0
                            BoundaryCondition(dof=2, value=0.0)         # uy = 0
                         ],
                'corner': BoundaryCondition(dof=1, value=0.1)           # ux = 0.1
        }
        """
        fix = []
        for group_name, bc_input in self.bcs.items():
            if group_name not in self.mesh.node_groups:
                print(f"[BC] Warning: group '{group_name}' not in mesh")
                continue
            
            bc_list = bc_input if isinstance(bc_input, list) else [bc_input]   # Convert to list if single condition

            for bc in bc_list:
                doffix = bc.dof		                    # Prescribed DOF (1: X-DISPLACEMENT, 2: Y-DISPLACEMENT)
                valuefix = bc.value		                # Prescribed value
                for nid in self.mesh.node_groups[group_name]:
                    ieqn = 2 * (nid-1) + doffix - 1	        # Prescribed DOF position
                    self.udisp[ieqn] = valuefix		        # Impose initial boundary condition
                    fix.append(ieqn)					    # Record prescribed DOF
        
        if self.verbose:
            print(f"[BC] Applied {len(self.bcs)} Dirichlet conditions")

        fix=np.array(fix)                               # Convert "fix" to numpy array
        return np.setdiff1d(range(self.ndof),fix)		# Free DOFs

    def _apply_body_loads(self):
        """Apply volume (body) loads."""
        if not self.body_loads:
            return

        for group_name, load in self.body_loads.items():
            if group_name not in self.mesh.element_groups:
                print(f"[body_load] Warning: group '{group_name}' not in mesh")
                continue

            for eid in self.mesh.element_groups[group_name]:
                elem = self.mesh.elements[eid - 1]
                f_body = elem.compute_body_loads(load.vector)
                dofs = elem.dofs
                self.Fext_total[dofs] += f_body

        if self.verbose:
            print(f"[body_load] Applied {len(self.body_loads)} body load groups")

    def _apply_edge_loads(self):
        """Apply surface (edge) tractions."""
        if not self.edge_loads:
            return

        for group_name, load in self.edge_loads.items():
            if group_name not in self.mesh.element_groups:
                print(f"[edge_load] Warning: group '{group_name}' not in mesh")
                continue

            for eid in self.mesh.element_groups[group_name]:
                elem = self.mesh.elements[eid - 1]
                f_edge = elem.compute_edge_load(
                    load.side, load.vector, load.reference
                )
                dofs = elem.dofs
                self.Fext_total[dofs] += f_edge

        if self.verbose:
            print(f"[edge_load] Applied {len(self.edge_loads)} edge load groups")

    def _apply_nodal_loads(self):
        """Apply concentrated nodal loads.
            Expected format:
            nodal_loads = {
                'top_nodes': [nodal_x, nodal_y],
                'left_nodes': [nodal_x, nodal_y]
            }
        """
        if not self.nodal_loads:
            return
        
        for group_name, load_value in self.nodal_loads.items():
            if group_name not in self.mesh.node_groups:
                print(f"[nodal_load] Warning: group '{group_name}' not in mesh")
                continue
            for nid in self.mesh.node_groups[group_name]:
                ieqn = 2 * (nid-1)	        # Prescribed position
                fx, fy = load_value.fx, load_value.fy
                self.Fext_total[ieqn] += fx
                self.Fext_total[ieqn + 1] += fy

        if self.verbose:
            print(f"[nodal_load] Applied {len(self.nodal_loads)} nodal load groups")      

    def _assemble_internal_forces(self):
        """
        Assemble global internal force vector
        """

        u = self.udisp
        Fint = np.zeros(self.ndof)

        if self.formulation == 'infinitesimal':
            compute_f = lambda elem: elem.compute_linear_intfor(u)
        elif self.formulation == 'TLF':
            compute_f = lambda elem: elem.compute_nonlinear_intfor(u)
        else:
            raise ValueError("Invalid formulation. Must be 'infinitesimal' or 'TLF'.")
        
        for elem in self.mesh.elements:
            np.add.at(Fint, elem.dofs, compute_f(elem))

        return Fint

    def set_load_factor(self, lam: float):
        """
        Set the current load factor λ and update external force vector.
        """
        self.load_factor = lam
        self.Fext = lam * self.Fext_total
        if self.verbose:
            print(f"[Load] Set load factor = {lam:.4f}.")

    def update_load_factor(self, dlam: float):
        """
        Update the current load factor λ and external force vector.
        """
        lam = self.load_factor + dlam
        self.set_load_factor(lam)

    def run_complete(self, nsteps: int = 10):
        self.run_steps(nsteps=nsteps, reset=True)

    def run_steps(self, initial_load_factor: float = 0, final_load_factor: float = 1, nsteps: int = 10, reset: bool = False):
        """
        Run multiple load steps from initial_load_factor to final_load_factor.
        """
        if reset:
            self._reinit()
        
        load_factors = np.linspace(initial_load_factor, final_load_factor, nsteps + 1)[1:]  # exclude initial
        for lam in load_factors:
            self.run_step(lam)
        
    def run_step(self, target_lambda: float, max_substeps=10, reset: bool = False):
        """
        Run a load step until reaching target_lambda.
        Uses adaptive load stepping
        """
        if reset:
            self._reinit()
        
        if target_lambda <= self.load_factor:
            print(f"Warning: target load factor = {target_lambda} lower or equal to current load factor = {self.load_factor}")
            return

        dlam = target_lambda - self.load_factor
        step_size = dlam
        attempts = 0

        while self.load_factor < target_lambda:
            udisp_old = self.udisp.copy()
            attempts += 1
            self.update_load_factor(step_size)
            
            converged = self._incremental_step()
            if converged:
                step_size = min(step_size * 1.5, target_lambda - self.load_factor)  # increase step
                attempts = 0
            else:
                step_size *= 0.5                # decrease step
                self.udisp = udisp_old.copy     # restore previous displacements
                attempts += 1

                if attempts > max_substeps or step_size < 1e-8:
                    raise RuntimeError(f"Failed to converge to load factor = {target_lambda:.4f}")
    
    def _incremental_step(self):
        """
        """
        pass

    def save_state(self, filename: str = None):
            """
            Saves the current state of the solver.
            """
            state = {
                'udisp': self.udisp.copy(),              # Current displacement
                'load_factor': self.load_factor,         # Current load factor
                'history': self.history.copy(),          # Convergence history
            }
            if filename is None:
                return state
            else:
                import pickle

                with open(filename, 'wb') as f:
                    pickle.dump(state, f)
                if self.verbose:
                    print(f"[State] Save state in '{filename}'")

    def load_state(self, filename: str):
        """
        Loads the solver state from a file.
        """
        import pickle
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        self.udisp[:] = state['udisp']
        self.load_factor = state['load_factor']
        self.history = state['history']
        
        if self.verbose:
            print(f"[State] Loaded state from '{filename}'")

    def plot_deformed_mesh(self, scale: float = 5.0,
                            show_original: bool = True,
                            show_nodes: bool = True,
                            original_alpha: float = 0.3,
                            title: str = None,
                            figsize: tuple = (10, 8)):
            """
            Plot deformed mesh (and, optionally, the original one).

            Parameters:
                scale           displacement scale factor
                show_original   if True, shows the original mesh
                original_alpha  transparency of the original mesh
                title             
                figsize
            """
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            fig, ax = plt.subplots(figsize=figsize)

            coords = self.mesh.coordinates.copy()                   # (nnodes, 2)
            u_nodal = self.udisp.reshape(-1, 2)                     # (nnodes, 2)
            deformed = coords + scale * u_nodal

            deformed_color = '#1f77b4'
            original_color = 'gray'

            label_def = "Deformed"
            label_orig = "Original"

            if show_nodes:
                # Plot all nodes
                ax.scatter(deformed[:, 0], deformed[:, 1], c=deformed_color, s= 10)

            for elem in self.mesh.elements:

                # Determine element nodes (0-based)
                elem_nodes = elem.nodes - 1
                
                # Determine the order of nodes
                edge_order = elem.edge_order
                
                # Extract original and deformed coordinates
                Xo = coords[elem_nodes[edge_order], 0]
                Yo = coords[elem_nodes[edge_order], 1]
                Xd = deformed[elem_nodes[edge_order], 0]
                Yd = deformed[elem_nodes[edge_order], 1]

                # Plot deformed mesh
                ax.plot(Xd, Yd, color=deformed_color, lw=1.6,
                        label=label_def if elem is self.mesh.elements[0] else "")
                ax.fill(Xd, Yd, color=deformed_color, alpha=0.07,
                        label="" if elem is self.mesh.elements[0] else "")

                # Optionally plot original mesh
                if show_original:
                    ax.plot(Xo, Yo, color=original_color, lw=0.8, ls='--', 
                            alpha=original_alpha, label=label_orig if elem is self.mesh.elements[0] else "")
            
            if show_original:
                if show_nodes:
                    # Plot all nodes
                    ax.scatter(coords[:, 0], coords[:, 1], c=original_color, s= 5)

            # Configuración estética
            ax.set_aspect('equal')
            ax.axis('off')

            if title is None:
                title = f"Deformed Mesh - displacement scale x{scale}"

            ax.set_title(title, fontsize=16, pad=20)

            # Leyenda limpia
            handles = [Line2D([0], [0], color=deformed_color, lw=2, label=f'Deformed x{scale}')]
            if show_original:
                handles.append(Line2D([0], [0], color=original_color, lw=1, ls='--', alpha=original_alpha, label='Original'))

            ax.legend(handles=handles,
                        loc='upper left',
                        bbox_to_anchor=(1.02, 1),
                        borderaxespad=0,
                        frameon=True,
                        fancybox=False,
                        edgecolor='black',
                        fontsize=11)

            plt.subplots_adjust(left=0.01, right=0.80, top=0.94, bottom=0.06)
            plt.show()

            return

class NFEA(BaseSolver):
    """
    This class implements a nonlinear finite element solver
    """
    def __init__(self,  mesh: object, 
                        bcs: object, 
                        matfld: dict, 
                        body_loads: dict = None, 
                        edge_loads: dict  = None, 
                        nodal_loads: dict  = None,
                        verbose: bool = False):
        
        super().__init__(mesh, bcs, matfld, body_loads, edge_loads, nodal_loads, verbose)
    
        self.dtol = 1e-6        # Displacement tolerance
        self.etol = 1e-6        # Energy tolerance 
        self.ftol = 1e-6        # Force tolerance

        self.maxit = 20         # Maximum number of iterations
    
    def _assemble_stiffness(self):
        """
        Assemble global stiffness matrix
        """
        K = np.zeros((self.ndof, self.ndof))

        if self.formulation == 'infinitesimal':
            for elem in self.mesh.elements:
                np.add.at(K, np.ix_(elem.dofs, elem.dofs), elem.compute_linear_stiff())
        else:
            u = self.udisp
            for elem in self.mesh.elements:
                np.add.at(K, np.ix_(elem.dofs, elem.dofs), elem.compute_nonlinear_stiff(u))

        return K

    def _incremental_step(self):
        """
        """

        # ----------------------------------------------------
        # Initialization of some variables
        # ----------------------------------------------------

        err_d = [100.0]          # Inicialization of "error" variable. Displacement criterion
        err_f = [100.0]          # Inicialization of "error" variable. Force criterion
        err_e = [100.0]          # Inicialization of "error" variable. Energetic criterion

        it = 0                  # Iteration counter
        
        du = np.zeros(self.ndof)                        # Displacement increment vector initialization

        Fint = self._assemble_internal_forces()         # Internal force vector initalization
        residual = self.Fext - Fint

        # Norma inicial del residuo (solo dofs libres)
        res_norm0 = np.linalg.norm(residual[self.free_dofs])    # Calculate the 2-norm of residual vector 
        if res_norm0 < 1e-14:
            res_norm0 = 1.0

        # ----------------------------------------------------
        # Newton-Raphson Algorithm         
        # ----------------------------------------------------
        
        while it < self.maxit:

            it += 1											                # Counter iteration update  

            Ktan = self._assemble_stiffness()                               # Tangent stiffness matrix initialization
            residual = self.Fext - Fint	                                    # Unbalanced forces vector at iteration "it" for the n-step of time    
            res_free = residual[self.free_dofs]                             # Unbalanced forces vector at free dofs only
            # Solve for correction of the displacement increment vector at iteration "it" for the n-step
            du[self.free_dofs] = np.linalg.solve(
                    Ktan[np.ix_(self.free_dofs, self.free_dofs)], 
                    res_free
                )

            self.udisp[self.free_dofs] += du[self.free_dofs]                # Update displacement vector for the n-step of time  

            Fint = self._assemble_internal_forces()                         # Update internal force vector
            residual = self.Fext - Fint                                     # Update unbalanced forces vector at iteration "it" for the n-step of time                   

            # --- Criterios de convergencia ---
            du_norm = np.linalg.norm(du[self.free_dofs])                    # Norm-2. Displacement increment vector for the n-step
            u_norm  = np.linalg.norm(self.udisp[self.free_dofs])            # Norm-2. Displacement vector for the n-step of the time
            res_norm = np.linalg.norm(residual[self.free_dofs])

            err_d.append(du_norm / (u_norm + 1e-12))
            err_f.append(res_norm / res_norm0)

            energy = du.T @ residual                                        # Update energy
            if it == 1:
                energy0 = energy if energy > 1e-14 else 1.0
            err_e.append(energy / energy0)

            if self.verbose:
                print(f"Iter {it:3d} | du error: {err_d[-1]:.2e}  Res error: {err_f[-1]:.2e}  Energetic error: {err_e[-1]:.2e}")

            # --- Check convergence ---
            if (err_d[-1] < self.dtol and 
                err_f[-1] < self.ftol and 
                err_e[-1] < self.etol):
                if self.verbose:
                    print(f"Converged in {it} iterations")
                # Guardar historial
                self.history['load_factor'].append(self.load_factor)
                self.history['displacement_error'].append(err_d[-1])
                self.history['residual_error'].append(err_f[-1])
                self.history['energetic_error'].append(err_e[-1])
                self.history['iterations'].append(it)
                return True

            # --- Protection against divergence ---
            if res_norm > 1e8 * res_norm0:
                if self.verbose:
                    print(f"Newton-Raphson diverged at iteration {it}")
                return False
        else:
            print("MAXIMUM NUMBER OF ITERATIONS REACHED - NO CONVERGENCE")
            return False



# TODO: - for incremental analysis: warm-start from previous step. The NN must calculate displacements increments. Total displacements are given as u_prev + du_nn
#       - for incremental analysis: fine-tunning from previous step. The NN must inicialize weights from previous step and then calculate total displacements u_nn

class feinn(BaseSolver):
    """
    This class implements a finite element-integrated neural network solver
    """

    import torch
    import torch.nn as nn
    from dataclasses import dataclass

    def __init__(self, mesh: object, 
                 bcs: object, 
                 matfld: dict, 
                 body_loads: dict = None, 
                 edge_loads: dict  = None, 
                 nodal_loads: dict  = None,
                 nnet: nn.Module = None,
                 verbose: bool = False):
        
        super().__init__(mesh, bcs, matfld, body_loads, edge_loads, nodal_loads, verbose)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def _to_torch_tensor(np_array: np.ndarray) -> torch.Tensor:
            return torch.tensor(np_array, dtype=torch.float32, device=self.device)
        
        def _to_numpy_array(tensor: torch.Tensor) -> np.ndarray:
            return tensor.cpu().detach().numpy()

        self._to_torch_tensor = _to_torch_tensor
        self._to_numpy_array = _to_numpy_array

        self.coordinates_t = _to_torch_tensor(self.coordinates)
        self.Fext_total_t = _to_torch_tensor(self.Fext_total)
        self.Fext_t = torch.zeros_like(self.Fext_total_t)
        self.udisp_prescribed = np.copy(self.udisp)
        self.udisp_prescribed_t = _to_torch_tensor(self.udisp_prescribed)
        self.free_dofs_list = list(self.free_dofs)
        self.fixed_dofs_list = list(self.fixed_dofs)

        if nnet is None:
            class MLP(nn.Module):
                def __init__(self, layers, activation=nn.Tanh()):
                    super().__init__()
                    self.layers = nn.ModuleList()
                    for i in range(len(layers) - 1):
                        self.layers.append(nn.Linear(layers[i], layers[i + 1]))
                    self.activation = activation

                def forward(self, x):
                    for layer in self.layers[:-1]:
                        x = self.activation(layer(x))
                    return self.layers[-1](x)

            self.nnet = MLP([2, 20, 20, 20, 2]).to(self.device)
            self.nnet.apply(self.init_xavier)
        else:
            self.nnet = nnet.to(self.device)

        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=0.001)
        self.alpha_bc = 1e3  # Weight for soft BC enforcement
        self.dtol = 1e-6
        self.etol = 1e-6
        self.ftol = 1e-6

    @staticmethod
    def init_xavier(m):
        if isinstance(m, nn.Linear) and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            m.bias.data.fill_(0)

    @staticmethod
    def init_he(m):
        if isinstance(m, nn.Linear) and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('relu')
            torch.nn.init.kaiming_uniform_(m.weight, a=g, mode='fan_in', nonlinearity='relu')
            m.bias.data.fill_(0)

    def _assemble_internal_forces(self, u=None):
        if u is not None:
            original_u = self.udisp.copy()
            self.udisp[:] = u
        Fint = super()._assemble_internal_forces()
        if u is not None:
            self.udisp[:] = original_u
        return Fint

    def _forces_residual(self, udisp: np.ndarray) -> torch.Tensor:
        """
        Compute internal forces and residual given displacements.
        """
        Fint = self._assemble_internal_forces(udisp)
        Fint_t = self._to_torch_tensor(Fint)
        return self.Fext_t - Fint_t
    
    def set_load_factor(self, lam: float):
        """
        Set the current load factor λ and update external force vector.
        """
        super().set_load_factor(lam)
        self.Fext_t = lam * self.Fext_total_t

    def update_load_factor(self, dlam: float):
        """
        Update the current load factor λ and external force vector.
        """
        super().update_load_factor(dlam)
        self.Fext_t = self.load_factor * self.Fext_total_t

    def _incremental_step(self):
        self.Fext_t = self.load_factor * self.Fext_total_t
        self.fit(epochs=2000)  # Train for this step
        self.predict()
        # Check convergence (simplified)
        udisp_np = self.udisp
        residual_t = self._forces_residual(udisp_np)
        res_norm = torch.norm(residual_t[self.free_dofs_list])
        fext_norm = torch.norm(self.Fext_t[self.free_dofs_list]) + 1e-12
        err_f = res_norm / fext_norm
        if err_f < self.ftol:
            return True
        else:
            print(f"Warning: Residual error {err_f:.2e} above tolerance")
            return False

    def _train_one_epoch(self):
        """
        Train the neural network for one epoch.
        """
        self.nnet.train()
        self.optimizer.zero_grad()

        udisp_t_2d = self.nnet(self.coordinates_t)  # (nnod, 2)
        udisp_t = udisp_t_2d.flatten()  # (ndof,)
        udisp_np = self._to_numpy_array(udisp_t)

        residual_t = self._forces_residual(udisp_np)
        loss_physics = torch.mean(residual_t[self.free_dofs_list] ** 2)

        loss_bc = torch.mean((udisp_t[self.fixed_dofs_list] - self.udisp_prescribed_t[self.fixed_dofs_list]) ** 2)

        loss = loss_physics + self.alpha_bc * loss_bc
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _validate_one_epoch(self):
        """
        Validate the neural network for one epoch.
        """
        # No validation data available, return dummy
        return 0.0

    def fit(self, epochs: int = 1000, lr: float = 0.001):
        """
        Train the neural network for multiple epochs.
        """
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=lr)
        train_loss_history = []
        val_loss_history = []

        for epoch in range(epochs):
            train_loss = self._train_one_epoch()
            train_loss_history.append(train_loss)

            val_loss = self._validate_one_epoch()
            val_loss_history.append(val_loss)

            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e}")

        return train_loss_history, val_loss_history
    
    def plot_training_history(self, train_loss_history, val_loss_history=None):
        """
        Plot training and validation loss history.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label='Train Loss')
        if val_loss_history:
            plt.plot(val_loss_history, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.yscale('log')
        plt.show()
        
    def predict(self, coordinates: np.ndarray = None) -> np.ndarray:
        """
        Predict displacements for given coordinates using the trained neural network.
        """
        self.nnet.eval()
        with torch.no_grad():
            if coordinates is None:
                coord_t = self.coordinates_t
            else:
                coord_t = self._to_torch_tensor(coordinates)
            udisp_t_2d = self.nnet(coord_t)
            udisp_np = self._to_numpy_array(udisp_t_2d.flatten())
        return udisp_np