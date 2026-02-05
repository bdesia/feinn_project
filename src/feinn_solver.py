import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from fem_elements import LineElement, QuadElement

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
    
    @property
    def tensor(self) -> torch.tensor:
        """Return the load as a Torch tensor [bx, by]."""
        return torch.tensor([self.bx, self.by], dtype=torch.float64)
    
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
    
    @property
    def tensor(self) -> torch.tensor:
        """Return the load as a Torch tensor [ftangential, fnormal]."""
        return torch.tensor([self.ftangential, self.fnormal], dtype=torch.float64)
    
class EdgeLoads(dict):
    """Only accepts EdgeLoad objects."""
    def __setitem__(self, key: str, value: EdgeLoad) -> None:
        if not isinstance(value, EdgeLoad):
            raise TypeError("Value must be an instance of EdgeLoad")
        super().__setitem__(key, value)

# ----------------------------------------------------------------------
# Line Loads classes
# ----------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class LineLoad:
    """
    Line load applied to a specific 1D element.

    Attributes
    ----------
    fx : float
        X-component of the line load (force per unit length)
    fy : float
        Y-component of the line load (force per unit length)
    reference : str
        'local'  - components are given in the element local system
        'global' - components are given in the global X-Y system
    """

    fx: float
    fy: float
    reference: str = "local"

    def __post_init__(self) -> None:
        if self.reference not in {"local", "global"}:
            raise ValueError(
                f"Invalid reference '{self.reference}'. Must be 'local' or 'global'."
            )

    @property
    def vector(self) -> np.ndarray:
        """Return the load as a NumPy array [fx, fy]."""
        return np.array([self.fx, self.fy], dtype=float)
    
    @property
    def tensor(self) -> torch.tensor:
        """Return the load as a Torch tensor [fx, fy]."""
        return torch.tensor([self.fx, self.fy], dtype=torch.float64)
    
class LineLoads(dict):
    """Only accepts LineLoad objects."""
    def __setitem__(self, key: str, value: LineLoad) -> None:
        if not isinstance(value, LineLoad):
            raise TypeError("Value must be an instance of LineLoad")
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
    
    @property
    def tensor(self) -> torch.tensor:
        """Return the load as a Torch tensor [fx, fy]."""
        return torch.tensor([self.fx, self.fy], dtype=torch.float64)
    
class NodalLoads(dict):
    """Only accepts NodalLoad objects."""
    def __setitem__(self, key: str, value: NodalLoad) -> None:
        if not isinstance(value, NodalLoad):
            raise TypeError("Value must be an instance of NodalLoad")
        super().__setitem__(key, value)

# ----------------------------------------------------------------------
# Thickness classes
# ----------------------------------------------------------------------

class Thickness(dict):
    """For defining thickness per element group."""
    def __setitem__(self, key: str, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be a number (int or float)")
        super().__setitem__(key, value)

# ----------------------------------------------------------------------
# Solver classes
# ----------------------------------------------------------------------

class BaseSolver:
    "Base class for different types of solvers for 2D problems"

    def __init__(self,  mesh: object, 
                        bcs: BoundaryConditions, 
                        matfld: dict, 
                        thickness: Thickness = None,
                        body_loads: BodyLoads = None, 
                        edge_loads: EdgeLoads  = None, 
                        line_loads: LineLoads  = None,
                        nodal_loads: NodalLoads  = None,
                        formulation: str = 'infinitesimal',
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
        thickness: dict
            {'group_name': thickness_value, ...}
            Thickness by group of elements.
        body_loads: dict
            {'group_name': BodyLoad(), ...}
            Body loads by group of elements.
        edge_loads: dict
            {'group_name': EdgeLoad(), ...}
            Edge loads by group of elements.
        line_loads: dict
            {'group_name': LineLoad(), ...}
            Line loads by group of elements.
        nodal_loads: dict
            {'group_name': NodalLoad(), ...}
            Nodal loads by group of nodes.   
        """
        self.mesh = mesh
        self.bcs = bcs                          # Diritchlet Boundary conditions. [node_id, dof (1=u,2=v), value]  
        self.matfld = matfld                    # Where elements material properties are assigned
        self.thickness = thickness             # Thickness per element group (if any)
        self.body_loads = body_loads            # Body forces (if any)
        self.edge_loads = edge_loads            # Surface tractions (if any)
        self.line_loads = line_loads            # Line loads (if any)
        self.nodal_loads = nodal_loads          # Concentrated forces (if any)
        self.formulation = formulation          # "infinitesimal" / "TLF"
        self.coordinates = mesh.coordinates
        self.elements = []
        self.line_elements = []

        self.verbose = verbose

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # === LOAD FINITE ELEMENTS ===
        if 'quad' in mesh.elements.keys():
            for element_index, quad_connectivity in enumerate(mesh.elements['quad']):
                self.elements.append(QuadElement(element_index + 1, quad_connectivity))
                self.elements[element_index].get_nodal_coordinates(self.coordinates)

        if 'line' in mesh.elements.keys():
            for element_index, line_connectivity in enumerate(mesh.elements['line']):
                self.line_elements.append(LineElement(element_index + 1, line_connectivity))
                self.line_elements[element_index].get_nodal_coordinates(self.coordinates)
                

        # === MESH ATTRIBUTES ===
        self.nnod = mesh.nnod               # Number of nodes
        self.nelem = mesh.nelem             # Number of elements
        self.ndof = self.nnod * 2           # Total number of DOFs  

        # === APPLY MATERIALS ===
        self._assign_matfld(matfld)
        self._check_material()              # Checks if all elements have an assigned material

        # === APPLY THICKNESS ===
        self._assign_thickness()            # For plane stress problems

        # === INICIALIZE SOLUTION AND SOLVER DATA ===
        self.udisp = torch.zeros(self.ndof, dtype=torch.float64, device=self.device)  # Displacement vector initialization

        # === ASSEMBLE EXTERNAL FORCE VECTOR ===
        self.Fext_total = torch.zeros_like(self.udisp)
        self._apply_body_loads()
        self._apply_edge_loads()
        self._apply_line_loads()
        self._apply_nodal_loads()

        # Current state
        self.load_factor = 0.0                              # Current λ (0 ≤ λ ≤ 1 or more)
        self.Fext = torch.zeros_like(self.udisp)            # Current external load = λ * Fext_total
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
    
    def _assign_thickness(self):
        """Apply thickness to 2D finite elements."""
        if not self.thickness:
            return

        for group_name, value in self.thickness.items():
            if group_name not in self.mesh.element_groups:
                print(f"[thickness] Warning: group '{group_name}' not in mesh")
                continue

            for eid in self.mesh.element_groups[group_name]:
                elem = self.elements[eid - 1]
                elem.thickness = value               # assign thickness to elements

        if self.verbose:
            print(f"[thickness] Applied {len(self.thickness)} thickness groups")

    def _reinit(self):
        self.udisp.zero_()
        self.load_factor = 0.0
        self.Fext.zero_()
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
        for elem in self.elements:
            elem.material = default_material  # Default: 'all'

        for group_name, material in matfld.items():
            if group_name == 'all':
                continue
            if group_name not in self.mesh.element_groups:
                print(f"[matfld] Warning: group '{group_name}' does not exist in mesh")
                continue
            for elem_id in self.mesh.element_groups[group_name]:
                self.elements[elem_id - 1].material = material

        if self.verbose:
            print(f"[matfld] Assigned: {list(matfld.keys())}")
        
    def _check_material(self):
        """
        Checks that every element has a material assigned.
        Raises error if any element has material = None.
        """
        missing = [elem.id for elem in self.elements if elem.material is None]
        if missing:
            raise ValueError(f"Elements without material: {missing}")
        if self.verbose:
            print(f"[matfld] All {self.nelem} elements have assigned material")

    def _get_dirichlet_bc_data(self) -> list[dict]:
        """
        Process all Dirichlet boundary conditions and return a structured list.
        
        Returns
        -------
        list[dict]
            Each dict contains:
            - 'group_name': str
            - 'nodes_0': torch.LongTensor (n_nodes,)           # 0-based node indices
            - 'dofs': torch.LongTensor (n_conditions,)         # global DOFs affected
            - 'values': torch.DoubleTensor (n_conditions,)     # prescribed values
        """
        bc_data_list = []

        for group_name, bc_input in self.bcs.items():
            if group_name not in self.mesh.node_groups:
                if self.verbose:
                    print(f"[BC] Warning: group '{group_name}' not found in mesh.node_groups. Skipping.")
                continue

            # Convert 1-based node IDs → 0-based tensor
            node_ids_1based = self.mesh.node_groups[group_name]
            nodes_0 = torch.tensor(
                [nid - 1 for nid in node_ids_1based],
                dtype=torch.long,
                device=self.device
            )  # shape: (n_nodes_in_group,)

            # Allow single BoundaryCondition or list
            bc_list = bc_input if isinstance(bc_input, list) else [bc_input]

            # Process all BCs in this group
            for bc in bc_list:
                if not isinstance(bc, BoundaryCondition):
                    raise TypeError(f"Expected BoundaryCondition, got {type(bc)}")

                dof_local = bc.dof  # 1 → ux (offset 0), 2 → uy (offset 1)
                if dof_local not in (1, 2):
                    raise ValueError(f"Invalid dof {dof_local}. Must be 1 (ux) or 2 (uy).")

                value = bc.value

                # Vectorized computation of global DOFs
                # global_dof = 2 * node_id + (dof_local - 1)
                offset = dof_local - 1  # 0 for ux, 1 for uy
                dofs = 2 * nodes_0 + offset  # shape: (n_nodes_in_group,)

                # Constant prescribed value repeated for all nodes in group
                values = torch.full_like(dofs, fill_value=value, dtype=torch.float64)

                bc_data_list.append({
                    'group_name': group_name,
                    'nodes_0': nodes_0,                    # same for all BCs in group
                    'dofs': dofs,                          # LongTensor, global DOFs
                    'values': values,                      # DoubleTensor, prescribed values
                })

        # Optional debug info
        if self.verbose and bc_data_list:
            total_conditions = sum(len(item['dofs']) for item in bc_data_list)
            print(f"[BC] Processed {len(bc_data_list)} Dirichlet condition(s) → {total_conditions} total constraint(s)")

        return bc_data_list

    def _apply_dirichlet_bcs(self):
        """
        Apply hard Dirichlet boundary conditions.
        """
        fix = []

        bc_list = self._get_dirichlet_bc_data()

        for bc in bc_list:
            dofs = bc['dofs']           # Prescribed DOF (1: X-DISPLACEMENT, 2: Y-DISPLACEMENT)
            values = bc['values']       # Prescribed value
            
            self.udisp[dofs] = values   # Impose initial boundary condition
            fix.extend(dofs.tolist())   # Record prescribed DOF

        all_dofs = torch.arange(self.ndof, device=self.device)
        fixed_dofs = torch.tensor(fix, dtype=torch.long, device=self.device)
        free_dofs = all_dofs[~torch.isin(all_dofs, fixed_dofs)]

        if self.verbose:
            print(f"[BC] Applied Dirichlet BCs → {len(fixed_dofs)} fixed DOFs")

        return fixed_dofs, free_dofs

    def _apply_body_loads(self):
        """Apply volume (body) loads."""
        if not self.body_loads:
            return

        for group_name, load in self.body_loads.items():
            if group_name not in self.mesh.element_groups:
                print(f"[body_load] Warning: group '{group_name}' not in mesh")
                continue

            for eid in self.mesh.element_groups[group_name]:
                elem = self.elements[eid - 1]
                f_body = elem.compute_body_loads(load.tensor)
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
                elem = self.elements[eid - 1]
                f_edge = elem.compute_edge_load(
                    load.side, load.tensor, load.reference
                )
                dofs = elem.dofs
                self.Fext_total[dofs] += f_edge

        if self.verbose:
            print(f"[edge_load] Applied {len(self.edge_loads)} edge load groups")

    def _apply_line_loads(self):
        """Apply line loads."""
        if not self.line_loads:
            return

        for group_name, load in self.line_loads.items():
            if group_name not in self.mesh.element_groups:
                print(f"[line_load] Warning: group '{group_name}' not in mesh")
                continue

            for eid in self.mesh.element_groups[group_name]:
                elem = self.line_elements[eid - 1]
                f_edge = elem.compute_line_load(
                    load.tensor, load.reference
                )
                dofs = elem.dofs
                self.Fext_total[dofs] += f_edge

        if self.verbose:
            print(f"[line_load] Applied {len(self.line_loads)} line load groups")

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

    def _assemble_internal_forces(self, u: torch.Tensor = None) -> torch.Tensor:
        """
        Assemble global internal force vector
        """

        u = self.udisp if u is None else u
        Fint = torch.zeros(self.ndof, dtype=torch.float64, device=self.device)

        if self.formulation == 'infinitesimal':
            compute_f = lambda elem: elem.compute_linear_intfor(u)
        elif self.formulation == 'TLF':
            compute_f = lambda elem: elem.compute_nonlinear_intfor(u)
        else:
            raise ValueError("Invalid formulation. Must be 'infinitesimal' or 'TLF'.")
        
        for elem in self.elements:
            fint_local = compute_f(elem)           # (ndof_local,)
            dofs = elem.dofs                          # torch.long tensor, shape (ndof_local,)
            Fint.scatter_add_(0, dofs, fint_local)
        
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
            udisp_old = self.udisp.clone().detach()
            attempts += 1
            self.update_load_factor(step_size)
            
            converged = self._incremental_step()
            if converged:
                step_size = min(step_size * 1.5, target_lambda - self.load_factor)  # increase step
                attempts = 0
            else:
                step_size *= 0.5                # decrease step
                self.udisp = udisp_old.clone()  # restore previous displacements
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
            u_nodal = u_nodal.detach().cpu().numpy()
            deformed = coords + scale * u_nodal

            deformed_color = '#1f77b4'
            original_color = 'gray'

            label_def = "Deformed"
            label_orig = "Original"

            if show_nodes:
                # Plot all nodes
                ax.scatter(deformed[:, 0], deformed[:, 1], c=deformed_color, s= 10)

            for elem in self.elements:

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
                        label=label_def if elem is self.elements[0] else "")
                ax.fill(Xd, Yd, color=deformed_color, alpha=0.07,
                        label="" if elem is self.elements[0] else "")

                # Optionally plot original mesh
                if show_original:
                    ax.plot(Xo, Yo, color=original_color, lw=0.8, ls='--', 
                            alpha=original_alpha, label=label_orig if elem is self.elements[0] else "")
            
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

    # def export_resu(
    #     self,
    #     filename: str | PathLike,
    #     nodal_data: Dict[str, Tensor] = {},
    #     elem_data: Dict[str, List[Tensor]] = {},
    #       ):
    #     from meshio import Mesh, read
    #     etype = mesh.etype.meshio_type

    #     msh = Mesh(
    #         points=model.coordinates.cpu().detach(),
    #         cells={etype: model.connectivity.cpu().detach()},
    #         point_data={key: tensor.cpu().detach() for key, tensor in nodal_data.items()},
    #         cell_data={
    #             key: [tensor.cpu().detach() for tensor in tensor_list]
    #             for key, tensor_list in elem_data.items()
    #         },
    #     )
    #     msh.write(filename)

    # export_mesh(model, "result.vtu", elem_data={"rho": [rho[-1]]})

class NFEA(BaseSolver):
    """
    This class implements a nonlinear finite element solver
    """
    def __init__(self,  mesh: object, 
                        bcs: BoundaryConditions, 
                        matfld: dict, 
                        thickness: dict,
                        body_loads: BodyLoads = None, 
                        edge_loads: EdgeLoads  = None, 
                        line_loads: LineLoads  = None,
                        nodal_loads: NodalLoads  = None,
                        formulation: str = 'infinitesimal',
                        verbose: bool = False):
        
        super().__init__(mesh, bcs, matfld, thickness, body_loads, edge_loads, line_loads, nodal_loads, formulation, verbose)
    
        self.dtol = 1e-6        # Displacement tolerance
        self.etol = 1e-6        # Energy tolerance 
        self.ftol = 1e-6        # Force tolerance

        self.maxit = 20         # Maximum number of iterations

        # === APPLY DIRICHLET BOUNDARY CONDITIONS ===
        self.fixed_dofs, self.free_dofs = self._apply_dirichlet_bcs()

    def _assemble_stiffness(self)-> torch.Tensor:
        """
        Assemble global stiffness matrix
        """
        K = torch.zeros(self.ndof, self.ndof, 
                    dtype=torch.float64, device=self.device)

        compute_Ke = (
            lambda e: e.compute_linear_stiff()
            if self.formulation == 'infinitesimal' else
            e.compute_nonlinear_stiff(self.udisp)
        )

        for elem in self.elements:
            Ke = compute_Ke(elem)
            dofs = elem.dofs
            K[dofs[:, None], dofs] += Ke

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
        
        du = torch.zeros_like(self.udisp)               # Displacement increment vector initialization
        Fint = self._assemble_internal_forces()         # Internal force vector initalization
        residual = self.Fext - Fint

        # Norma inicial del residuo (solo dofs libres)
        res_norm0 = torch.norm(residual[self.free_dofs])    # Calculate the 2-norm of residual vector 
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
            du[self.free_dofs] = torch.linalg.solve(
                    Ktan[self.free_dofs][:, self.free_dofs], 
                    res_free
                )

            self.udisp[self.free_dofs] += du[self.free_dofs]                # Update displacement vector for the n-step of time  

            Fint = self._assemble_internal_forces()                         # Update internal force vector
            residual = self.Fext - Fint                                     # Update unbalanced forces vector at iteration "it" for the n-step of time                   

            # --- Criterios de convergencia ---
            du_norm = torch.norm(du[self.free_dofs])                    # Norm-2. Displacement increment vector for the n-step
            u_norm  = torch.norm(self.udisp[self.free_dofs])            # Norm-2. Displacement vector for the n-step of the time
            res_norm = torch.norm(residual[self.free_dofs])

            err_d.append(du_norm / (u_norm + 1e-12))
            err_f.append(res_norm / res_norm0)

            energy = torch.einsum('i, i', du, residual)                                        # Update energy
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



DEFAULT_NNET = torch.nn.Sequential(
    torch.nn.Linear(2, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2)
)   

class FEINN(BaseSolver):
    """
    Finite Element Integrated Neural Network solver for 2D solid mechanics problems.
    Dirichlet boundary conditions enforced SOFTLY via penalty method.
    """
    
    def __init__(self, 
                 mesh: object, 
                 bcs: BoundaryConditions, 
                 matfld: dict,
                 thickness: dict,
                 body_loads: BodyLoads = None, 
                 edge_loads: EdgeLoads = None, 
                 line_loads: LineLoads  = None,
                 nodal_loads: NodalLoads = None,
                 formulation: str = 'infinitesimal',
                 verbose: bool = False,
                 nnet: nn.Module = None,
                 nnet_init: str = None,
                 isData: bool = False,
                 bc_weight: float = 1e4,                # Penalty weight for soft BCs
                 normalize_coords = True,
                 ):

        super().__init__(mesh, bcs, matfld, thickness, body_loads, edge_loads, line_loads, nodal_loads, formulation, verbose)

        self.isData = isData
        
        self.bc_weight = bc_weight
        
        # Neural network
        self.nnet = nnet if nnet is not None else DEFAULT_NNET
        self.nnet = self.nnet.to(self.device)
        self.nnet.double()

        # Weight initialization
        if nnet_init == 'xavier':
            self.nnet.apply(self.init_xavier)
        elif nnet_init == 'he':
            self.nnet.apply(self.init_he)
        elif nnet_init is not None:
            raise ValueError("Invalid initialization method. Choose 'xavier', 'he', or None.")

        # Nodal coordinates
        self.coords_tensor = torch.tensor(
            self.coordinates, dtype=torch.float64, device=self.device)  # (nnod, 2)
        if normalize_coords:
            self._normalize_coords()

        self.loss_fun = nn.MSELoss()
        self.loss_res0 =  self.loss_fun(self.Fext_total, torch.zeros_like(self.Fext_total))
        if self.loss_res0 < 1e-12:
            self.loss_res0 = 1.0
        
        if self.verbose:
            n_params = sum(p.numel() for p in self.nnet.parameters())
            print(f"[FEINN] Initialized with {n_params} trainable parameters")
            print(f"[FEINN] Soft Dirichlet BC enforcement (weight = {self.bc_weight:.1e})")

        # === GET FREE AND FIXED DOFS ===
        self.fixed_dofs, self.free_dofs = self._apply_dirichlet_bcs()

        # === SET L-BFGS OPTIMIZER ===
        self.lbfgs_optimizer = torch.optim.LBFGS(
            self.nnet.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,           # default 1e-7
            tolerance_change=1e-9,
        )

        self.maxit = 100        # Maximum number of iterations
    
        self.dtol = 1e-6        # Displacement tolerance
        self.etol = 1e-6        # Energy tolerance 
        self.ftol = 1e-6        # Force tolerance

    def warmup_zero_displacement(self, epochs=500, lr=1e-3):
        """
        Warmup training to initialize the neural network to output near-zero displacements.
        This helps to start the training from a physically reasonable state.
        """
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=lr)
        X = self.coords_tensor
        
        self.nnet.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            u_pred = self.nnet(X)
            loss_zero = self.loss_fun(u_pred, torch.zeros_like(u_pred))
            loss_zero.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0 and self.verbose:
                print(f"Warmup epoch {epoch+1}/{epochs} - zero loss: {loss_zero.item():.2e}")
        
        print("[FEINN] Warmup completado - salida inicial ≈ 0")

    @staticmethod
    def init_xavier(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    @staticmethod
    def init_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def _normalize_coords(self) -> None:
        # Compute min and max
        min_vals = self.coords_tensor.min(dim=0, keepdim=True).values
        max_vals = self.coords_tensor.max(dim=0, keepdim=True).values
        range_vals = torch.clamp(max_vals - min_vals, min=1e-12)         # Avoid division by zero

        # Normalization between [-1, 1]
        self.coords_tensor = 2.0 * (self.coords_tensor - min_vals) / range_vals - 1.0

    def _forces_residual(self, u: torch.Tensor) -> torch.Tensor:
        Fint = self._assemble_internal_forces(u)
        return self.Fext_total - Fint                 

    def train_one_epoch2(self, model, optimizer, scheduler=None):

        model.train()
        optimizer.zero_grad()
        
        loss_list = []
        tag_keys = []
        
        grad_flow = {'Domain': {}, 'BoundaryConditions': {}}

        # =============================
        # Domain loss (PDE residual)
        # =============================
        X_domain = self.coords_tensor                   # All nodes contribute to residual
        u_pred = model(X_domain)                        # (nnod, 2)              

        res_pred = self._forces_residual(u_pred.reshape(-1))[self.free_dofs]        # Convert u_pred to (ndof,). Retain only free dofs
        res_true = torch.zeros_like(res_pred)

        loss_domain = self.loss_fun(res_pred, res_true) / self.loss_res0
        loss_domain.backward(retain_graph=True)
        
        # Store domain gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_flow['Domain'][name] = param.grad.clone().detach()
        
        loss_list.append(loss_domain.detach())
        tag_keys.append('Domain')

        # =============================
        # Soft Dirichlet BCs
        # =============================
        loss_bc = torch.tensor(0.0, device=self.device)
        bc_data = self._get_dirichlet_bc_data()

        if bc_data:
            nodes = torch.cat([bc['nodes_0'] for bc in bc_data])
            dofs  = torch.cat([bc['dofs']      for bc in bc_data])
            u_true  = torch.cat([bc['values']    for bc in bc_data])

            u_pred_bc = model(self.coords_tensor[nodes])
            u_pred_bc = u_pred_bc[torch.arange(len(u_true)), dofs % 2]

            loss_bc = self.bc_weight * self.loss_fun(u_pred_bc, u_true)
            loss_bc.backward(retain_graph=True)

            # Store BC-only contribution: current grad minus previous (domain) grad
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_bc_only = param.grad.clone().detach() - grad_flow['Domain'].get(name, torch.zeros_like(param.grad))
                    grad_flow['BoundaryConditions'][name] = grad_bc_only

        loss_list.append(loss_bc.detach().clone())
        tag_keys.append('BoundaryConditions')

        # =============================
        # Labelled Data (if available)
        # =============================
        loss_data = torch.tensor(0.0, device=self.device)
        if self.isData:
            # Placeholder: implement if you have supervised data
            pass
        loss_list.append(loss_data)
        tag_keys.append('LabelledData')

        # =============================
        # Total loss
        # =============================
        total_loss = loss_domain + loss_bc + loss_data
        loss_list.append(total_loss.detach())
        tag_keys.append('Overall')
        
        # Optimization step
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        return dict(zip(tag_keys, loss_list)), grad_flow

    def _compute_total_loss(self, model: nn.Module) -> torch.Tensor:
        
        # Domain loss
        X_domain = self.coords_tensor
        u_pred = model(X_domain)

        res_pred = self._forces_residual(u_pred.reshape(-1))[self.free_dofs]
        res_true = torch.zeros_like(res_pred)
        loss_domain = self.loss_fun(res_pred, res_true) / self.loss_res0

        # BC loss
        loss_bc = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        bc_data = self._get_dirichlet_bc_data()
        if bc_data:
            nodes = torch.cat([bc['nodes_0'] for bc in bc_data])
            dofs  = torch.cat([bc['dofs']      for bc in bc_data])
            u_true = torch.cat([bc['values']    for bc in bc_data])

            u_pred_bc = model(self.coords_tensor[nodes])
            u_pred_bc = u_pred_bc[torch.arange(len(u_true)), dofs % 2]
            loss_bc = self.bc_weight * self.loss_fun(u_pred_bc, u_true)

        # Data loss
        loss_data = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        if self.isData:
            pass  #to be implemented

        total_loss = loss_domain + loss_bc + loss_data
        return total_loss, loss_domain, loss_bc, loss_data
        
    def train_one_epoch(self, model, optimizer, scheduler=None):
        
        model.train()
        optimizer.zero_grad()

        total_loss, loss_domain, loss_bc, loss_data = self._compute_total_loss(model)
        total_loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_list = [loss_domain.detach(), loss_bc.detach(), loss_data.detach(), total_loss.detach()]
        tag_keys = ['Domain', 'BoundaryConditions', 'LabelledData', 'Overall']
        
        return dict(zip(tag_keys, loss_list))

    def train(self, epochs: int, 
              optimizer=None, 
              scheduler=None, 
              lbfgs_epochs: int = 0,
              warmup=True,
              verbose=True):
    
        if warmup:
            print("[FEINN] Starting warmup for zero initial displacement")
            self.warmup_zero_displacement(epochs=1000, lr=1e-4)
            # Final prediction
            with torch.no_grad():
                initial_disp = self.nnet(self.coords_tensor)   
                max_disp = initial_disp.abs().max().item()
                print(f"[FEINN] Zero-output init: max |u| inicial = {max_disp:.2e}")
        
        if self.verbose:
            print(f"[FEINN] Starting training – max {epochs} epochs")

        history = {'total': [], 'domain': [], 'bc': [], 'data': []}

        model = self.nnet
        
        # ========================================
        # 1st stage: User or Default optimizer
        # ========================================
        
        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr = 1e-4,
                weight_decay = 0,
                )

        for epoch in range(1, epochs + 1):
            train_results = self.train_one_epoch(model = model, 
                                                            optimizer = optimizer, 
                                                            scheduler = scheduler)

            # Logging
            if verbose and (epoch == 1 or epoch % 500 == 0 or epoch == epochs):
                print(f"\nEpoch {epoch}/{epochs}")
                print(f"Total Loss: {train_results['Overall']:.3e}")
                print(f"  Domain: {train_results['Domain']:.3e}")
                print(f"  BC:     {train_results['BoundaryConditions']:.3e}")
                if self.isData and 'LabelledData' in train_results:
                    print(f"  Data:   {train_results['LabelledData']:.3e}")

            # Store history
            history['total'].append(train_results['Overall'])
            history['domain'].append(train_results['Domain'])
            history['bc'].append(train_results['BoundaryConditions'])
            if self.isData and 'LabelledData' in train_results:
                history['data'].append(train_results['LabelledData'])

        # ========================================
        # 2nd stage: L-BFGS optimizer
        # ========================================
        
        if lbfgs_epochs > 0:
            def closure():
                self.lbfgs_optimizer.zero_grad()
                total_loss, _, _, _ = self._compute_total_loss(model)
                total_loss.backward()
                return total_loss

            for epoch in range(1, lbfgs_epochs + 1):
                loss = self.lbfgs_optimizer.step(closure)
                
                with torch.no_grad():
                    _, loss_domain, loss_bc, loss_data = self._compute_total_loss(model)
                    
                if verbose and (epoch == 1 or epoch % 100 == 0 or epoch == lbfgs_epochs):
                    print(f"\nEpoch {epoch}/{lbfgs_epochs} (L-BFGS)")
                    print(f"Total Loss: {loss.item():.3e}")
                    print(f"  Domain: {loss_domain.item():.3e}")
                    print(f"  BC:     {loss_bc.item():.3e}")

                history['total'].append(loss.item())
                history['domain'].append(loss_domain.item())
                history['bc'].append(loss_bc.item())
                history['data'].append(loss_data.item() if self.isData else 0.0)

        if self.verbose:
            print(f"[FEINN] Training complete – final loss: {train_results['Overall']:.3e}")
        
        with torch.no_grad():
            self.udisp = self.nnet(self.coords_tensor).reshape(-1).detach()

        self.history = {'loss': history}  # Store properly
        
    def plot_history(self, title: str = None):
        """
        Plot loss history during training.
        """
        import matplotlib.pyplot as plt

        # Check if history exists       
        if not hasattr(self, 'history') or 'loss' not in self.history:
            raise ValueError("No model has been trained yet. Execute .train() first.")

        history = self.history['loss']

        epochs = range(1, len(history['total']) + 1)

        plt.figure(figsize=(10, 6))

        plt.plot(epochs, history['total'], label='Overall Loss', color='black', linewidth=2.5)
        plt.plot(epochs, history['domain'], label='Domain', linestyle='--', linewidth=2)
        plt.plot(epochs, history['bc'], label='BoundaryConditions', linestyle='-.', linewidth=2)
        if self.isData and history['data'] and any(history['data']):
            plt.plot(epochs, history['data'], label='LabelledData', linestyle=':', linewidth=2)

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        
        if title is None:
            title = 'Training History - FEINN'
        plt.title(title)

        plt.tight_layout()
        plt.show()
    
    def _incremental_step(self):
        """
        Equivalent to the Newton-Raphson incremental step, but adapted for FEINN.
        Uses L-BFGS optimization to minimize the loss until physical convergence criteria
        are met, similar to displacement, force, and energetic errors in FEM.
        """

        # ----------------------------------------------------
        # Initialization of some variables
        # ----------------------------------------------------

        err_d = [100.0]          # Initialization of displacement error
        err_f = [100.0]          # Initialization of force residual error
        err_e = [100.0]          # Initialization of energetic error

        it = 0                  # Iteration counter

        def closure():
            self.lbfgs_optimizer.zero_grad()
            total_loss, _, _, _ = self._compute_total_loss(self.nnet)
            total_loss.backward()
            return total_loss

        # Initial displacement prediction (flattened for vector ops)
        with torch.no_grad():
            u_old = self.nnet(self.coords_tensor).flatten()

        Fint = self._assemble_internal_forces()         # Internal force vector initialization
        residual = self.Fext - Fint

        # Initial residual norm (only free dofs)
        res_norm0 = torch.norm(residual[self.free_dofs])
        if res_norm0 < 1e-14:
            res_norm0 = 1.0

        # ----------------------------------------------------
        # Training loop
        # ----------------------------------------------------

        while it < self.maxit:

            it += 1											# Counter iteration update

            # Perform one L-BFGS optimization step
            loss = self.lbfgs_optimizer.step(closure)

            # Update displacement and compute increments
            with torch.no_grad():
                u_new = self.nnet(self.coords_tensor).flatten()

            du = u_new - u_old

            self.udisp = u_new                              # Update displacement vector

            Fint = self._assemble_internal_forces()         # Update internal force vector
            residual = self.Fext - Fint                     # Update residual

            # --- Convergence criteria ---
            du_norm = torch.norm(du[self.free_dofs])
            u_norm  = torch.norm(self.udisp[self.free_dofs])
            res_norm = torch.norm(residual[self.free_dofs])

            err_d.append(du_norm / (u_norm + 1e-12))
            err_f.append(res_norm / res_norm0)

            energy = torch.dot(du, residual)                # Energetic criterion
            if it == 1:
                energy0 = energy if abs(energy) > 1e-14 else 1.0
            err_e.append(energy / energy0)

            if self.verbose:
                print(f"Iter {it:3d} | du error: {err_d[-1]:.2e}  Res error: {err_f[-1]:.2e}  Energetic error: {err_e[-1]:.2e}  Loss: {loss.item():.2e}")

            # --- Check convergence ---
            if (err_d[-1] < self.dtol and
                err_f[-1] < self.ftol and
                err_e[-1] < self.etol):
                if self.verbose:
                    print(f"Converged in {it} iterations")
                # Store history (adapted from reference code)
                self.history['load_factor'].append(self.load_factor)
                self.history['displacement_error'].append(err_d[-1])
                self.history['residual_error'].append(err_f[-1])
                self.history['energetic_error'].append(err_e[-1])
                self.history['iterations'].append(it)
                return True

            # --- Protection against divergence ---
            if res_norm > 1e8 * res_norm0:
                if self.verbose:
                    print(f"Optimization diverged at iteration {it}")
                return False

            u_old = u_new  # Update for next iteration

        else:
            print("MAXIMUM NUMBER OF ITERATIONS REACHED - NO CONVERGENCE")
            return False