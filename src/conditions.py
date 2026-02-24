from dataclasses import dataclass, field
from typing import List, Tuple, Union, Iterator
import numpy as np
import torch

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
# MultiPoint Constraints classes
# ----------------------------------------------------------------------

@dataclass
class MultiPointConstraint:
    """
    General linear constraint: Sum(coeff * u_node) + b0 = 0.
    """
    b0: float = 0.0
    # Internal storage: (node, dof_idx, coeff) where dof_idx is 0 (X) or 1 (Y)
    terms: List[Tuple[int, int, float]] = field(default_factory=list)

    def add_term(self, node: int, dof: Union[str, int], coeff: float) -> 'MultiPointConstraint':
        """
        Adds a term to the MPC equation. Supports method chaining.
        dof: 'x' or 1 for X-displacement, 'y' or 2 for Y-displacement.
        """
        if isinstance(dof, str):
            dof_idx = 0 if dof.lower() == 'x' else 1
        elif isinstance(dof, int):
            dof_idx = 0 if dof == 1 else 1
        else:
            raise ValueError("Invalid DOF. Must be 'x', 'y', 1, or 2.")
            
        self.terms.append((node, dof_idx, coeff))
        return self

@dataclass
class MultiPointConstraints:
    """
    Collection of MultiPointConstraint instances.
    """
    constraints: List[MultiPointConstraint] = field(default_factory=list)

    def add(self, mpc: MultiPointConstraint) -> 'MultiPointConstraints':
        """
        Adds an MPC to the collection. Supports method chaining.
        """
        self.constraints.append(mpc)
        return self

    def __iter__(self) -> Iterator[MultiPointConstraint]:
        """Allows direct iteration over the collection."""
        return iter(self.constraints)

    def __len__(self) -> int:
        """Returns the total number of constraints."""
        return len(self.constraints)

# ----------------------------------------------------------------------
# Periodic Boundary Condition classes
# ----------------------------------------------------------------------

@dataclass
class PeriodicBoundaryCondition:
    """
    Generates and updates periodic boundary conditions for an RVE.
    Supports time-varying macroscopic strain histories.
    """
    exx: float
    eyy: float
    gxy: float
    
    left_right_pairs: List[Tuple[int, int]]
    bottom_top_pairs: List[Tuple[int, int]]
    
    delta_x: float 
    delta_y: float

    mpcs: MultiPointConstraints = field(default_factory=MultiPointConstraints, init=False)

    def __post_init__(self):
        """Initializes constraints with starting strains."""
        self._generate_constraints()

    def update_strains(self, new_exx: float, new_eyy: float, new_gxy: float):
        """
        Updates macroscopic strains for a new load step/time increment
        and recalculates the MPCs independent terms (b0).
        """
        self.exx = new_exx
        self.eyy = new_eyy
        self.gxy = new_gxy
        
        # Clear previous constraints and regenerate with new strains
        self.mpcs.constraints.clear()
        self._generate_constraints()

    def _generate_constraints(self):
        """Computes boundary jumps and populates the MPC collection."""
        # Left-Right boundary jumps
        du_x_lr = self.exx * self.delta_x
        dv_y_lr = 0.50 *self.gxy * self.delta_x
        
        for n_left, n_right in self.left_right_pairs:
            self.mpcs.add(MultiPointConstraint(b0=-du_x_lr).add_term(n_right, 'x', 1.0).add_term(n_left, 'x', -1.0))
            self.mpcs.add(MultiPointConstraint(b0=-dv_y_lr).add_term(n_right, 'y', 1.0).add_term(n_left, 'y', -1.0))

        # Bottom-Top boundary jumps
        du_x_bt = 0.50 * self.gxy * self.delta_y
        dv_y_bt = self.eyy * self.delta_y
        
        for n_bottom, n_top in self.bottom_top_pairs:
            self.mpcs.add(MultiPointConstraint(b0=-du_x_bt).add_term(n_top, 'x', 1.0).add_term(n_bottom, 'x', -1.0))
            self.mpcs.add(MultiPointConstraint(b0=-dv_y_bt).add_term(n_top, 'y', 1.0).add_term(n_bottom, 'y', -1.0))

    def get_constraints(self) -> MultiPointConstraints:
        """Returns the current constraint collection."""
        return self.mpcs

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

