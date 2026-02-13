
import torch
from typing import Tuple

def LineShapeFunctions(r: torch.tensor, nnode: int) -> torch.tensor:
    """
    1D Segment Lagrangian Finite Element Shape Functions

    Parameters:
        r      : torch.Tensor  (..., )   natural coordinate r ∈ [-1, 1]
        nnode  : int                     2 (SEG2), 3 (SEG3) o 4 (SEG4)

    Returns:
        H      : torch.Tensor  (..., nnode)  shape functions evaluated at (r)
    """
    torch.set_default_dtype(torch.float64)

    if nnode == 2: # Linear element

        #	(-1)	1--------2 (+1)

        H = 0.5 * torch.stack([
            (1 + r),   # N1
            (1 - r),   # N2
        ], dim=-1)

    elif nnode == 3:  # Quadratic element

        #	1----3----2
        #	(-1) (0)  (+1)

        H = torch.stack([
            (r**2 - r) / 2,   # N1
            (r**2 + r) / 2,   # N2
            1 - r**2          # N3
        ], dim=-1)

    elif nnode == 4:  # Cubic element

        #	1----2----3----4
        #	(-1) (-1/3)(1/3)(+1)

        b = 1.0 / 3
        r2 = r**2
        r2_b2 = r2 - b * b

        H = torch.stack([
            -(9 / 16) * (r - 1) * r2_b2,         # N1
            (27 / 16) * (r2 - 1) * (r - b),      # N2
            (27 / 16) * (r2 - 1) * (r + b),      # N3
            (9 / 16) * (r + 1) * r2_b2           # N4
        ], dim=-1)

    else:
        raise ValueError("Unsupported number of nodes. Supported: 2, 3, 4")
    return H

def LineShapeDerivatives(r: torch.Tensor, nnode: int) -> torch.Tensor:
    """
    Derivatives of line shape functions ∂N/∂r

    Returns:
        dHr : torch.Tensor  (..., 1, nnode)
               dHr[..., 0, :] = ∂N/∂r
    """

    torch.set_default_dtype(torch.float64)

    batch_shape = r.shape
    device = r.device

    if nnode == 2:
        dHr = torch.zeros(*batch_shape, 2, device=device)
        dHr[..., 0] = -0.5 * torch.ones_like(r)
        dHr[..., 1] = 0.5 * torch.ones_like(r)

    elif nnode == 3:
        dHr = torch.zeros(*batch_shape, 3, device=device)
        dHr[..., 0] = r - 0.5
        dHr[..., 1] = r + 0.5
        dHr[..., 2] = -2 * r

    elif nnode == 4:
        dHr = torch.zeros(*batch_shape, 4, device=device)

        b = 1.0 / 3
        r2 = r**2

        dHr[..., 0] = - (27 / 16) * r2 + (9 / 8) * r + 1 / 16
        dHr[..., 1] = (81 / 16) * r2 - (9 / 8) * r - 27 / 16
        dHr[..., 2] = (81 / 16) * r2 + (9 / 8) * r - 27 / 16
        dHr[..., 3] = (27 / 16) * r2 + (9 / 8) * r - 1 / 16

    else:
        raise ValueError("Unsupported number of nodes. Supported: 2, 3, 4")

    return dHr

def QuadShapeFunctions(r: torch.tensor, s: torch.tensor, nnode: int) -> torch.tensor:
    """
    2D Quadrilateral Lagrangian Finite Element Shape Functions

    Parameters:
        r      : torch.Tensor  (..., )   natural coordinate r ∈ [-1, 1]
        s      : torch.Tensor  (..., )   natural coordinate s ∈ [-1, 1]
        nnode  : int                    4 (Q4), 8 (Q8) o 9 (Q9)

    Returns:
        H      : torch.Tensor  (..., nnode)  shape functions evaluated at (r,s)
    """
    torch.set_default_dtype(torch.float64)

    if nnode == 4:  # Element Q4

        #	(-1,+1)	2--------1 (+1,+1)
        #			|		 |
        #			|		 |
        #			|		 |
        #	(-1,-1)	3--------4 (+1,-1)

        H = 0.25 * torch.stack([
            (1 + r) * (1 + s),   # N1
            (1 - r) * (1 + s),   # N2
            (1 - r) * (1 - s),   # N3
            (1 + r) * (1 - s)    # N4
        ], dim=-1)

    elif nnode == 8:  # Element Q8

        #			   (0,+1)
        #	(-1,+1)	2----5----1 (+1,+1)
        #			|		  |
        #	(-1,0)	6		  8 (+1,0)
        #			|		  |
        #	(-1,-1)	3----7----4 (+1,-1)
        #			   (0,-1)

        H = torch.empty(*r.shape, 8, device=r.device)

        # Corner nodes
        H[..., 0] = (1 + r) * (1 + s) * (r + s - 1) / 4   # N1 ( 1, 1)
        H[..., 1] = (1 - r) * (1 + s) * (-r + s - 1) / 4  # N2 (-1, 1)
        H[..., 2] = (1 - r) * (1 - s) * (-r - s - 1) / 4  # N3 (-1,-1)
        H[..., 3] = (1 + r) * (1 - s) * (r - s - 1) / 4   # N4 ( 1,-1)

        # Midside nodes
        H[..., 4] = (1 - r**2) * (1 + s) / 2              # N5 top
        H[..., 5] = (1 - r) * (1 - s**2) / 2              # N6 left
        H[..., 6] = (1 - r**2) * (1 - s) / 2              # N7 bottom
        H[..., 7] = (1 + r) * (1 - s**2) / 2              # N8 right

    elif nnode == 9:  # Element Q9

        #			   (0,+1)
        #	(-1,+1)	2----5----1 (+1,+1)
        #			|		  |
        #	(-1,0)	6	 9	  8 (+1,0)
        #			|		  |
        #	(-1,-1)	3----7----4 (+1,-1)
        #			   (0,-1)

        H = torch.empty(*r.shape, 9, device=r.device)

        # Corner nodes
        H[..., 0] =  r * (1 + r) * s * (1 + s) / 4          # ( 1, 1)
        H[..., 1] =  r * (r - 1) * s * (1 + s) / 4          # (-1, 1)
        H[..., 2] =  r * (r -  1) * s * (s - 1) / 4       # (-1,-1)
        H[..., 3] =  r * (1 + r) * s * (s - 1) / 4          # ( 1,-1)

        # Midside nodes (excluded center)
        H[..., 4] = (1 - r**2) * s * (1 + s) / 2            # top
        H[..., 5] = (r - 1) * r * (1 - s**2) / 2            # left
        H[..., 6] = (1 - r**2) * s * (s - 1) / 2            # bottom
        H[..., 7] = r * (1 + r) * (1 - s**2) / 2            # right

        # Center node
        H[..., 8] = (1 - r**2) * (1 - s**2)

    else:
        raise ValueError("Unsupported number of nodes. Supported: 4, 8, 9")
    return H

def QuadShapeDerivatives(r: torch.Tensor, s: torch.Tensor, nnode: int) -> torch.Tensor:
    """
    Derivatives of quadrilateral shape functions ∂N/∂r, ∂N/∂s

    Returns:
        dHrs : torch.Tensor  (..., 2, nnode)
               dHrs[..., 0, :] = ∂N/∂r
               dHrs[..., 1, :] = ∂N/∂s
    """

    torch.set_default_dtype(torch.float64)

    batch_shape = r.shape
    device = r.device

    if nnode == 4:
        dHrs = torch.zeros(*batch_shape, 2, 4, device=device)

        dHrs[..., 0, 0] = (1 + s) / 4      # dN1/dr
        dHrs[..., 1, 0] = (1 + r) / 4      # dN1/ds
        dHrs[..., 0, 1] = -(1 + s) / 4
        dHrs[..., 1, 1] = (1 - r) / 4
        dHrs[..., 0, 2] = -(1 - s) / 4
        dHrs[..., 1, 2] = -(1 - r) / 4
        dHrs[..., 0, 3] = (1 - s) / 4
        dHrs[..., 1, 3] = -(1 + r) / 4

    elif nnode == 8:
        dHrs = torch.zeros(*batch_shape, 2, 8, device=device)

        # Corner node derivatives
        dHrs[..., 0, 0] = (1 + s) * (2*r + s) / 4
        dHrs[..., 1, 0] = (1 + r) * (r + 2*s) / 4

        dHrs[..., 0, 1] = (1 + s) * (2*r - s) / 4
        dHrs[..., 1, 1] = (r - 1) * (r - 2*s) / 4

        dHrs[..., 0, 2] = (1 - s) * (2*r + s) / 4
        dHrs[..., 1, 2] = (1 - r) * (r + 2*s) / 4

        dHrs[..., 0, 3] = (1 - s) * (2*r - s) / 4
        dHrs[..., 1, 3] = ( -1 - r) * (r - 2*s) / 4

        # Midside node derivatives
        dHrs[..., 0, 4] = r * (-1 - s)                      # N5 top
        dHrs[..., 1, 4] = (1 - r**2) / 2

        dHrs[..., 0, 5] = (-1 + s**2) / 2                   # N6 left
        dHrs[..., 1, 5] = (r - 1) * s

        dHrs[..., 0, 6] = r * (s - 1)                       # N7 bottom
        dHrs[..., 1, 6] = (r**2 - 1) / 2

        dHrs[..., 0, 7] = (1 - s**2) / 2                    # N8 right
        dHrs[..., 1, 7] = (-1 - r) * s

    elif nnode == 9:
        dHrs = torch.zeros(*batch_shape, 2, 9, device=device)

        # Corner nodes derivatives
        dHrs[..., 0, 0] = (1 + 2*r) * s * (1 + s) / 4
        dHrs[..., 1, 0] = r * (1 + r) * (1 + 2*s) / 4

        dHrs[..., 0, 1] = (2*r - 1) * s * (1 + s) / 4
        dHrs[..., 1,  1] = (r - 1) * r * (1 + 2*s) / 4

        dHrs[..., 0, 2] = (2*r - 1) * s * (s - 1) / 4
        dHrs[..., 1, 2] = (r - 1) * r * (2*s - 1) / 4

        dHrs[..., 0, 3] = (1 + 2*r) * s * (s - 1) / 4
        dHrs[..., 1, 3] = r * (1 + r) * (2*s - 1) / 4

        # Midside nodes derivatives
        dHrs[..., 0, 4] = -r * s * (1 + s)
        dHrs[..., 1, 4] = (1 - r**2) * (1 + 2*s) / 2

        dHrs[..., 0, 5] = (1 - 2*r) * (s**2 - 1) / 2
        dHrs[..., 1, 5] = r * s * (1 - r)

        dHrs[..., 0, 6] = -r * s * (s - 1)
        dHrs[..., 1, 6] = (1 - r**2) * (2*s - 1) / 2

        dHrs[..., 0, 7] = (1 + 2*r) * (1 - s**2) / 2
        dHrs[..., 1, 7] = -r * s * (1 + r)

        # Center node derivatives
        dHrs[..., 0, 8] = 2*r * (1 - s**2)
        dHrs[..., 1, 8] = 2*s * (1 - r**2)

    else:
        raise ValueError("Unsupported number of nodes. Supported: 4, 8, 9")

    return dHrs  # (..., 2, nnode)

def GaussQuad(ngp: int) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    Gauss Quadrature Data Base (Points and Weigths)

    Parameters:
    ngp : int
        Number of Gauss points (from 1 to 6 is available)

    Returns:
    Ri : torch.tensor
        Tensor with the Gauss point local position. Dimension (1 , ngp)
    Wi : torch.tensor
        Tensor with the Gauss point weigths. Dimension (1 , ngp)
    """

    torch.set_default_dtype(torch.float64)

    if ngp == 1:		    # 1 integration point
        Ri=torch.tensor([0.0000]) # Points
        Wi=torch.tensor([2.0000]) # Weigths
    elif ngp == 2:	        # 2 integration points
        Ri=torch.tensor([0.577350269189626,-0.577350269189626]) # Points
        Wi=torch.tensor([1.0000,1.0000]) # Weigths
    elif ngp == 3:	        # 3 integration points
        Ri=torch.tensor([0.774596669241483,0.0000,-0.774596669241483]) # Points
        Wi=torch.tensor([0.555555555555556,0.888888888888889,0.555555555555556]) # Weigths
    elif ngp == 4:          # 4 integration points
        Ri=torch.tensor([0.861136311594053,0.339981043584856,
            -0.339981043584856,-0.861136311594053]) # Points
        Wi=torch.tensor([0.347854845137454,0.652145154862546,
            0.652145154862546,0.347854845137454]) # Weigths
    elif ngp == 5:          # 5 integration points
        Ri=torch.tensor([0.906179845938664,0.538469310105683,0.0000,
            -0.538469310105683,-0.906179845938664]) # Points
        Wi=torch.tensor([0.236926885056189,0.478628670499366,
            0.568888888888889,0.478628670499366,
                                0.236926885056189]) # Weigths
    elif ngp == 6: # 6 integration points
        Ri=torch.tensor([0.932469514203152,0.661209386466265,
            0.238619186083197,-0.238619186083197,
            -0.661209386466265,-0.932469514203152]) # Points
        Wi=torch.tensor([0.171324492379170,0.360761573048139,
            0.467913934572691,0.467913934572691,
            0.360761573048139,0.171324492379170]) # Weigths
    else:
        raise ValueError("Unsupported number of Gauss Points. Supported: 1 up to 6")
    
    return Ri, Wi
