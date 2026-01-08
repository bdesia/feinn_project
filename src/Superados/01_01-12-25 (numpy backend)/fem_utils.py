
import numpy as np

def QuadShapeFunctions(r: float, s: float, nnode: int) -> np.ndarray:
    """
    2D Quadrilateral Lagrangian Finite Element Library

    Parameters:
    r : float
        Convective coordinate r of the element
    s : float
        Convective coordinate s of the element
    nnode : int
        Number of nodes in the element (4, 8, or 9)

    Returns:
    H : numpy.ndarray
        Column vector with shape functions (shape: (nnode, 1))
    """
    if nnode == 4:  # Element Q4

        #	(-1,+1)	2--------1 (+1,+1)
        #			|		 |
        #			|		 |
        #			|		 |
        #	(-1,-1)	3--------4 (+1,-1)

        H = np.zeros((4, 1))
        H[0, 0] = (1 + r) * (1 + s) / 4
        H[1, 0] = (1 - r) * (1 + s) / 4
        H[2, 0] = (1 - r) * (1 - s) / 4
        H[3, 0] = (1 + r) * (1 - s) / 4

    elif nnode == 8:  # Element Q8

        #			   (0,+1)
        #	(-1,+1)	2----5----1 (+1,+1)
        #			|		  |
        #	(-1,0)	6		  8 (+1,0)
        #			|		  |
        #	(-1,-1)	3----7----4 (+1,-1)
        #			   (0,-1)

        H = np.zeros((8, 1))
        H[0, 0] = (1 + r) * (1 + s) * (r + s - 1) / 4
        H[1, 0] = (1 - r) * (1 + s) * (-r + s - 1) / 4
        H[2, 0] = (1 - r) * (1 - s) * (-r - s - 1) / 4
        H[3, 0] = (1 + r) * (1 - s) * (r - s - 1) / 4
        H[4, 0] = (1 + s) * (1 + r) * (1 - r) / 2
        H[5, 0] = (1 - r) * (1 + s) * (1 - s) / 2
        H[6, 0] = (1 - s) * (1 + r) * (1 - r) / 2
        H[7, 0] = (1 + r) * (1 + s) * (1 - s) / 2

    elif nnode == 9:  # Element Q9

        #			   (0,+1)
        #	(-1,+1)	2----5----1 (+1,+1)
        #			|		  |
        #	(-1,0)	6	 9	  8 (+1,0)
        #			|		  |
        #	(-1,-1)	3----7----4 (+1,-1)
        #			   (0,-1)

        H = np.zeros((9, 1))
        H[0, 0] = r * (1 + r) * s * (1 + s) / 4
        H[1, 0] = r * (r - 1) * s * (1 + s) / 4
        H[2, 0] = r * (r - 1) * s * (s - 1) / 4
        H[3, 0] = r * (1 + r) * s * (s - 1) / 4
        H[4, 0] = (1 - r) * (1 + r) * s * (1 + s) / 2
        H[5, 0] = (r - 1) * r * (1 - s) * (1 + s) / 2
        H[6, 0] = (r - 1) * (1 + r) * (1 - s) * s / 2
        H[7, 0] = r * (1 + r) * (1 - s) * (1 + s) / 2
        H[8, 0] = (1 - r) * (1 + r) * (1 - s) * (1 + s)

    else:
        raise ValueError("Unsupported number of nodes. Supported: 4, 8, 9")

    return H

def QuadShapeDerivatives(r: float, s: float, nnode: int) -> np.ndarray:
    """
    2D Quadrilateral Lagrangian Finite Element Library

    Parameters:
    r : float
        Convective coordinate r of the element
    s : float
        Convective coordinate s of the element
    nnode : int
        Number of nodes in the element (4, 8, or 9)

    Returns:
    dHrs : numpy.ndarray
        Matrix with derivatives of shape functions w.r.t. r (row 0) and s (row 1) (shape: (2, nnode))

    """
    if nnode == 4:  # Element Q4

        dHrs = np.zeros((2, 4))
        dHrs[0, 0] = (1 + s) / 4
        dHrs[1, 0] = (1 + r) / 4
        dHrs[0, 1] = -(1 + s) / 4
        dHrs[1, 1] = (1 - r) / 4
        dHrs[0, 2] = (s - 1) / 4
        dHrs[1, 2] = (r - 1) / 4
        dHrs[0, 3] = (1 - s) / 4
        dHrs[1, 3] = -(1 + r) / 4

    elif nnode == 8:  # Element Q8

        dHrs = np.zeros((2, 8))
        dHrs[0, 0] = (1 + s) * (2 * r + s) / 4
        dHrs[1, 0] = (1 + r) * (r + 2 * s) / 4
        dHrs[0, 1] = (2 * r - s) * (1 + s) / 4
        dHrs[1, 1] = (r - 1) * (r - 2 * s) / 4
        dHrs[0, 2] = (1 - s) * (2 * r + s) / 4
        dHrs[1, 2] = (1 - r) * (r + 2 * s) / 4
        dHrs[0, 3] = (2 * r - s) * (1 - s) / 4
        dHrs[1, 3] = (-1 - r) * (r - 2 * s) / 4
        dHrs[0, 4] = r * (-1 - s)
        dHrs[1, 4] = (1 - r**2) / 2
        dHrs[0, 5] = (-1 + s**2) / 2
        dHrs[1, 5] = (r - 1) * s
        dHrs[0, 6] = r * (s - 1)
        dHrs[1, 6] = (r**2 - 1) / 2
        dHrs[0, 7] = (1 - s**2) / 2
        dHrs[1, 7] = (-1 - r) * s

    elif nnode == 9:  # Element Q9

        dHrs = np.zeros((2, 9))
        dHrs[0, 0] = (1 + 2 * r) * s * (1 + s) / 4
        dHrs[1, 0] = r * (1 + r) * (1 + 2 * s) / 4
        dHrs[0, 1] = (2 * r - 1) * s * (1 + s) / 4
        dHrs[1, 1] = (r - 1) * r * (1 + 2 * s) / 4
        dHrs[0, 2] = (2 * r - 1) * s * (s - 1) / 4
        dHrs[1, 2] = (r - 1) * r * (2 * s - 1) / 4
        dHrs[0, 3] = (1 + 2 * r) * s * (s - 1) / 4
        dHrs[1, 3] = r * (1 + r) * (2 * s - 1) / 4
        dHrs[0, 4] = -r * s * (1 + s)
        dHrs[1, 4] = (1 - r**2) * (1 + 2 * s) / 2
        dHrs[0, 5] = (1 - 2 * r) * (s**2 - 1) / 2
        dHrs[1, 5] = r * s * (1 - r)
        dHrs[0, 6] = -r * s * (s - 1)
        dHrs[1, 6] = (1 - r**2) * (2 * s - 1) / 2
        dHrs[0, 7] = (1 + 2 * r) * (1 - s**2) / 2
        dHrs[1, 7] = -r * s * (1 + r)
        dHrs[0, 8] = 2 * r * (s**2 - 1)
        dHrs[1, 8] = 2 * (r**2 - 1) * s

    else:
        raise ValueError("Unsupported number of nodes. Supported: 4, 8, 9")

    return dHrs

def GaussQuad(ngp):

    """
    Gauss Quadrature Data Base (Points and Weigths)

    Parameters:
    ngp : int
        Number of Gauss points (from 1 to 6 is available)

    Returns:
    Ri : numpy.ndarray
        Array with the Gauss point local position. Dimension (1 , ngp)
    Wi : numpy.ndarray
        Array with the Gauss point weigths. Dimension (1 , ngp)

    """

    if ngp == 1:		    # 1 integration point
        Ri=[0.0000] # Points
        Wi=[2.0000] # Weigths
    elif ngp == 2:	        # 2 integration points
        Ri=[0.577350269189626,-0.577350269189626] # Points
        Wi=[1.0000,1.0000] # Weigths
    elif ngp == 3:	        # 3 integration points
        Ri=[0.774596669241483,0.0000,-0.774596669241483] # Points
        Wi=[0.555555555555556,0.888888888888889,0.555555555555556] # Weigths
    elif ngp == 4:          # 4 integration points
        Ri=[0.861136311594053,0.339981043584856,
            -0.339981043584856,-0.861136311594053] # Points
        Wi=[0.347854845137454,0.652145154862546,
            0.652145154862546,0.347854845137454] # Weigths
    elif ngp == 5:          # 5 integration points
        Ri=[0.906179845938664,0.538469310105683,0.0000,
            -0.538469310105683,-0.906179845938664] # Points
        Wi=[0.236926885056189,0.478628670499366,
            0.568888888888889,0.478628670499366,
                                0.236926885056189] # Weigths
    elif ngp == 6: # 6 integration points
        Ri=[0.932469514203152,0.661209386466265,
            0.238619186083197,-0.238619186083197,
            -0.661209386466265,-0.932469514203152] # Points
        Wi=[0.171324492379170,0.360761573048139,
            0.467913934572691,0.467913934572691,
            0.360761573048139,0.171324492379170] # Weigths
    else:
        raise ValueError("Unsupported number of Gauss Points. Supported: 1 up to 6")
    
    return Ri, Wi
