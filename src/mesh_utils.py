import numpy as np
import matplotlib.pyplot as plt
from fem_elements import QuadElement

class Mesh2D:
    """
    A 2D finite element mesh representation.
    
    Attributes:
        nodes (ndarray): (nnod x 2) array with [x, y] coordinates of nodes.
        elements (list): List of elements, each element is an Element object
        node_groups (dict): Dictionary mapping group names to sets of node indices (1-based).
        element_groups (dict): Dictionary mapping group names to sets of element indices (1-based).
    """
    
    def __init__(self):
        """
        Initialize the mesh with coordinates and connectivity.
        
        Args:
            coordinates (list or ndarray): (nnod x 2) array of node coordinates.
            elements (list or ndarray): (nelem x nnode) array of connectivity (1-based indices).
            ndof_by_node (int, optional): Degrees of freedom per node. Defaults to 2 for 2D.
        """
        self.coordinates = []
        self.elements = []
        self.nnod = 0
        self.nelem = 0
        self.node_groups = {}                   # Initialize empty node groups
        self.element_groups = {}                # Initialize empty element groups
    
    def add_node(self, x: float, y: float):
        self.coordinates.append(np.array[x, y])

    def add_element(self, element: object):
        self.elements.append(element)

    def get_node_coords(self, node_id: int):
        """
        Return coordinates of a specific node (1-based index).
        
        Args:
            node_id (int): Node index (1-based).
            
        Returns:
            ndarray: [x, y] coordinates of the node.
        """
        return self.coordinates[node_id - 1]
    
    def get_element_nodes(self, elem_id: int):
        """
        Return nodes of a specific element (1-based index).
        
        Args:
            elem_id (int): Element index (1-based).
            
        Returns:
            ndarray: Array of node indices (1-based) for the element.
        """
        if elem_id > 0:
            return self.elements[elem_id - 1].nodes
        else:
            raise ValueError("Element id must be between 1 and nelem.")
    
    def add_node_group(self, group_name: str, node_indices: list):
        """
        Add a group of nodes under a specified name.
        
        Args:
            group_name (str): Name of the node group.
            node_indices (list): Set of node indices (1-based).
        """
        self.node_groups[group_name] = set(node_indices)
        if not all(1 <= idx <= self.nnod for idx in node_indices):
            raise ValueError("Node indices must be between 1 and nnod.")
    
    def add_element_group(self, group_name: str, element_indices: list):
        """
        Add a group of elements under a specified name.
        
        Args:
            group_name (str): Name of the element group.
            element_indices (list): Set of element indices (1-based).
        """
        self.element_groups[group_name] = set(element_indices)
        if not all(1 <= idx <= self.nelem for idx in element_indices):
            raise ValueError("Element indices must be between 1 and nelem.")

    def find_nodes(self, value: float, axis: int, tol = 1e-10):
        """
        Find node IDs where the specified coordinate (x or y) is approximately equal to a given value.

        Args:
            value (float): The coordinate value to search for.
            axis (int): The axis to consider (1 for x-coordinate, 2 for y-coordinate).
            tol (float): Tolerance for matching the coordinate value. Defaults to 1e-10.

        Returns:
        list[int]: List of node IDs (starting from 1) that satisfy the condition.      
        """
        mask = np.abs(self.coordinates[:, axis - 1] - value) < tol
        node_ids = np.where(mask)[0] + 1  
        return node_ids.tolist()

    def plot(self, node_groups_to_plot=None, element_groups_to_plot=None, show_node_ids=False, show_element_ids=False, ax=None):
        """
        Plot the 2D mesh using Matplotlib, with options to highlight node/element groups.
        
        Args:
            node_groups_to_plot (list or None): List of node group names to highlight. None for no groups.
            element_groups_to_plot (list or None): List of element group names to highlight. None for no groups.
            show_node_ids (bool): If True, label nodes with their 1-based indices. Defaults to False.
            show_element_ids (bool): If True, label elements with their 1-based indices. Defaults to False.
            ax (matplotlib.axes.Axes or None): Axes to plot on. If None, creates a new figure.
        
        Returns:
            matplotlib.axes.Axes: The axes object used for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot all nodes
        ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], c='black', s=20, label='Nodes')
        
        # Plot elements
        for i in range(self.nelem):
            elem_nodes = self.get_element_nodes(i + 1) - 1  # Convert to 0-based for NumPy
            edge_order = self.elements[i].edge_order
            x = self.coordinates[elem_nodes[edge_order], 0]
            y = self.coordinates[elem_nodes[edge_order], 1]
            ax.plot(x, y, 'b-', alpha=0.5, label='Elements' if i == 0 else None)
            if show_element_ids:
                # Plot element ID at centroid
                centroid = np.mean(self.coordinates[elem_nodes, :], axis=0)
                ax.text(centroid[0], centroid[1], str(i + 1), color='blue', ha='center', va='center')
        
        # Plot node groups with different colors
        if node_groups_to_plot:
            colors = plt.cm.tab10(np.linspace(0, 1, len(node_groups_to_plot)))
            for group, color in zip(node_groups_to_plot, colors):
                if group in self.node_groups:
                    node_indices = np.array(list(self.node_groups[group])) - 1  # Convert to 0-based
                    ax.scatter(self.coordinates[node_indices, 0], self.coordinates[node_indices, 1], 
                              c=[color], s=50, label=group)
        
        # Plot element groups with different edge colors
        if element_groups_to_plot:
            colors = plt.cm.Set1(np.linspace(0, 1, len(element_groups_to_plot)))
            for group, color in zip(element_groups_to_plot, colors):
                if group in self.element_groups:
                    for i in self.element_groups[group]:
                        elem_nodes = self.get_element_nodes(i) - 1
                        x = self.coordinates[elem_nodes[edge_order], 0]
                        y = self.coordinates[elem_nodes[edge_order], 1]
                        ax.plot(x, y, c=color, alpha=0.8, label=group if i == min(self.element_groups[group]) else None)
        
        # Plot node IDs if requested
        if show_node_ids:
            for i in range(self.nnod):
                ax.text(self.coordinates[i, 0], self.coordinates[i, 1], str(i + 1), color='black', fontsize=8)
        
        # Set plot properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('2D Finite Element Mesh')
        # ax.grid(True)
        ax.legend()
        ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
        return

class UniformQuadMesh2D(Mesh2D):
    """
    A subclass of Mesh2D for generating uniform quadrilateral meshes in 2D rectangular domains.
    
    Attributes (inherited from Mesh2D, set after compute()):
        nodes (ndarray): (nnod x 2) array with [x, y] coordinates of nodes.
        elements (list): (nelem) list with element objetcs
        nnod (int): Number of nodes.
        nelem (int): Number of elements.
        node_groups (dict): Dictionary mapping group names to sets of node indices (1-based).
        element_groups (dict): Dictionary mapping group names to sets of element indices (1-based).
    
    Additional Attributes:
        lx (float): Domain length in x-direction.
        ly (float): Length in y-direction.
        nx (int): Number of elements in x-direction.
        ny (int): Number of elements in y-direction.
        element_type (str): Quadrilateral element type ('Q4', 'Q8', 'Q9').
    """
    
    def __init__(self, lx: float, ly: float, nx: int, ny: int, element_type='Q4'):
        super().__init__()
        """
        Initialize the uniform quadrilateral mesh parameters. Call compute() to generate the mesh.
        
        Args:
            lx (float): Length in x-direction.
            ly (float): Length in y-direction.
            nx (int): Number of elements in x-direction.
            ny (int): Number of elements in y-direction.
            element_type (str, optional): Element type ('Q4', 'Q8', 'Q9'). Defaults to 'Q4'.
        
        Raises:
            ValueError: If element_type is invalid or inputs are non-positive.
        """
        if element_type.upper() not in ['Q4', 'Q8', 'Q9']:
            raise ValueError("element_type must be 'Q4', 'Q8', or 'Q9'.")
        if lx <= 0 or ly <= 0:
            raise ValueError("lx and ly must be positive.")
        if nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive integers.")
        
        self.lx = float(lx)
        self.ly = float(ly)
        self.nx = int(nx)
        self.ny = int(ny)
        self.element_type = element_type.upper()
        
    def compute(self):
        """
        Generate nodes and element connectivity for a uniform quadrilateral mesh.
        Sets the inherited Mesh2D attributes and adds boundary node groups.
        
        Node numbering is row-major (left to right, bottom to top).
        - Q4: top-right, top-left, bottom-left, bottom-right
        - Q8/Q9: top-right, top-left, bottom-left, bottom-right, mid-top, mid-left, mid-bottom, mid-right, [center for Q9]
        
        Returns:
            None
        """

        # -----------------------------------------------
        # NODES GENERATION
        # -----------------------------------------------

        npe = 4 if self.element_type == 'Q4' else 8 if self.element_type == 'Q8' else 9

        step1 = self.lx / self.nx
        step2 = self.ly / self.ny
        a = 1.0
        if npe == 8 or npe == 9:
            step2 = step2 / 2
            a = 0.5

        # x1 = np.arange(0, self.lx + a * step1, a * step1)
        x2 = np.arange(0, self.ly + step2, step2)

        coordinates = []
        k = 1
        for j in x2:
            for i in np.arange(0, self.lx + (step1 * a), step1 * a):
                coordinates.append([i, j])
                k += 1
            if npe == 8:
                a = 1.5 - a  # Alternar entre 0.5 y 1.0 para zigzag

        self.coordinates = np.array(coordinates)
        self.nnod = len(self.coordinates)

        # -----------------------------------------------
        # ELEMENT GENERATION
        # -----------------------------------------------

        aux = 1
        m = 1
        n = 1
        l = 1

        if npe == 4:
            b = 1
            c = 2
        elif npe == 8:
            b = 3
            c = 4
            p = 1
        elif npe == 9:
            b = 4
            c = 4
            p = 2

        self.nelem = self.nx * self.ny

        for h in range(self.nelem):
            connectivity = np.zeros(npe, dtype=int)

            connectivity[0] = b * self.nx + c + m
            connectivity[1] = b * self.nx + m + (c // 2)
            connectivity[2] = m
            connectivity[3] = m + (c // 2)

            if npe != 4:
                connectivity[4] = b * self.nx + c + m - 1
                connectivity[5] = 2 * self.nx + aux + l
                connectivity[6] = m + (c // 2) - 1
                connectivity[7] = 2 * self.nx + aux + l + p
                if npe == 9:
                    connectivity[8] = 2 * self.nx + m + 2

            # Actualización de contadores
            if n == self.nx:
                m = aux + b * self.nx + (c // 2)
                aux = aux + b * self.nx + (c // 2)
                n = 1
                l = 1
            else:
                if npe == 4:
                    m += 1
                else:
                    m += 2
                    if npe == 8:
                        l += 1
                    elif npe == 9:
                        l += 2
                n += 1

            self.elements.append(QuadElement(h+1, connectivity))
            self.elements[h].get_nodal_coordinates(coordinates)

        # -----------------------------------------------
        # GROUP GENERATION
        # -----------------------------------------------
        
        self.node_groups['all']    = list(range(1, self.nnod + 1))
        self.node_groups['bottom'] = find_nodes(0.0, axis=2)
        self.node_groups['top']    = find_nodes(self.ly, axis=2)
        self.node_groups['left']   = find_nodes(0.0, axis=1)
        self.node_groups['right']  = find_nodes(self.lx, axis=1)

        # --- Grupos de nodos ---
        self.element_groups['all'] = list(range(1, self.nelem + 1))         # All elements
        self.element_groups['bottom'] = list(range(1, self.nx + 1))        
        self.element_groups['top'] = list(range(self.nx * (self.ny-1) + 1, self.nx * self.ny + 1))
        self.element_groups['left'] = list(range(1, (self.ny - 1) * self.nx + 2, self.nx))
        self.element_groups['right'] = list(range(self.nx, self.ny * self.nx + 1, self.nx))