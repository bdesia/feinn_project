import numpy as np
import matplotlib.pyplot as plt

class Mesh2D:
    """
    A 2D finite element mesh representation.
    
    Attributes:
        nodes (ndarray): (nnod x 2) array with [x, y] coordinates of nodes.
        elements (ndarray): (nelem x nnode) array with element connectivity (1-based node indices).
        nnod (int): Number of nodes.
        nelem (int): Number of elements.
        nnode (int): Number of nodes per element.
        ndof (int): Total number of degrees of freedom (nnod * ndof_by_node).
        node_groups (dict): Dictionary mapping group names to sets of node indices (1-based).
        element_groups (dict): Dictionary mapping group names to sets of element indices (1-based).
    """
    
    def __init__(self, coordinates, elements, ndof_by_node=2):
        """
        Initialize the mesh with coordinates and connectivity.
        
        Args:
            coordinates (list or ndarray): (nnod x 2) array of node coordinates.
            elements (list or ndarray): (nelem x nnode) array of connectivity (1-based indices).
            ndof_by_node (int, optional): Degrees of freedom per node. Defaults to 2 for 2D.
        """
        self.coordinates = np.array(coordinates, dtype=float)
        self.elements = np.array(elements, dtype=int)
        
        self.nnod = self.coordinates.shape[0]         # Number of nodes
        self.nelem = self.elements.shape[0]     # Number of elements
        self.nnode = self.elements.shape[1]     # Number of nodes by element. Assumes all elements have same nnode
        self.ndof = self.nnod * ndof_by_node    # Total DOFs
        self.node_groups = {}                   # Initialize empty node groups
        self.element_groups = {}                # Initialize empty element groups
        
        # Validation
        if self.coordinates.shape[1] != 2:
            raise ValueError("Coordinates must be 2D (nnod x 2).")
        if np.any(self.elements < 1) or np.any(self.elements > self.nnod):
            raise ValueError("Element indices must be between 1 and nnod.")
    
    def get_node_coords(self, node_id):
        """
        Return coordinates of a specific node (1-based index).
        
        Args:
            node_id (int): Node index (1-based).
            
        Returns:
            ndarray: [x, y] coordinates of the node.
        """
        return self.coordinates[node_id - 1]
    
    def get_element_nodes(self, elem_id):
        """
        Return nodes of a specific element (1-based index).
        
        Args:
            elem_id (int): Element index (1-based).
            
        Returns:
            ndarray: Array of node indices (1-based) for the element.
        """
        return self.elements[elem_id - 1]
    
    def add_node_group(self, group_name, node_indices):
        """
        Add a group of nodes under a specified name.
        
        Args:
            group_name (str): Name of the node group.
            node_indices (list or set): Set of node indices (1-based).
        """
        self.node_groups[group_name] = set(node_indices)
        if not all(1 <= idx <= self.nnod for idx in node_indices):
            raise ValueError("Node indices must be between 1 and nnod.")
    
    def add_element_group(self, group_name, element_indices):
        """
        Add a group of elements under a specified name.
        
        Args:
            group_name (str): Name of the element group.
            element_indices (list or set): Set of element indices (1-based).
        """
        self.element_groups[group_name] = set(element_indices)
        if not all(1 <= idx <= self.nelem for idx in element_indices):
            raise ValueError("Element indices must be between 1 and nelem.")
    
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
        
        # Define element node order for plotting edges (based on Library2DFE.m)
        if self.nnode == 4:
            edge_order = [0, 1, 2, 3, 0]  # Q4: top-right, top-left, bottom-left, bottom-right, close
        elif self.nnode == 8:
            edge_order = [0, 4, 1, 5, 2, 6, 3, 7, 0]  # Q8: top-right, mid-top, top-left, mid-left, ...
        elif self.nnode == 9:
            edge_order = [0, 4, 1, 5, 2, 6, 3, 7, 0]  # Q9: same as Q8 for edges, center node (8) ignored
        else:
            raise ValueError(f"Unsupported number of nodes per element: {self.nnode}")
        
        # Plot elements
        for i in range(self.nelem):
            elem_nodes = self.get_element_nodes(i + 1) - 1  # Convert to 0-based for NumPy
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
        ax.grid(True)
        ax.legend()
        ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
        return ax

class UniformQuadMesh2D(Mesh2D):
    """
    A subclass of Mesh2D for generating uniform quadrilateral meshes in 2D rectangular domains.
    
    Attributes (inherited from Mesh2D, set after compute()):
        nodes (ndarray): (nnod x 2) array with [x, y] coordinates of nodes.
        elements (ndarray): (nelem x nnode) array with element connectivity (1-based indices).
        nnod (int): Number of nodes.
        nelem (int): Number of elements.
        nnode (int): Number of nodes per element (4, 8, or 9).
        ndof (int): Total degrees of freedom (nnod * ndof_by_node).
        node_groups (dict): Dictionary mapping group names to sets of node indices (1-based).
        element_groups (dict): Dictionary mapping group names to sets of element indices (1-based).
    
    Additional Attributes:
        lx (float): Domain length in x-direction.
        ly (float): Length in y-direction.
        nx (int): Number of elements in x-direction.
        ny (int): Number of elements in y-direction.
        element_type (str): Quadrilateral element type ('Q4', 'Q8', 'Q9').
        ndof_by_node (int): Degrees of freedom per node.
    """
    
    def __init__(self, lx, ly, nx, ny, element_type='Q4', ndof_by_node=2):
        """
        Initialize the uniform quadrilateral mesh parameters. Call compute() to generate the mesh.
        
        Args:
            lx (float): Length in x-direction.
            ly (float): Length in y-direction.
            nx (int): Number of elements in x-direction.
            ny (int): Number of elements in y-direction.
            element_type (str, optional): Element type ('Q4', 'Q8', 'Q9'). Defaults to 'Q4'.
            ndof_by_node (int, optional): Degrees of freedom per node. Defaults to 2.
        
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
        self.ndof_by_node = int(ndof_by_node)
        
        # Initialize inherited attributes as None until compute() is called
        self.coordinates = None
        self.elements = None
        self.nnod = 0
        self.nelem = 0
        self.nnode = 0
        self.ndof = 0
        self.node_groups = {}
        self.element_groups = {}
        
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
        # Determine node grid size and element properties
        if self.element_type == 'Q4':
            step = 1
            npx = self.nx + 1  # Node points in x
            npy = self.ny + 1  # Node points in y
            self.nnode = 4
        else:  # Q8 or Q9
            step = 2
            npx = 2 * self.nx + 1
            npy = 2 * self.ny + 1
            self.nnode = 8 if self.element_type == 'Q8' else 9
        
        # Generate node coordinates using meshgrid
        x = np.linspace(0, self.lx, npx)
        y = np.linspace(0, self.ly, npy)
        X, Y = np.meshgrid(x, y)
        self.coordinates = np.column_stack((X.ravel(), Y.ravel()))
        self.nnod = self.coordinates.shape[0]
        
        # Generate element connectivity
        self.nelem = self.nx * self.ny
        elements = np.zeros((self.nelem, self.nnode), dtype=int)
        elem_idx = 0
        for ey in range(self.ny):
            for ex in range(self.nx):
                base_i = ex * step
                base_j = ey * step
                # Corner nodes (1-based)
                bottom_left = base_j * npx + base_i + 1
                bottom_right = base_j * npx + (base_i + step) + 1
                top_left = (base_j + step) * npx + base_i + 1
                top_right = (base_j + step) * npx + (base_i + step) + 1
                
                if self.element_type == 'Q4':
                    elements[elem_idx] = [top_right, top_left, bottom_left, bottom_right]
                
                elif self.element_type in ['Q8', 'Q9']:
                    # Mid-side nodes
                    bottom_mid = base_j * npx + (base_i + step // 2) + 1
                    right_mid = (base_j + step // 2) * npx + (base_i + step) + 1
                    top_mid = (base_j + step) * npx + (base_i + step // 2) + 1
                    left_mid = (base_j + step // 2) * npx + base_i + 1
                    elem = [top_right, top_left, bottom_left, bottom_right,
                            top_mid, left_mid, bottom_mid, right_mid]
                    if self.element_type == 'Q9':
                        center = (base_j + step // 2) * npx + (base_i + step // 2) + 1
                        elem.append(center)
                    elements[elem_idx] = elem
                
                elem_idx += 1
        
        self.elements = elements
        self.ndof = self.nnod * self.ndof_by_node
        
       # Add boundary node groups
        self.node_groups['bottom'] = set(range(1, npx + 1))  # Bottom edge (y=0)
        self.node_groups['top'] = set(range((npy - 1) * npx + 1, npy * npx + 1))  # Top edge
        self.node_groups['left'] = set(range(1, npy * npx + 1, npx))  # Left edge (x=0)
        self.node_groups['right'] = set(range(npx, npy * npx + 1, npx))  # Right edge
        self.element_groups['all'] = set(range(1, self.nelem + 1))  # All elements